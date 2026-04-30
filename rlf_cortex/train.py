"""Joint trainer: byte CE + JEPA + SIGreg + primitive aux.

W3.1 of the JEPA-Cortex plan. Built on the existing PyTorch CortexLM
(cortex_counting.py) — no MLX yet because SIGreg's KS-quantile op is
cleanest on top of torch.special.erfinv. A port comes after Gate 3.2
passes if we want it.

Each step samples one of three batch kinds with the configured mixing
weights:

  teacher  — JEPT records: byte CE + JEPA thought regression + SIGreg
  biling   — bilingual.txt: byte CE only (safety belt against degradation)
  unary    — synthetic ***:aaa\\n: byte CE + counter aux (keeps primitive sharp)

The model object is a vanilla CortexLM with two extra modules wrapped on
top: ThoughtHead, and (implicitly via SIGreg) the intent-pool path that
already lives in CortexLM.forward(return_jepa=True).

Designed for 4-card vast.ai fanout: pin with CUDA_VISIBLE_DEVICES at the
shell level, vary --run-name and the loss weights, point AsyncCheckpointer
at a per-GPU subdir.
"""
from __future__ import annotations
import argparse
import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from cortex_counting import CortexLM, CortexLMConfig
from arch import ThoughtHead, jepa_loss, sigreg_loss
from data_loader import (
    TeacherThoughtsDataset, TeacherIterator, BilingualByteIterator,
    CountingByteIterator, Batch,
)
from checkpoint import AsyncCheckpointer, capture_rng

# Cross-experiment Firebase pusher (lives at the repo root). Optional —
# trainer runs without it if the import fails or PUSH_FIREBASE != "1".
try:
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from experiment_pusher import ExperimentPusher
    _HAS_PUSHER = True
except ImportError:
    _HAS_PUSHER = False


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class TrainConfig:
    # I/O
    teacher_path:    str = "data/teacher_thoughts"
    bilingual_path:  str = "data/bilingual.txt"
    ckpt_root:       str = "checkpoints/rlf_cortex"
    runs_root:       str = "runs/rlf_cortex"
    run_name:        str = "default"

    # Model (CortexLMConfig fields, repeated so the trainer config is
    # the single source of truth saved into checkpoints)
    n_layers:                  int = 4
    d_model:                   int = 192
    d_state:                   int = 16
    expand:                    int = 2
    headdim:                   int = 16
    vocab_size:                int = 256
    max_seq_len:               int = 512
    use_counter:               bool = True
    n_counters:                int = 2
    counter_layer:             int = 0
    counter_readout:           str = "unbounded"
    counter_head_bias:         bool = True
    counter_injection_scale:   float = 1.0
    n_loops:                   int = 1

    # Teacher dim — set automatically from data on first batch if 0.
    d_teacher: int = 0
    stride_bytes: int = 16     # must match what make_teacher_thoughts.py used

    # Mixing
    mix_teacher: float = 0.70
    mix_biling:  float = 0.25
    mix_unary:   float = 0.05

    # Loss weights
    lambda_jepa:   float = 1.0
    lambda_sigreg: float = 0.1
    lambda_aux:    float = 0.5

    # Schedule
    steps:       int = 30_000
    batch_size:  int = 64
    seq_len:     int = 512
    lr:          float = 3e-4
    warmup:      int = 1000
    grad_clip:   float = 1.0
    jepa_warmup: int = 2000      # λ_jepa ramps from 0 to its target over warmup
                                  # steps starting at this step

    # Checkpointing
    light_every:  int = 50
    heavy_every:  int = 500
    sample_every: int = 100      # write canary samples to runs/<run>/samples.jsonl

    device: str = "cuda"
    seed:   int = 0

    # Resume
    resume_from: str = ""


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------
def lr_at(step: int, base_lr: float, warmup: int, total: int) -> float:
    if step < warmup:
        return base_lr * (step + 1) / warmup
    # Cosine decay to 10% of base after total steps.
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return base_lr * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress)))


def jepa_weight_at(step: int, target: float, jepa_warmup: int,
                   ramp_steps: int = 1000) -> float:
    """Hold λ_jepa = 0 for the first jepa_warmup steps, then ramp linearly."""
    if step < jepa_warmup:
        return 0.0
    return target * min(1.0, (step - jepa_warmup) / ramp_steps)


# ---------------------------------------------------------------------------
# Per-batch loss
# ---------------------------------------------------------------------------
def byte_ce_loss(logits: torch.Tensor, tokens: torch.Tensor,
                 mask: torch.Tensor) -> torch.Tensor:
    """Standard next-byte CE. logits/tokens/mask are full-length; we shift inside."""
    B, L, V = logits.shape
    pred = logits[:, :-1].reshape(-1, V)
    tgt = tokens[:, 1:].reshape(-1)
    m = mask[:, 1:].reshape(-1).float()
    raw = F.cross_entropy(pred, tgt, reduction="none")
    return (raw * m).sum() / m.sum().clamp_min(1.0)


# ---------------------------------------------------------------------------
# Iterator selection
# ---------------------------------------------------------------------------
class MixedIterator:
    """Round-by-RNG sampler from teacher / biling / unary iterators."""

    def __init__(self, cfg: TrainConfig, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        ds = TeacherThoughtsDataset(cfg.teacher_path)
        self._teacher = TeacherIterator(ds, cfg.batch_size,
                                        max_bytes=cfg.seq_len,
                                        seed=int(rng.integers(0, 2**31)))
        self._biling = BilingualByteIterator(cfg.bilingual_path,
                                             cfg.batch_size,
                                             seq_len=cfg.seq_len,
                                             seed=int(rng.integers(0, 2**31)))
        self._unary = CountingByteIterator(cfg.batch_size,
                                           seed=int(rng.integers(0, 2**31)))
        self._weights = np.array([cfg.mix_teacher, cfg.mix_biling,
                                  cfg.mix_unary], dtype=np.float64)
        self._weights /= self._weights.sum()

    def __next__(self) -> tuple[str, Batch]:
        kind = self.rng.choice(["teacher", "biling", "unary"], p=self._weights)
        it = {"teacher": self._teacher, "biling": self._biling,
              "unary": self._unary}[kind]
        return kind, next(it)


# ---------------------------------------------------------------------------
# Canary samples
# ---------------------------------------------------------------------------
CANARY_PROMPTS_BYTES: list[bytes] = [
    b"Hello, how are you?\n",
    b"Hola, como estas?\n",
    b"Count from 1 to 12, one per line.\n",
    b"Cuenta de 1 a 12, uno por linea.\n",
    b"The cat sat on the\n",
    b"En un lugar de la Mancha\n",
    b"***:",                     # raw unary counting prompt, n=3
    b"************:",            # n=12
]


@torch.no_grad()
def write_canary(model: CortexLM, run_dir: Path, step: int) -> None:
    samples = []
    for prompt in CANARY_PROMPTS_BYTES:
        out = model.generate_greedy(list(prompt), max_new=80)
        samples.append({
            "prompt": prompt.decode("utf-8", errors="replace"),
            "out": bytes(out).decode("utf-8", errors="replace"),
        })
    line = json.dumps({"step": step, "samples": samples})
    (run_dir / "samples.jsonl").open("a").write(line + "\n")


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
def train(cfg: TrainConfig) -> None:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = torch.device(cfg.device)
    run_dir = Path(cfg.runs_root) / cfg.run_name
    ckpt_dir = Path(cfg.ckpt_root) / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Probe the dataset for D_teacher.
    if cfg.d_teacher == 0:
        ds = TeacherThoughtsDataset(cfg.teacher_path)
        cfg.d_teacher = int(ds[0].thoughts.shape[1])
        print(f"[init] inferred d_teacher = {cfg.d_teacher}", flush=True)

    # Build model + thought head.
    model_cfg = CortexLMConfig(
        n_layers=cfg.n_layers, d_model=cfg.d_model, d_state=cfg.d_state,
        expand=cfg.expand, headdim=cfg.headdim,
        vocab_size=cfg.vocab_size, max_seq_len=cfg.max_seq_len,
        use_counter=cfg.use_counter, n_counters=cfg.n_counters,
        counter_layer=cfg.counter_layer, counter_readout=cfg.counter_readout,
        counter_head_bias=cfg.counter_head_bias,
        counter_injection_scale=cfg.counter_injection_scale,
        n_loops=cfg.n_loops,
    )
    model = CortexLM(model_cfg).to(device)
    thought_head = ThoughtHead(cfg.d_model, cfg.d_teacher).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_th_params = sum(p.numel() for p in thought_head.parameters())
    print(f"[init] cortex={n_params:,} thought_head={n_th_params:,}",
          flush=True)

    # One optimizer over the union of params.
    opt = torch.optim.AdamW(
        list(model.parameters()) + list(thought_head.parameters()),
        lr=cfg.lr, betas=(0.9, 0.95), weight_decay=0.1,
    )

    # Mixed-precision: 4070 Ti has good BF16 throughput; weights stay fp32.
    scaler = None  # bf16 doesn't need a scaler

    rng = np.random.default_rng(cfg.seed)
    iterator = MixedIterator(cfg, rng)
    ckpt = AsyncCheckpointer(ckpt_dir)
    cfg_dict = asdict(cfg)

    # Firebase mirror — always on. The pusher itself is fault-tolerant:
    # network failures spool to a local JSONL outbox so we never lose
    # data, and the worker is a daemon thread so it never blocks training.
    # `kind` is auto-derived from this file's parent folder (jepa, rlf_cortex,
    # ...) — no env var needed.
    pusher = None
    if _HAS_PUSHER:
        kind = Path(__file__).resolve().parent.name        # "rlf_cortex"
        exp_id = f"{kind}-{time.strftime('%Y-%m-%d')}"
        pusher = ExperimentPusher(
            experiment_id=exp_id, run_id=cfg.run_name, kind=kind,
            config=cfg_dict, outbox_dir=run_dir,
        )
        pusher.declare_experiment(name=kind, hypothesis="")
        try:
            gpu = int(os.environ.get("CUDA_VISIBLE_DEVICES", "-1") or "-1")
        except ValueError:
            gpu = -1
        pusher.declare_run(purpose=cfg.run_name, gpu=gpu)
        print(f"[firebase] pushing to /experiments/{exp_id}/runs/{cfg.run_name}",
              flush=True)

    start_step = 0
    if cfg.resume_from:
        payload = torch.load(cfg.resume_from, map_location="cpu",
                             weights_only=False)
        model.load_state_dict(payload["model"], strict=False)
        if "optimizer" in payload and payload["optimizer"] is not None:
            opt.load_state_dict(payload["optimizer"])
        start_step = int(payload.get("step", 0))
        print(f"[resume] from step {start_step}", flush=True)

    last_metrics: dict = {}
    t0 = time.time()

    for step in range(start_step, cfg.steps):
        # LR & λ_jepa schedules
        lr = lr_at(step, cfg.lr, cfg.warmup, cfg.steps)
        for pg in opt.param_groups:
            pg["lr"] = lr
        jw = jepa_weight_at(step, cfg.lambda_jepa, cfg.jepa_warmup)

        kind, batch = next(iterator)
        tokens = batch.tokens.to(device)
        byte_pad = batch.byte_pad_mask.to(device)

        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
            if kind == "teacher":
                plens = batch.prompt_lens.to(device)
                logits, prim_out, residual, intent = model(
                    tokens, return_jepa=True, prompt_lens=plens,
                )
                student_thoughts = thought_head(residual)
                l_byte = byte_ce_loss(logits, tokens, byte_pad)
                l_jepa = jepa_loss(
                    student_thoughts.float(),
                    batch.teacher_thoughts.to(device).float(),
                    batch.thought_byte_pos.to(device),
                    batch.thought_pad_mask.to(device),
                    stride_bytes=cfg.stride_bytes,
                )
                l_sig = sigreg_loss(intent.float())
                # Aux loss only if a primitive in this batch has byte targets
                # to ground it. The teacher batches don't include the unary
                # format; primitive aux only fires on unary batches.
                l_aux = torch.zeros((), device=device)
                loss = l_byte + jw * l_jepa + cfg.lambda_sigreg * l_sig
            elif kind == "biling":
                logits = model(tokens)
                l_byte = byte_ce_loss(logits, tokens, byte_pad)
                l_jepa = torch.zeros((), device=device)
                l_sig = torch.zeros((), device=device)
                l_aux = torch.zeros((), device=device)
                loss = l_byte
            else:  # unary
                logits, prim_out = model(tokens, return_aux=True)
                l_byte = byte_ce_loss(logits, tokens, byte_pad)
                l_aux = model.aux_loss(tokens, prim_out, byte_pad)
                l_jepa = torch.zeros((), device=device)
                l_sig = torch.zeros((), device=device)
                loss = l_byte + cfg.lambda_aux * l_aux

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(thought_head.parameters()),
            cfg.grad_clip,
        )
        opt.step()

        # ----- logging --------------------------------------------------
        last_metrics = {
            "kind": kind,
            "lr": lr,
            "jepa_w": jw,
            "loss": float(loss.detach()),
            "l_byte": float(l_byte.detach()),
            "l_jepa": float(l_jepa.detach()),
            "l_sig": float(l_sig.detach()),
            "l_aux": float(l_aux.detach()),
        }
        if step % 50 == 0:
            dt = time.time() - t0
            sps = (step - start_step + 1) / max(dt, 1e-6)
            print(f"step={step:6d} kind={kind:7s} loss={last_metrics['loss']:.4f} "
                  f"byte={last_metrics['l_byte']:.4f} "
                  f"jepa={last_metrics['l_jepa']:.4f} (w={jw:.2f}) "
                  f"sig={last_metrics['l_sig']:.4f} "
                  f"aux={last_metrics['l_aux']:.4f} "
                  f"lr={lr:.2e} sps={sps:.1f}", flush=True)
            (run_dir / "loss.jsonl").open("a").write(
                json.dumps({"step": step, **last_metrics}) + "\n"
            )
            if pusher is not None:
                pusher.metrics(step=step, **{
                    k: last_metrics[k] for k in
                    ("loss", "l_byte", "l_jepa", "l_sig", "l_aux", "lr", "jepa_w")
                })
                pusher.heartbeat(step=step, sps=sps)

        # ----- checkpoints ---------------------------------------------
        if step % cfg.light_every == 0 and step > 0:
            ckpt.submit_light(step, model, last_metrics, cfg_dict, cfg.run_name)
        if step % cfg.heavy_every == 0 and step > 0:
            ckpt.submit_heavy(
                step, model, opt, capture_rng(), None,
                last_metrics, cfg_dict, cfg.run_name,
            )
            # Also try-and-promote-best on byte loss; eval_daemon has a
            # better held-out estimate but in-loop best is a safety net.
            ckpt.maybe_submit_best(
                step, model, last_metrics["l_byte"], "l_byte_train",
                last_metrics, cfg_dict, cfg.run_name, lower_is_better=True,
            )
        if step % cfg.sample_every == 0:
            model.eval()
            try:
                write_canary(model, run_dir, step)
                if pusher is not None:
                    try:
                        last = json.loads(
                            (run_dir / "samples.jsonl").read_text().strip().splitlines()[-1]
                        )
                        if last["samples"]:
                            s = last["samples"][0]
                            pusher.canary_sample(step=step,
                                                 prompt=s["prompt"],
                                                 completion=s["out"])
                    except (FileNotFoundError, IndexError, json.JSONDecodeError):
                        pass
            finally:
                model.train()

    ckpt.submit_heavy(cfg.steps, model, opt, capture_rng(), None,
                      last_metrics, cfg_dict, cfg.run_name)
    ckpt.flush(timeout=60.0)
    if pusher is not None:
        pusher.event(type="run_complete", step=cfg.steps,
                     details=f"reached step {cfg.steps}")
        pusher.complete(final_state="completed", final_metrics=last_metrics)
    print("[done]", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default="default")
    ap.add_argument("--steps", type=int, default=TrainConfig.steps)
    ap.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    ap.add_argument("--seq-len", type=int, default=TrainConfig.seq_len)
    ap.add_argument("--lr", type=float, default=TrainConfig.lr)

    ap.add_argument("--lambda-jepa", type=float, default=TrainConfig.lambda_jepa)
    ap.add_argument("--lambda-sigreg", type=float, default=TrainConfig.lambda_sigreg)
    ap.add_argument("--lambda-aux", type=float, default=TrainConfig.lambda_aux)

    ap.add_argument("--mix-teacher", type=float, default=TrainConfig.mix_teacher)
    ap.add_argument("--mix-biling", type=float, default=TrainConfig.mix_biling)
    ap.add_argument("--mix-unary", type=float, default=TrainConfig.mix_unary)

    ap.add_argument("--teacher-path", default=TrainConfig.teacher_path)
    ap.add_argument("--bilingual-path", default=TrainConfig.bilingual_path)
    ap.add_argument("--ckpt-root", default=TrainConfig.ckpt_root)
    ap.add_argument("--runs-root", default=TrainConfig.runs_root)

    ap.add_argument("--d-model", type=int, default=TrainConfig.d_model)
    ap.add_argument("--n-layers", type=int, default=TrainConfig.n_layers)

    ap.add_argument("--use-counter", type=str, default="true",
                    choices=["true", "false"],
                    help="If false, no CounterPrimitive is built; aux loss "
                         "path returns 0. Use for the no-cortex variant.")
    ap.add_argument("--n-loops", type=int, default=1,
                    help="RLF-inspired layer recursion. n=1 (default) is "
                         "standard behavior. n>1 re-runs the SSM stack n "
                         "times per token with a decayed lifeline of the "
                         "original embedding re-injected each loop.")
    ap.add_argument("--override-stride-bytes", type=int,
                    default=TrainConfig.stride_bytes,
                    help="Must match the stride used to make teacher_thoughts")
    ap.add_argument("--device", default=TrainConfig.device)
    ap.add_argument("--seed", type=int, default=TrainConfig.seed)
    ap.add_argument("--resume-from", default="")
    args = ap.parse_args()

    cfg = TrainConfig(
        run_name=args.run_name,
        steps=args.steps,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        lr=args.lr,
        lambda_jepa=args.lambda_jepa,
        lambda_sigreg=args.lambda_sigreg,
        lambda_aux=args.lambda_aux,
        mix_teacher=args.mix_teacher,
        mix_biling=args.mix_biling,
        mix_unary=args.mix_unary,
        teacher_path=args.teacher_path,
        bilingual_path=args.bilingual_path,
        ckpt_root=args.ckpt_root,
        runs_root=args.runs_root,
        d_model=args.d_model,
        n_layers=args.n_layers,
        use_counter=(args.use_counter == "true"),
        n_loops=args.n_loops,
        stride_bytes=args.override_stride_bytes,
        device=args.device,
        seed=args.seed,
        resume_from=args.resume_from,
    )
    train(cfg)


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    main()
