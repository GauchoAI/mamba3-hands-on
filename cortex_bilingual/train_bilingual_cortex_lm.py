"""Train a CortexLM on a primitive bilingual English+Spanish corpus.

Uses the same architecture that hosted the counter primitive in
cortex_counting.py — but with primitives=[] so it's a plain language
model. Plug-ready: the same model can be re-instantiated with primitives
attached later for composition experiments.

Saves a checkpoint every `ckpt_every` steps and records a generation
sample at each checkpoint, so progress is visible offline:

    checkpoints/lm/
      step_000500.pt
      step_001000.pt
      ...
      step_FINAL.pt
      training.log     # loss curve + samples at each checkpoint

Run:
    python train_bilingual_cortex_lm.py
    python train_bilingual_cortex_lm.py --steps 20000 --ckpt-every 1000
    python train_bilingual_cortex_lm.py --resume checkpoints/lm/step_005000.pt
"""
from __future__ import annotations
import argparse
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F

from cortex_counting import CortexLM, CortexLMConfig

try:
    from experiment_pusher import ExperimentPusher
    _HAS_PUSHER = True
except ImportError:
    _HAS_PUSHER = False


# ----------------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------------

class ByteCorpus:
    """Byte-level streaming dataset. Random fixed-length chunks."""

    def __init__(self, path: str, seq_len: int, device: str = "cpu"):
        with open(path, "rb") as f:
            self.data = torch.tensor(list(f.read()), dtype=torch.long)
        self.seq_len = seq_len
        self.device = device
        n_bytes = len(self.data)
        print(f"corpus: {n_bytes:,} bytes from {path}", flush=True)
        if n_bytes < seq_len + 1:
            raise ValueError(f"corpus too small ({n_bytes} bytes) for seq_len={seq_len}")

    def get_batch(self, batch_size: int):
        max_start = len(self.data) - self.seq_len - 1
        starts = torch.randint(0, max_start, (batch_size,))
        x = torch.stack([self.data[s : s + self.seq_len] for s in starts])
        y = torch.stack([self.data[s + 1 : s + self.seq_len + 1] for s in starts])
        return x.to(self.device), y.to(self.device)


# ----------------------------------------------------------------------------
# Sampling — fixed prompts so we can compare across checkpoints
# ----------------------------------------------------------------------------

# A mix of English, Spanish, parallel-format, and the cortex unary form
# (so we can later check whether language training broke counter-readiness).
PROBE_PROMPTS = [
    "The cat ",
    "El gato ",
    "Where does ",
    "¿Dónde ",
    "I have ",
    "Tengo ",
    "The book is ",
    "El libro es ",
    "***:",
]

@torch.no_grad()
def sample(model: CortexLM, prompt: str, max_new: int = 80,
           temperature: float = 0.8, top_k: int = 40) -> str:
    model.eval()
    device = next(model.parameters()).device
    prompt_bytes = list(prompt.encode("utf-8"))
    toks = torch.tensor([prompt_bytes], dtype=torch.long, device=device)
    max_ctx = model.cfg.max_seq_len
    for _ in range(max_new):
        ctx = toks[:, -max_ctx:]
        logits = model(ctx)[:, -1] / max(temperature, 1e-5)
        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float("-inf")
        probs = F.softmax(logits, dim=-1)
        nxt = torch.multinomial(probs, 1)
        toks = torch.cat([toks, nxt], dim=1)
    out = bytes(toks[0].tolist()).decode("utf-8", errors="replace")
    model.train()
    return out


def render_samples(model: CortexLM) -> str:
    lines = []
    for p in PROBE_PROMPTS:
        text = sample(model, p, max_new=80, temperature=0.8, top_k=40)
        # show only the continuation, abbreviate
        cont = text[len(p):]
        cont_clean = cont.replace("\n", "↵")
        lines.append(f"  {p!r:>14} → {cont_clean[:80]!r}")
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------

@dataclass
class LMTrainConfig:
    corpus_path: str = "data/bilingual.txt"
    ckpt_dir: str = "checkpoints/lm"
    run_name: str = ""               # for Firebase run_id; defaults to ckpt_dir name
    seq_len: int = 96
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 0.05
    warmup_steps: int = 200
    total_steps: int = 10000
    log_every: int = 100
    ckpt_every: int = 500
    seed: int = 0
    # Model
    n_layers: int = 4
    d_model: int = 128
    d_state: int = 16
    expand: int = 2
    headdim: int = 16


def lr_at(step, warmup, total):
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def train(cfg: LMTrainConfig, resume: str | None = None):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {device}", flush=True)

    torch.manual_seed(cfg.seed)
    ckpt_dir = Path(cfg.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = ckpt_dir / "training.log"

    # Build model — CortexLM with NO primitives is just a plain Mamba LM,
    # but plug-ready: future runs can attach a CounterPrimitive at any layer.
    lm_cfg = CortexLMConfig(
        n_layers=cfg.n_layers,
        d_model=cfg.d_model,
        d_state=cfg.d_state,
        expand=cfg.expand,
        headdim=cfg.headdim,
        vocab_size=256,
        max_seq_len=cfg.seq_len,
        use_counter=False,
    )
    model = CortexLM(lm_cfg).to(device)

    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
        betas=(0.9, 0.95),
    )
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: lr_at(s, cfg.warmup_steps, cfg.total_steps)
    )

    start_step = 1
    if resume:
        sd = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(sd["model"])
        opt.load_state_dict(sd["opt"])
        sched.load_state_dict(sd["sched"])
        start_step = sd["step"] + 1
        print(f"resumed from {resume} @ step {start_step - 1}", flush=True)

    corpus = ByteCorpus(cfg.corpus_path, cfg.seq_len, device=device)
    n_params = sum(p.numel() for p in model.parameters())

    log_lines = [
        f"# bilingual cortex LM training",
        f"# device={device} params={n_params:,} layers={cfg.n_layers} d_model={cfg.d_model}",
        f"# seq_len={cfg.seq_len} batch={cfg.batch_size} lr={cfg.lr} steps={cfg.total_steps}",
        f"# corpus={cfg.corpus_path}",
        "",
    ]
    if not log_path.exists() or not resume:
        log_path.write_text("\n".join(log_lines) + "\n")

    def append_log(text: str):
        with open(log_path, "a") as f:
            f.write(text + "\n")

    # ─── Firebase mirror — fault-tolerant, daemon-thread, never blocks training ───
    pusher = None
    if _HAS_PUSHER:
        kind = Path(__file__).resolve().parent.name        # "cortex_bilingual"
        exp_id = f"{kind}-{time.strftime('%Y-%m-%d')}"
        run_id = cfg.run_name or ckpt_dir.name
        try:
            from dataclasses import asdict
            cfg_dict = asdict(cfg)
        except TypeError:
            cfg_dict = {k: getattr(cfg, k) for k in vars(cfg) if not k.startswith("_")}
        pusher = ExperimentPusher(
            experiment_id=exp_id, run_id=run_id, kind=kind,
            config=cfg_dict, outbox_dir=str(ckpt_dir),
        )
        pusher.declare_experiment(name=kind,
            hypothesis="Bilingual byte-level Mamba-3 LM (en+es Tatoeba "
                       "+5% cortex unary mixin) — host for cortex primitive "
                       "attach experiments.")
        pusher.declare_run(purpose=f"{cfg.n_layers}L d={cfg.d_model} "
                                   f"steps={cfg.total_steps} batch={cfg.batch_size}",
                           gpu=-1)  # MPS, not a CUDA index
        print(f"[firebase] pushing to /experiments/{exp_id}/runs/{run_id}", flush=True)

    def save_ckpt(step: int, tag: str = ""):
        path = ckpt_dir / (f"step_{tag}.pt" if tag else f"step_{step:06d}.pt")
        torch.save({
            "model": model.state_dict(),
            "cfg": lm_cfg,
            "step": step,
            "opt": opt.state_dict(),
            "sched": sched.state_dict(),
        }, path)
        return path

    print(f"params: {n_params:,}  steps: {start_step}..{cfg.total_steps}", flush=True)
    t0 = time.time()
    model.train()
    last_metrics: dict = {}

    for step in range(start_step, cfg.total_steps + 1):
        x, y = corpus.get_batch(cfg.batch_size)
        logits = model(x)
        loss = F.cross_entropy(
            logits.reshape(-1, lm_cfg.vocab_size), y.reshape(-1)
        )
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if step % cfg.log_every == 0 or step == 1:
            elapsed = time.time() - t0
            bpc = loss.item() / math.log(2)
            lr_now = sched.get_last_lr()[0]
            sps = step / max(elapsed, 1e-3)
            line = (f"step {step:6d}/{cfg.total_steps}  "
                    f"loss={loss.item():.3f}  bpc={bpc:.3f}  "
                    f"lr={lr_now:.2e}  elapsed={elapsed:.1f}s")
            print(line, flush=True)
            append_log(line)
            if pusher is not None:
                pusher.metrics(step=step, byte_ce=float(loss.item()),
                               bpc=float(bpc), lr=float(lr_now))
                pusher.heartbeat(step=step, sps=float(sps))
            last_metrics = {"byte_ce": float(loss.item()), "bpc": float(bpc),
                            "lr": float(lr_now)}

        if step % cfg.ckpt_every == 0 or step == cfg.total_steps:
            path = save_ckpt(step)
            samples = render_samples(model)
            block = (f"\n--- step {step} samples (saved {path.name}) ---\n"
                     f"{samples}\n")
            print(block, flush=True)
            append_log(block)
            if pusher is not None:
                # Push the first probe prompt's continuation as the canary —
                # ExperimentPusher throttles internally to ~hourly.
                first = PROBE_PROMPTS[0]
                cont = sample(model, first, max_new=80, temperature=0.8, top_k=40)
                pusher.canary_sample(step=step, prompt=first,
                                     completion=cont[len(first):])

    final_path = save_ckpt(cfg.total_steps, tag="FINAL")
    print(f"\nDone. Final checkpoint: {final_path}", flush=True)
    print(f"Training log: {log_path}", flush=True)
    if pusher is not None:
        pusher.event(type="run_complete", step=cfg.total_steps,
                     details=f"reached step {cfg.total_steps}")
        pusher.complete(final_state="completed", final_metrics=last_metrics)


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/bilingual.txt")
    ap.add_argument("--ckpt-dir", default="checkpoints/lm")
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--seq-len", type=int, default=96)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--n-layers", type=int, default=4)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--ckpt-every", type=int, default=500)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run-name", default="",
                    help="Firebase run_id; defaults to ckpt-dir basename")
    args = ap.parse_args()

    cfg = LMTrainConfig(
        corpus_path=args.corpus,
        ckpt_dir=args.ckpt_dir,
        run_name=args.run_name,
        total_steps=args.steps,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        lr=args.lr,
        n_layers=args.n_layers,
        d_model=args.d_model,
        ckpt_every=args.ckpt_every,
        log_every=args.log_every,
        seed=args.seed,
    )
    train(cfg, resume=args.resume)


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    main()
