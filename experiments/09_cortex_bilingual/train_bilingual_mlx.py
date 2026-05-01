"""MLX training loop for CortexLM on bilingual byte data.

Mirrors train_bilingual_cortex_lm.py exactly: same hyperparameters,
same checkpoint cadence, same probe prompts, same training.log
format. Only the framework differs (MLX vs PyTorch).

Run:
    python train_bilingual_mlx.py
    python train_bilingual_mlx.py --steps 2000 --ckpt-every 250
    python train_bilingual_mlx.py --corpus data/opensubtitles.txt --ckpt-dir checkpoints/lm_mlx_opensubs

Outputs identical to the PyTorch trainer: checkpoint .npz files plus
checkpoints/lm_mlx/training.log accumulating loss curve + per-checkpoint
samples at fixed probe prompts.
"""
from __future__ import annotations
import argparse
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# Allow `from mamba3_mlx import ...` from sibling file when run anywhere.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Allow `from lab_platform.experiment_pusher import ...` from repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import mlx.core as mx
import mlx.nn as mx_nn
import mlx.optimizers as optim
import mlx.utils

from mamba3_mlx import CortexLM, CortexLMConfig

try:
    from lab_platform.experiment_pusher import ExperimentPusher
    _HAS_PUSHER = True
except ImportError:
    _HAS_PUSHER = False

try:
    from lab_platform.cloud_archive import CloudArchive
    _HAS_ARCHIVE = True
except ImportError:
    _HAS_ARCHIVE = False


# ----------------------------------------------------------------------------
# Dataset — fixed-window memmap on the byte tensor
# ----------------------------------------------------------------------------

class ByteCorpus:
    def __init__(self, path: str, seq_len: int):
        self.data = np.frombuffer(Path(path).read_bytes(), dtype=np.uint8)
        self.seq_len = seq_len
        if len(self.data) < seq_len + 1:
            raise ValueError(f"corpus too small: {len(self.data)} bytes")
        print(f"corpus: {len(self.data):,} bytes from {path}", flush=True)

    def get_batch(self, batch_size: int, rng: np.random.Generator):
        max_start = len(self.data) - self.seq_len - 1
        starts = rng.integers(0, max_start, size=batch_size)
        xs = np.stack([self.data[s : s + self.seq_len] for s in starts]).astype(np.int32)
        ys = np.stack([self.data[s + 1 : s + self.seq_len + 1] for s in starts]).astype(np.int32)
        return mx.array(xs), mx.array(ys)


# ----------------------------------------------------------------------------
# Probe prompts (same as PyTorch trainer for direct comparison)
# ----------------------------------------------------------------------------

PROBE_PROMPTS = [
    "The cat ", "El gato ", "Where does ", "¿Dónde ",
    "I have ", "Tengo ", "The book is ", "El libro es ",
    "***:",
]


def sample(model: CortexLM, prompt: str, max_new: int = 80,
           temperature: float = 0.8, top_k: int = 40) -> str:
    """Greedy + top-k sampling for evaluation."""
    prompt_bytes = list(prompt.encode("utf-8"))
    toks_np = np.array(prompt_bytes, dtype=np.int32)[None, :]
    toks = mx.array(toks_np)
    max_ctx = model.cfg.max_seq_len
    for _ in range(max_new):
        ctx = toks[:, -max_ctx:]
        logits = model(ctx)
        last = logits[:, -1] / max(temperature, 1e-5)
        if top_k > 0:
            # top-k filter (mx scalars broadcast in `where`)
            v = mx.sort(last, axis=-1)[:, -top_k:]
            kth = v[:, 0:1]
            last = mx.where(last < kth, -1e9, last)
        # Sample from softmax
        probs = mx.softmax(last, axis=-1)
        # mx.random.categorical samples from logits, not probs
        # Use mx.random.categorical on log-probs (== last after softmax up to const)
        nxt = mx.random.categorical(last)[:, None]
        toks = mx.concatenate([toks, nxt], axis=1)
        mx.eval(toks)
    out_bytes = bytes(np.array(toks[0]).astype(np.uint8).tolist())
    return out_bytes.decode("utf-8", errors="replace")


def render_samples(model: CortexLM) -> str:
    lines = []
    for p in PROBE_PROMPTS:
        text = sample(model, p, max_new=80, temperature=0.8, top_k=40)
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
    ckpt_dir: str = "checkpoints/lm_mlx"
    run_name: str = ""
    seq_len: int = 128
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 0.05
    warmup_steps: int = 200
    total_steps: int = 10000
    log_every: int = 100
    ckpt_every: int = 500
    seed: int = 0
    n_layers: int = 4
    d_model: int = 128
    d_state: int = 16
    expand: int = 2
    headdim: int = 16
    dtype: str = "float32"   # "float32" | "bfloat16"


def _cast_tree(tree, dtype):
    """Recursively cast all mx.array leaves to `dtype`."""
    if isinstance(tree, dict):
        return {k: _cast_tree(v, dtype) for k, v in tree.items()}
    if isinstance(tree, list):
        return [_cast_tree(v, dtype) for v in tree]
    if isinstance(tree, mx.array):
        return tree.astype(dtype)
    return tree


_DTYPE_MAP = {
    "float32": mx.float32,
    "fp32": mx.float32,
    "bfloat16": mx.bfloat16,
    "bf16": mx.bfloat16,
}


def lr_at(step, warmup, total):
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def loss_fn(model: CortexLM, x: mx.array, y: mx.array) -> mx.array:
    logits = model(x)
    # Cross entropy; reshape to (B*L, V)
    B, L, V = logits.shape
    flat = logits.reshape(-1, V)
    targ = y.reshape(-1)
    return mx_nn.losses.cross_entropy(flat, targ, reduction="mean")


def save_ckpt(model: CortexLM, opt_state, step: int, ckpt_dir: Path,
              tag: str = "") -> Path:
    name = f"step_{tag}.npz" if tag else f"step_{step:06d}.npz"
    path = ckpt_dir / name
    flat = mlx.utils.tree_flatten(model.parameters())
    state = {"step": np.array(step), **{f"model/{k}": np.array(v) for k, v in flat}}
    # Also save optimizer state (allows resume)
    opt_flat = mlx.utils.tree_flatten(opt_state)
    state.update({f"opt/{k}": np.array(v) for k, v in opt_flat})
    np.savez(path, **state)
    return path


def train(cfg: LMTrainConfig, resume: str | None = None):
    print(f"device: {mx.default_device()}  metal: {mx.metal.is_available()}")
    print(f"mlx training: {cfg.n_layers}L d_model={cfg.d_model} "
          f"seq={cfg.seq_len} batch={cfg.batch_size}")

    np.random.seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    mx.random.seed(cfg.seed)

    ckpt_dir = Path(cfg.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = ckpt_dir / "training.log"

    lm_cfg = CortexLMConfig(
        n_layers=cfg.n_layers, d_model=cfg.d_model, d_state=cfg.d_state,
        expand=cfg.expand, headdim=cfg.headdim,
        vocab_size=256, max_seq_len=cfg.seq_len,
    )
    model = CortexLM(lm_cfg)
    if cfg.dtype not in _DTYPE_MAP:
        raise ValueError(f"unknown dtype: {cfg.dtype}")
    dtype = _DTYPE_MAP[cfg.dtype]
    if dtype != mx.float32:
        model.update(_cast_tree(model.parameters(), dtype))
        print(f"cast model parameters -> {cfg.dtype}")
    mx.eval(model.parameters())

    opt = optim.AdamW(learning_rate=cfg.lr, weight_decay=cfg.weight_decay,
                      betas=[0.9, 0.95])
    n_params = sum(v.size for k, v in mlx.utils.tree_flatten(model.parameters()))

    start_step = 1
    if resume:
        loaded = np.load(resume, allow_pickle=False)
        flat = []
        for k in loaded.files:
            if k.startswith("model/"):
                flat.append((k[6:], mx.array(loaded[k])))
        model.update(mlx.utils.tree_unflatten(flat))
        start_step = int(loaded["step"]) + 1
        print(f"resumed from {resume} @ step {start_step - 1}", flush=True)

    log_lines = [
        f"# bilingual cortex LM training (MLX)",
        f"# device={mx.default_device()} metal={mx.metal.is_available()} params={n_params:,}",
        f"# layers={cfg.n_layers} d_model={cfg.d_model} seq_len={cfg.seq_len} "
        f"batch={cfg.batch_size} lr={cfg.lr} steps={cfg.total_steps}",
        f"# corpus={cfg.corpus_path}",
        "",
    ]
    if not log_path.exists() or not resume:
        log_path.write_text("\n".join(log_lines) + "\n")

    def append_log(text: str):
        with open(log_path, "a") as f:
            f.write(text + "\n")

    # ─── Firebase mirror ───
    pusher = None
    if _HAS_PUSHER:
        kind = Path(__file__).resolve().parent.name        # "cortex_bilingual"
        exp_id = f"{kind}-{time.strftime('%Y-%m-%d')}"
        run_id = cfg.run_name or ckpt_dir.name
        from dataclasses import asdict
        pusher = ExperimentPusher(
            experiment_id=exp_id, run_id=run_id, kind=kind,
            config=asdict(cfg), outbox_dir=str(ckpt_dir),
        )
        pusher.declare_experiment(name=kind,
            hypothesis="Bilingual byte-level Mamba-3 LM trained on MLX "
                       "(Apple-Silicon-native), with optional bf16. "
                       "Host for cortex primitive attach experiments.")
        pusher.declare_run(purpose=f"{cfg.n_layers}L d={cfg.d_model} "
                                   f"steps={cfg.total_steps} dtype={cfg.dtype}",
                           gpu=-1)
        print(f"[firebase] pushing to /experiments/{exp_id}/runs/{run_id}", flush=True)

    # ─── HuggingFace bucket archive — silent no-op without HF_TOKEN ───
    archive = None
    if _HAS_ARCHIVE:
        kind = Path(__file__).resolve().parent.name        # "cortex_bilingual"
        archive = CloudArchive(
            experiment_kind=kind,
            run_name=cfg.run_name or ckpt_dir.name,
            local_dir=str(ckpt_dir),
        )

    corpus = ByteCorpus(cfg.corpus_path, cfg.seq_len)

    # Build the value_and_grad function once
    state = [model.state, opt.state]

    def step_fn(x, y):
        loss_and_grad = mx_nn.value_and_grad(model, loss_fn)
        loss, grads = loss_and_grad(model, x, y)
        # Gradient clipping by global norm
        grads_flat = mlx.utils.tree_flatten(grads)
        sq = sum((mx.sum(g * g) for _, g in grads_flat), mx.array(0.0))
        norm = mx.sqrt(sq)
        scale = mx.minimum(mx.array(1.0), 1.0 / (norm + 1e-6))
        grads = mlx.utils.tree_map(lambda g: g * scale, grads)
        opt.update(model, grads)
        return loss

    print(f"params: {n_params:,}  steps: {start_step}..{cfg.total_steps}", flush=True)
    t0 = time.time()
    last_metrics: dict = {}

    for step in range(start_step, cfg.total_steps + 1):
        # Set LR per cosine schedule
        opt.learning_rate = cfg.lr * lr_at(step, cfg.warmup_steps, cfg.total_steps)

        x, y = corpus.get_batch(cfg.batch_size, rng)
        loss = step_fn(x, y)
        mx.eval(model.parameters(), opt.state, loss)

        if step % cfg.log_every == 0 or step == 1:
            elapsed = time.time() - t0
            lv = float(loss)
            bpc = lv / math.log(2)
            lr_now = opt.learning_rate.item() if hasattr(opt.learning_rate, 'item') else opt.learning_rate
            sps = step / max(elapsed, 1e-3)
            line = (f"step {step:6d}/{cfg.total_steps}  "
                    f"loss={lv:.3f}  bpc={bpc:.3f}  "
                    f"lr={lr_now:.2e}  "
                    f"elapsed={elapsed:.1f}s")
            print(line, flush=True)
            append_log(line)
            if pusher is not None:
                pusher.metrics(step=step, byte_ce=lv, bpc=bpc, lr=float(lr_now))
                pusher.heartbeat(step=step, sps=float(sps))
            last_metrics = {"byte_ce": lv, "bpc": bpc, "lr": float(lr_now)}

        if step % cfg.ckpt_every == 0 or step == cfg.total_steps:
            path = save_ckpt(model, opt.state, step, ckpt_dir)
            samples = render_samples(model)
            block = (f"\n--- step {step} samples (saved {path.name}) ---\n"
                     f"{samples}\n")
            print(block, flush=True)
            append_log(block)
            if pusher is not None:
                first = PROBE_PROMPTS[0]
                cont = sample(model, first, max_new=80, temperature=0.8, top_k=40)
                pusher.canary_sample(step=step, prompt=first,
                                     completion=cont[len(first):])

    final_path = save_ckpt(model, opt.state, cfg.total_steps, ckpt_dir, tag="FINAL")
    print(f"\nDone. Final checkpoint: {final_path}", flush=True)
    print(f"Training log: {log_path}", flush=True)
    if pusher is not None:
        pusher.event(type="run_complete", step=cfg.total_steps,
                     details=f"reached step {cfg.total_steps}")
        pusher.complete(final_state="completed", final_metrics=last_metrics)
    if archive is not None:
        archive.complete()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/bilingual.txt")
    ap.add_argument("--ckpt-dir", default="checkpoints/lm_mlx")
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--n-layers", type=int, default=4)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--ckpt-every", type=int, default=500)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", default="float32",
                    choices=["float32", "fp32", "bfloat16", "bf16"])
    ap.add_argument("--run-name", default="",
                    help="Firebase run_id; defaults to ckpt-dir basename")
    args = ap.parse_args()

    cfg = LMTrainConfig(
        corpus_path=args.corpus, ckpt_dir=args.ckpt_dir,
        run_name=args.run_name,
        total_steps=args.steps, seq_len=args.seq_len, batch_size=args.batch_size,
        lr=args.lr, n_layers=args.n_layers, d_model=args.d_model,
        ckpt_every=args.ckpt_every, log_every=args.log_every, seed=args.seed,
        dtype=args.dtype,
    )
    train(cfg, resume=args.resume)


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    main()
