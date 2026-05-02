"""exp_03 — sequence-length curriculum (V4 4k→1M analog).

DeepSeek V4 starts training with short sequences (4k tokens) and
gradually expands to 1M as the model stabilizes. The intuition: the
model first learns local structure (grammar, syntax, token-pair
patterns) on cheap short examples, then gradually expands the working
memory.

Our analog at byte-LM scale: seq_len starts at 32 and doubles every
~500 steps until 128. Same total step count as exp_00, just a different
distribution of context lengths. Combined with the byte-CE only baseline
to isolate the curriculum effect from any distillation lever.

Hypothesis: a small Mamba-3 might find the autopilot regime *because*
its short-context predictions are too easy to satisfy with surface
continuation. Forcing it to do well on tiny seq_len=32 first may
build a tighter prompt-conditional regime, which then transfers as
seq_len grows.

Schedule:
  step 0–500:    seq_len = 32
  step 500–1000: seq_len = 64
  step 1000–2000: seq_len = 128

Same RNG seed and lr schedule as exp_00 for clean comparison.

Run:
  .venv/bin/python experiments/13_mini_sprint/exp_03_curriculum.py \\
      --corpus data/movie_pairs_clean.txt
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "experiments" / "07_jepa"))
sys.path.insert(0, str(HERE))
from cortex_counting import CortexLM, CortexLMConfig
from exp_00_clean_corpus_baseline import (CleanByteIterator, lr_at,
                                           eval_canary)


def seq_len_at(step: int) -> int:
    if step < 500:
        return 32
    if step < 1000:
        return 64
    return 128


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/movie_pairs_clean.txt")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-seq-len", type=int, default=128)
    ap.add_argument("--d-model", type=int, default=96)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run-name", default="exp_03_curriculum")
    args = ap.parse_args()

    if args.device == "auto":
        device = torch.device("mps" if torch.backends.mps.is_available()
                              else "cpu")
    else:
        device = torch.device(args.device)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = CortexLMConfig(
        n_layers=args.n_layers, d_model=args.d_model, d_state=16,
        expand=2, headdim=16, vocab_size=256, max_seq_len=args.max_seq_len,
        use_counter=False,
    )
    model = CortexLM(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[init] params={n_params:,} device={device}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            betas=(0.9, 0.95), weight_decay=0.1)

    # Maintain three iterators (one per seq_len) so we don't re-create
    # the Path memmap every step.
    rng_seed = args.seed
    iters = {
        sl: CleanByteIterator(args.corpus, args.batch_size,
                              seq_len=sl, seed=rng_seed + sl)
        for sl in (32, 64, 128)
    }
    run_dir = Path("runs") / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    loss_log = run_dir / "loss.jsonl"
    eval_log = run_dir / "eval.json"

    t0 = time.time()
    for step in range(args.steps):
        sl = seq_len_at(step)
        lr = lr_at(step, args.lr, args.warmup, args.steps)
        for pg in opt.param_groups:
            pg["lr"] = lr
        tokens = next(iters[sl]).to(device)
        logits = model(tokens)
        pred = logits[:, :-1].reshape(-1, 256)
        tgt = tokens[:, 1:].reshape(-1)
        loss = F.cross_entropy(pred, tgt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % 50 == 0:
            sps = (step + 1) / max(time.time() - t0, 1e-6)
            line = json.dumps({"step": step, "seq_len": sl,
                               "loss": float(loss.detach()),
                               "lr": lr, "sps": sps})
            print(f"step={step:5d} sl={sl:3d} loss={float(loss.detach()):.4f} "
                  f"lr={lr:.2e} sps={sps:.2f}", flush=True)
            loss_log.open("a").write(line + "\n")

    print("\n[eval]", flush=True)
    metrics = eval_canary(model, device)
    metrics["final_loss"] = float(loss.detach())
    metrics["steps"] = args.steps
    metrics["d_model"] = args.d_model
    metrics["params"] = n_params
    metrics["curriculum"] = "32@500, 64@1000, 128@2000"
    eval_log.write_text(json.dumps(metrics, indent=2))
    print(f"[eval] retention={metrics['retention']:.4f} "
          f"drift={metrics['drift']:.4f} "
          f"diversity={metrics['diversity']:.4f}", flush=True)
    print(f"[done] {run_dir}", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    main()
