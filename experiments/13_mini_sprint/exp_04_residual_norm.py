"""exp_04 — bounded residual-norm constraint (V4 MHC-light analog).

DeepSeek V4 uses Manifold-Constrained Hyperconnections (MHC) with
Sinkhorn-Knopp iterations to enforce doubly-stochastic constraints on
residual mixing matrices, preventing signal explosion at trillion-param
scale. We don't have a signal-explosion problem at our scale, but the
*spirit* is interesting: structurally constrain the residual stream
geometry.

Light variant: penalize the residual norm growth across layers. Keep
total signal energy bounded. Specifically: at each step, compute the
ratio ||residual_layer_n|| / ||residual_layer_0|| across the seq, and
add a soft penalty on log-ratio drift > 1.0. Normalizes the residual
trajectory without forcing exact orthogonality.

Hypothesis: SSM residuals can drift in magnitude during training in a
way that masks the directional information we care about (retention is
a cosine, but if magnitudes blow up the SSM becomes attractor-like).
Bounding norm growth could keep the residual subspace better-shaped
for prompt-conditional encoding.

This is the V4 idea most likely to NOT help us at small scale — small
models don't have signal-explosion. Including for completeness.

Run:
  .venv/bin/python experiments/13_mini_sprint/exp_04_residual_norm.py \\
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/movie_pairs_clean.txt")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--d-model", type=int, default=96)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--lambda-norm", type=float, default=0.05)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run-name", default="exp_04_residual_norm")
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
        expand=2, headdim=16, vocab_size=256, max_seq_len=args.seq_len,
        use_counter=False,
    )
    model = CortexLM(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[init] params={n_params:,} device={device}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            betas=(0.9, 0.95), weight_decay=0.1)

    iterator = CleanByteIterator(args.corpus, args.batch_size,
                                 seq_len=args.seq_len, seed=args.seed)
    run_dir = Path("runs") / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    loss_log = run_dir / "loss.jsonl"
    eval_log = run_dir / "eval.json"

    t0 = time.time()
    for step in range(args.steps):
        lr = lr_at(step, args.lr, args.warmup, args.steps)
        for pg in opt.param_groups:
            pg["lr"] = lr
        tokens = next(iterator).to(device)
        plens = torch.full((tokens.size(0),), args.seq_len // 2,
                           dtype=torch.long, device=device)
        logits, _, residual, _ = model(tokens, return_jepa=True,
                                        prompt_lens=plens)
        l_byte = F.cross_entropy(
            logits[:, :-1].reshape(-1, 256),
            tokens[:, 1:].reshape(-1),
        )
        # Residual norm regularization: penalize per-position residual
        # norms that diverge from a target (1.0 after L2-norm by hidden dim).
        # Computed in fp32 for numerical stability on MPS.
        r = residual.float()
        norms = r.norm(dim=-1) / (args.d_model ** 0.5)            # (B, L)
        # Penalize log-norm-squared distance from 1.0
        l_norm = (norms.log().pow(2)).mean()
        loss = l_byte + args.lambda_norm * l_norm

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % 50 == 0:
            sps = (step + 1) / max(time.time() - t0, 1e-6)
            line = json.dumps({
                "step": step, "loss": float(loss.detach()),
                "l_byte": float(l_byte.detach()),
                "l_norm": float(l_norm.detach()),
                "lr": lr, "sps": sps,
            })
            print(f"step={step:5d} loss={float(loss.detach()):.4f} "
                  f"byte={float(l_byte.detach()):.4f} "
                  f"norm={float(l_norm.detach()):.4f} "
                  f"sps={sps:.2f}", flush=True)
            loss_log.open("a").write(line + "\n")

    print("\n[eval]", flush=True)
    metrics = eval_canary(model, device)
    metrics["final_loss"] = float(loss.detach())
    metrics["steps"] = args.steps
    metrics["d_model"] = args.d_model
    metrics["params"] = n_params
    eval_log.write_text(json.dumps(metrics, indent=2))
    print(f"[eval] retention={metrics['retention']:.4f} "
          f"drift={metrics['drift']:.4f} "
          f"diversity={metrics['diversity']:.4f}", flush=True)
    print(f"[done] {run_dir}", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    main()
