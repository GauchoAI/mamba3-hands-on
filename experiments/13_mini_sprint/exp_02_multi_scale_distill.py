"""exp_02 — multi-scale residual matching (V4 hybrid-attention analog).

DeepSeek V4 uses three parallel attention pathways (heavily compressed for
breadth, sparse-retrieved blocks for precision, sliding window for
fidelity) — three complementary signals composed into one representation.

Our analog: instead of a single end-of-prompt smooth-L1 target (which we
already showed fails — see findings §6), match the EMA student's residual
at THREE positions per sample: (1) start of prompt, (2) middle, (3) end.
Three smooth-L1 losses summed. Each pulls the residual toward stability
at its own time scale.

This is also a self-distillation variant of round 7 (which used Qwen
teacher hiddens at end-of-prompt). Here teacher = EMA student. No Qwen
forward needed → much faster iteration.

Hypothesis: a single-position alignment is recoverable by averaging
behavior; matching three positions simultaneously requires the residual
to be position-aware, which forces input-dependent encoding.

Config: same scale as exp_00. Differs from exp_01 only in the loss
location — alignment at 3 positions instead of all positions.

Run:
  .venv/bin/python experiments/13_mini_sprint/exp_02_multi_scale_distill.py \\
      --corpus data/movie_pairs_clean.txt
"""
from __future__ import annotations
import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "experiments" / "07_jepa"))
sys.path.insert(0, str(HERE))
from cortex_counting import CortexLM, CortexLMConfig
from exp_00_clean_corpus_baseline import (CleanByteIterator, lr_at,
                                           eval_canary)
from exp_01_ema_self_distill import Predictor, update_ema


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
    ap.add_argument("--ema-momentum", type=float, default=0.996)
    ap.add_argument("--lambda-ms", type=float, default=1.0)
    ap.add_argument("--ms-warmup", type=int, default=200)
    ap.add_argument("--ms-ramp", type=int, default=400)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run-name", default="exp_02_multi_scale_distill")
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
    ema_model = copy.deepcopy(model).to(device).eval()
    for p in ema_model.parameters():
        p.requires_grad = False
    predictor = Predictor(args.d_model).to(device)

    n_live = sum(p.numel() for p in model.parameters())
    print(f"[init] live={n_live:,} predictor={sum(p.numel() for p in predictor.parameters()):,} device={device}",
          flush=True)

    opt = torch.optim.AdamW(
        list(model.parameters()) + list(predictor.parameters()),
        lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1,
    )

    iterator = CleanByteIterator(args.corpus, args.batch_size,
                                 seq_len=args.seq_len, seed=args.seed)
    run_dir = Path("runs") / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    loss_log = run_dir / "loss.jsonl"
    eval_log = run_dir / "eval.json"

    L = args.seq_len
    # Three sampling positions: 1/4, 1/2, 3/4 of the seq.
    positions = torch.tensor(
        [L // 4, L // 2, 3 * L // 4], device=device,
    )

    t0 = time.time()
    for step in range(args.steps):
        lr = lr_at(step, args.lr, args.warmup, args.steps)
        for pg in opt.param_groups:
            pg["lr"] = lr
        if step < args.ms_warmup:
            ms_w = 0.0
        else:
            ms_w = args.lambda_ms * min(
                1.0, (step - args.ms_warmup) / max(1, args.ms_ramp)
            )

        tokens = next(iterator).to(device)
        plens = torch.full((tokens.size(0),), L // 2,
                           dtype=torch.long, device=device)
        logits, _, residual_live, _ = model(tokens, return_jepa=True,
                                             prompt_lens=plens)
        l_byte = F.cross_entropy(
            logits[:, :-1].reshape(-1, 256),
            tokens[:, 1:].reshape(-1),
        )

        with torch.no_grad():
            _, _, residual_ema, _ = ema_model(tokens, return_jepa=True,
                                              prompt_lens=plens)

        # Multi-scale: gather residuals at the 3 positions, predictor on live,
        # cosine-align with EMA. Sum of 3 cosine distances.
        live_pts = residual_live[:, positions, :].float()       # (B, 3, D)
        ema_pts = residual_ema[:, positions, :].float().detach()
        pred_pts = predictor(live_pts)
        pred_n = F.normalize(pred_pts, dim=-1)
        targ_n = F.normalize(ema_pts, dim=-1)
        l_ms = (1.0 - (pred_n * targ_n).sum(-1)).mean()

        loss = l_byte + ms_w * l_ms

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(predictor.parameters()), 1.0,
        )
        opt.step()
        update_ema(ema_model, model, args.ema_momentum)

        if step % 50 == 0:
            sps = (step + 1) / max(time.time() - t0, 1e-6)
            line = json.dumps({
                "step": step, "loss": float(loss.detach()),
                "l_byte": float(l_byte.detach()),
                "l_ms": float(l_ms.detach()),
                "ms_w": ms_w, "lr": lr, "sps": sps,
            })
            print(f"step={step:5d} loss={float(loss.detach()):.4f} "
                  f"byte={float(l_byte.detach()):.4f} "
                  f"ms={float(l_ms.detach()):.4f} (w={ms_w:.2f}) "
                  f"sps={sps:.2f}", flush=True)
            loss_log.open("a").write(line + "\n")

    print("\n[eval]", flush=True)
    metrics = eval_canary(model, device)
    metrics["final_loss"] = float(loss.detach())
    metrics["steps"] = args.steps
    metrics["d_model"] = args.d_model
    metrics["positions"] = [int(p) for p in positions]
    eval_log.write_text(json.dumps(metrics, indent=2))
    print(f"[eval] retention={metrics['retention']:.4f} "
          f"drift={metrics['drift']:.4f} "
          f"diversity={metrics['diversity']:.4f}", flush=True)
    print(f"[done] {run_dir}", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    main()
