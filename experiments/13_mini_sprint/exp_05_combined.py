"""exp_05 — combined: V4 "compose many complementary signals" lesson.

Stack everything from exp_01..04:
  • clean within-movie corpus (data/movie_pairs_clean.txt)
  • EMA self-distillation (BYOL-style, asymmetric predictor)
  • Multi-scale residual matching (3 positions: 1/4, 1/2, 3/4)
  • Sequence-length curriculum (32 → 64 → 128)
  • Bounded residual-norm constraint

DeepSeek V4's central engineering thesis is that no single trick wins;
gains come from composing many small architectural pieces that each
target a different failure mode. This experiment tests that thesis on
our autopilot problem: even if no individual lever from exp_01-04
crosses retention 0.30, the combination might.

If exp_05 cracks retention while individual exp_NN don't, the V4
"compose" thesis transfers to small models. If it also fails, we have
strong evidence that small Mamba-3 + pseudo-distillation cannot be
made prompt-conditional, and the only remaining lever is logit-
projection KD (round 9 on vast.ai).

Run:
  .venv/bin/python experiments/13_mini_sprint/exp_05_combined.py \\
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
from exp_03_curriculum import seq_len_at


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
    ap.add_argument("--ema-momentum", type=float, default=0.996)
    ap.add_argument("--lambda-ms", type=float, default=1.0)
    ap.add_argument("--lambda-norm", type=float, default=0.05)
    ap.add_argument("--ms-warmup", type=int, default=200)
    ap.add_argument("--ms-ramp", type=int, default=400)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run-name", default="exp_05_combined")
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
    ema_model = copy.deepcopy(model).to(device).eval()
    for p in ema_model.parameters():
        p.requires_grad = False
    predictor = Predictor(args.d_model).to(device)
    n_live = sum(p.numel() for p in model.parameters())
    n_pred = sum(p.numel() for p in predictor.parameters())
    print(f"[init] live={n_live:,} predictor={n_pred:,} device={device}",
          flush=True)

    opt = torch.optim.AdamW(
        list(model.parameters()) + list(predictor.parameters()),
        lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1,
    )

    iters = {
        sl: CleanByteIterator(args.corpus, args.batch_size,
                              seq_len=sl, seed=args.seed + sl)
        for sl in (32, 64, 128)
    }
    run_dir = Path("runs") / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    loss_log = run_dir / "loss.jsonl"
    eval_log = run_dir / "eval.json"

    t0 = time.time()
    for step in range(args.steps):
        sl = seq_len_at(step)
        positions = torch.tensor([sl // 4, sl // 2, 3 * sl // 4],
                                 device=device)
        lr = lr_at(step, args.lr, args.warmup, args.steps)
        for pg in opt.param_groups:
            pg["lr"] = lr
        if step < args.ms_warmup:
            ms_w = 0.0
        else:
            ms_w = args.lambda_ms * min(
                1.0, (step - args.ms_warmup) / max(1, args.ms_ramp)
            )

        tokens = next(iters[sl]).to(device)
        plens = torch.full((tokens.size(0),), sl // 2,
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

        # Multi-scale BYOL: predictor on live at 3 positions, cosine vs EMA
        live_pts = residual_live[:, positions, :].float()
        ema_pts = residual_ema[:, positions, :].float().detach()
        pred_pts = predictor(live_pts)
        pred_n = F.normalize(pred_pts, dim=-1)
        targ_n = F.normalize(ema_pts, dim=-1)
        l_ms = (1.0 - (pred_n * targ_n).sum(-1)).mean()

        # Residual norm constraint
        r = residual_live.float()
        norms = r.norm(dim=-1) / (args.d_model ** 0.5)
        l_norm = (norms.log().pow(2)).mean()

        loss = l_byte + ms_w * l_ms + args.lambda_norm * l_norm

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
                "step": step, "sl": sl,
                "loss": float(loss.detach()),
                "l_byte": float(l_byte.detach()),
                "l_ms": float(l_ms.detach()),
                "l_norm": float(l_norm.detach()),
                "ms_w": ms_w, "lr": lr, "sps": sps,
            })
            print(f"step={step:5d} sl={sl:3d} "
                  f"loss={float(loss.detach()):.4f} "
                  f"byte={float(l_byte.detach()):.4f} "
                  f"ms={float(l_ms.detach()):.4f} (w={ms_w:.2f}) "
                  f"norm={float(l_norm.detach()):.4f} "
                  f"sps={sps:.2f}", flush=True)
            loss_log.open("a").write(line + "\n")

    print("\n[eval]", flush=True)
    metrics = eval_canary(model, device)
    metrics["final_loss"] = float(loss.detach())
    metrics["steps"] = args.steps
    metrics["d_model"] = args.d_model
    metrics["params_live"] = n_live
    metrics["params_predictor"] = n_pred
    metrics["levers_active"] = ["clean_corpus", "ema_self_distill",
                                 "multi_scale", "curriculum",
                                 "residual_norm"]
    eval_log.write_text(json.dumps(metrics, indent=2))
    print(f"[eval] retention={metrics['retention']:.4f} "
          f"drift={metrics['drift']:.4f} "
          f"diversity={metrics['diversity']:.4f}", flush=True)
    print(f"[done] {run_dir}", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    main()
