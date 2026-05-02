"""exp_01 — EMA self-distillation (BYOL/DINO style).

DeepSeek V4 anticipatory-routing analog. V4 uses slightly-older snapshots
of model parameters during training to "ignore noise and lock onto the
underlying trend." We adapt this as BYOL: maintain an EMA copy of the
student, and ask the live student's predictor to predict the EMA
student's residual at end-of-prompt.

Critical asymmetry (BYOL/DINO): there's a small predictor MLP on the
live branch but NOT the EMA branch, and gradients are stopped through
the EMA branch. Without that asymmetry both networks collapse to a
constant.

Hypothesis: the EMA's residual at end-of-prompt is *forced* to be a
function of the prompt (the EMA student processes the prompt). If the
live student's predictor learns to map the live residual onto that
EMA target, both networks pull each other toward more prompt-stable
representations. May or may not crack autopilot — could fail if both
networks are happy in the autopilot regime — but worth ~15 min wall-clock
to find out.

Config: same scale as exp_00 (d_model=96, n_layers=2, batch=32,
seq_len=128, 2000 steps). Adds:
  - EMA momentum 0.996
  - Predictor: 2-layer MLP, hidden=192, output=d_model=96 (~36k params)
  - λ_byol = 1.0, ramp from step 200 to step 600 (linear)

Run:
  .venv/bin/python experiments/13_mini_sprint/exp_01_ema_self_distill.py \\
      --corpus data/movie_pairs_clean.txt
"""
from __future__ import annotations
import argparse
import copy
import json
import math
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
                                           eval_canary, CANARY_PROMPTS)


class Predictor(nn.Module):
    """Small MLP that maps live residual → predicted EMA residual.

    Asymmetric capacity is the BYOL/DINO trick: live has predictor, EMA
    doesn't. Without it, both networks collapse to a trivial constant.
    """
    def __init__(self, d_model: int, hidden: int = 192):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


@torch.no_grad()
def update_ema(ema_model: nn.Module, live_model: nn.Module,
               momentum: float) -> None:
    for ep, lp in zip(ema_model.parameters(), live_model.parameters()):
        ep.data.mul_(momentum).add_(lp.data, alpha=1 - momentum)


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
    ap.add_argument("--lambda-byol", type=float, default=1.0)
    ap.add_argument("--byol-warmup", type=int, default=200)
    ap.add_argument("--byol-ramp", type=int, default=400)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run-name", default="exp_01_ema_self_distill")
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
    n_pred = sum(p.numel() for p in predictor.parameters())
    print(f"[init] live={n_live:,} predictor={n_pred:,} device={device}",
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

    t0 = time.time()
    for step in range(args.steps):
        lr = lr_at(step, args.lr, args.warmup, args.steps)
        for pg in opt.param_groups:
            pg["lr"] = lr
        if step < args.byol_warmup:
            byol_w = 0.0
        else:
            byol_w = args.lambda_byol * min(
                1.0, (step - args.byol_warmup) / max(1, args.byol_ramp)
            )

        tokens = next(iterator).to(device)
        # Live forward — full residuals
        # Use return_jepa=True to get the residual stream
        plens = torch.full((tokens.size(0),), tokens.size(1) // 2,
                           dtype=torch.long, device=device)
        logits, _, residual_live, _ = model(tokens, return_jepa=True,
                                             prompt_lens=plens)
        l_byte = F.cross_entropy(
            logits[:, :-1].reshape(-1, 256),
            tokens[:, 1:].reshape(-1),
        )

        # EMA forward (no grad)
        with torch.no_grad():
            _, _, residual_ema, _ = ema_model(tokens, return_jepa=True,
                                              prompt_lens=plens)

        # BYOL loss: predictor(live) → EMA, cosine alignment, all positions.
        # Pulls the entire residual stream (not just one position) toward
        # a smoother version of itself, which is the anticipatory-routing
        # spirit.
        pred = predictor(residual_live.float())
        target = residual_ema.float().detach()
        pred_n = F.normalize(pred, dim=-1)
        targ_n = F.normalize(target, dim=-1)
        l_byol = (1.0 - (pred_n * targ_n).sum(-1)).mean()

        loss = l_byte + byol_w * l_byol

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
                "l_byol": float(l_byol.detach()),
                "byol_w": byol_w, "lr": lr, "sps": sps,
            })
            print(f"step={step:5d} loss={float(loss.detach()):.4f} "
                  f"byte={float(l_byte.detach()):.4f} "
                  f"byol={float(l_byol.detach()):.4f} (w={byol_w:.2f}) "
                  f"lr={lr:.2e} sps={sps:.2f}", flush=True)
            loss_log.open("a").write(line + "\n")

    print("\n[eval] computing canary retention on LIVE model...",
          flush=True)
    metrics = eval_canary(model, device)
    metrics["final_loss"] = float(loss.detach())
    metrics["steps"] = args.steps
    metrics["d_model"] = args.d_model
    metrics["params_live"] = n_live
    metrics["params_predictor"] = n_pred
    metrics["ema_momentum"] = args.ema_momentum
    eval_log.write_text(json.dumps(metrics, indent=2))
    print(f"[eval] retention={metrics['retention']:.4f} "
          f"drift={metrics['drift']:.4f} "
          f"diversity={metrics['diversity']:.4f}", flush=True)
    print(f"[eval] sample completion 0: "
          f"{metrics['completions'][0][:120]!r}", flush=True)
    print(f"[done] {run_dir}", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    main()
