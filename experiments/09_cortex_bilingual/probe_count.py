"""probe_count.py — per-layer regression probe for the integer count.

Companion to probe_layers.py. The binary probe showed the "I'm in
unary" feature is layer-redundant and OOD-robust (98%+ accuracy at
N=200 with a probe trained on N≤60). This probe tests the next
hypothesis: does the LM's residual *also* carry the integer
position-within-run, or just the binary mode bit?

Mechanistic prediction. Today's counter primitive showed a uniform
+4 offset across N=10..500 — fires on the right bytes but emits a
constant offset's worth of extra `a`s. If the LM carries the count
correctly, the counter should be able to read it; the +4 is then a
calibration bug. If the LM saturates the count past training-N,
the counter has no fine-grained position signal OOD and falls back
to a learned run-length bias. This probe falsifies between those.

Per-layer linear regression: from residual at byte position i to
the integer "i is the k-th `a` in its unary run" (skipped on
non-unary positions). Trained on N≤60, eval'd in-dist (N≤60) and
OOD (N up to 200).

Run:
    python cortex_bilingual/probe_count.py \\
        --lm-ckpt checkpoints/lm_mlx_widerN/step_FINAL.pt
"""
from __future__ import annotations
import argparse
import os
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lab_platform.cortex_counting import CortexLM, CortexLMConfig
from cortex_bilingual.probe_layers import forward_with_taps


def make_count_batch(n_seqs: int, seq_len: int, *,
                     n_min: int, n_max: int,
                     unary_p: float = 1.0,
                     seed: int = 0
                     ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate sequences and per-position integer-count labels.

    Each sequence has a `*N:aaaa...a\n` payload.  At byte position i
    inside the run (`a` only, exclusive of `:` and `\n`), the count
    label = (position from start of run, 1-indexed).  Position-mask is
    1 at those bytes, 0 elsewhere.

    Returns:
      ids        (B, T)  byte ids
      counts     (B, T)  integer counts; meaningful where mask=1
      mask       (B, T)  1 at unary-run-interior bytes, 0 elsewhere
    """
    rng = random.Random(seed)
    ids = torch.zeros((n_seqs, seq_len), dtype=torch.long)
    counts = torch.zeros((n_seqs, seq_len), dtype=torch.float32)
    mask = torch.zeros((n_seqs, seq_len), dtype=torch.float32)

    for b in range(n_seqs):
        if rng.random() < unary_p:
            prefix_len = rng.randint(0, max(0, seq_len // 4))
            prefix = bytes(rng.randint(32, 126) for _ in range(prefix_len))
            n = rng.randint(n_min, min(n_max, seq_len - prefix_len - 8))
            payload = b"*" + str(n).encode() + b":" + b"a" * n + b"\n"
            seq = (prefix + payload)[:seq_len]
            seq = seq + bytes(rng.randint(32, 126) for _ in range(seq_len - len(seq)))
            seq_arr = torch.tensor(list(seq), dtype=torch.long)
            colon_idx = (seq_arr == ord(":")).nonzero(as_tuple=True)[0]
            newline_idx = (seq_arr == ord("\n")).nonzero(as_tuple=True)[0]
            ids[b] = seq_arr
            if len(colon_idx) and len(newline_idx):
                c = int(colon_idx[0])
                nl_after = newline_idx[newline_idx > c]
                if len(nl_after):
                    nl = int(nl_after[0])
                    for k, p in enumerate(range(c + 1, nl), start=1):
                        counts[b, p] = float(k)
                        mask[b, p] = 1.0
        else:
            seq = bytes(rng.randint(32, 126) for _ in range(seq_len))
            ids[b] = torch.tensor(list(seq), dtype=torch.long)
    return ids, counts, mask


def train_regressor(features: torch.Tensor, targets: torch.Tensor,
                    lr: float = 5e-3, steps: int = 400,
                    weight_decay: float = 1e-4) -> tuple[nn.Linear, float]:
    """Linear regression head; MSE loss. Returns (head, train_MAE)."""
    N, D = features.shape
    head = nn.Linear(D, 1)
    opt = torch.optim.AdamW(head.parameters(), lr=lr,
                            weight_decay=weight_decay)
    for _ in range(steps):
        pred = head(features).squeeze(-1)
        loss = nn.functional.mse_loss(pred, targets)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred = head(features).squeeze(-1)
        mae = (pred - targets).abs().mean().item()
    return head, mae


def eval_regressor(head: nn.Linear, features: torch.Tensor,
                   targets: torch.Tensor) -> dict:
    with torch.no_grad():
        pred = head(features).squeeze(-1)
        mae = (pred - targets).abs().mean().item()
        rmse = (pred - targets).pow(2).mean().sqrt().item()
        # Plot-friendly: bucket by true count, report mean predicted
        return {"mae": mae, "rmse": rmse, "pred": pred, "target": targets}


def bucketed_predictions(pred: torch.Tensor, target: torch.Tensor,
                         buckets: list[int]) -> list[tuple[int, int, float, float]]:
    """For each (low, high] bucket of true count, return
    (low, high, n, mean_pred, mean_target). Last bucket is open-ended."""
    rows = []
    for lo, hi in zip([0] + buckets, buckets + [int(target.max().item()) + 1]):
        m = (target > lo) & (target <= hi)
        if m.sum() == 0:
            continue
        rows.append((lo, hi, int(m.sum()),
                     float(pred[m].mean()),
                     float(target[m].mean())))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lm-ckpt", required=True)
    ap.add_argument("--n-train", type=int, default=512)
    ap.add_argument("--n-test", type=int, default=256)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--n-min", type=int, default=5)
    ap.add_argument("--n-max", type=int, default=60,
                    help="train + in-dist eval upper-bound")
    ap.add_argument("--n-max-ood", type=int, default=110,
                    help="OOD eval upper-bound (capped by seq_len margin)")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    if args.device == "auto":
        args.device = "mps" if torch.backends.mps.is_available() else "cpu"
    device = args.device

    payload = torch.load(args.lm_ckpt, map_location="cpu", weights_only=False)
    cfg: CortexLMConfig = payload["cfg"]
    cfg.use_counter = False
    model = CortexLM(cfg).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    print(f"loaded LM: {sum(p.numel() for p in model.parameters()):,} params")

    def collect_with_mask(ids: torch.Tensor, counts: torch.Tensor,
                          mask: torch.Tensor
                          ) -> tuple[dict[int, torch.Tensor], torch.Tensor]:
        feats = forward_with_taps(model, ids.to(device))
        flat_mask = mask.flatten().bool()
        flat_counts = counts.flatten()[flat_mask]
        feats_flat = {k: v.flatten(0, 1)[flat_mask] for k, v in feats.items()}
        return feats_flat, flat_counts

    # ── train + in-dist eval (N ∈ [n_min, n_max])
    print(f"\n=== training data: N ∈ [{args.n_min}, {args.n_max}], "
          f"unary-only ===")
    ids_tr, c_tr, m_tr = make_count_batch(args.n_train, args.seq_len,
                                          n_min=args.n_min, n_max=args.n_max,
                                          seed=0)
    ids_id, c_id, m_id = make_count_batch(args.n_test, args.seq_len,
                                          n_min=args.n_min, n_max=args.n_max,
                                          seed=1)
    Ftr, Ctr = collect_with_mask(ids_tr, c_tr, m_tr)
    Fid, Cid = collect_with_mask(ids_id, c_id, m_id)
    print(f"  train positions: {len(Ctr):,}  range=[{int(Ctr.min())},{int(Ctr.max())}]")

    # ── OOD eval (N > n_max)
    ids_ood, c_ood, m_ood = make_count_batch(
        args.n_test, args.seq_len,
        n_min=args.n_max + 1, n_max=args.n_max_ood, seed=2,
    )
    Food, Cood = collect_with_mask(ids_ood, c_ood, m_ood)
    print(f"  OOD positions: {len(Cood):,}  range=[{int(Cood.min())},{int(Cood.max())}]")

    print(f"\n  {'layer':<8}{'train_MAE':<12}{'in-dist_MAE':<14}{'OOD_MAE':<12}"
          f"{'OOD/in×':<10}")
    heads = {}
    eval_rows = {}
    for li in sorted(Ftr.keys()):
        head, train_mae = train_regressor(Ftr[li], Ctr)
        eid = eval_regressor(head, Fid[li], Cid)
        eood = eval_regressor(head, Food[li], Cood)
        ratio = eood["mae"] / max(1e-6, eid["mae"])
        heads[li] = head
        eval_rows[li] = (eid, eood)
        print(f"  L{li:<7}{train_mae:<12.3f}{eid['mae']:<14.3f}"
              f"{eood['mae']:<12.3f}{ratio:<10.2f}")

    # Bucketed bias check on the best layer
    best = min(eval_rows, key=lambda li: eval_rows[li][0]["mae"])
    print(f"\n=== bucketed predictions on L{best} (lowest in-dist MAE) ===")
    print(f"  {'count range':<14}{'n':<8}{'mean_pred':<12}{'mean_target':<12}{'bias':<8}")
    eid, eood = eval_rows[best]
    all_pred = torch.cat([eid["pred"], eood["pred"]])
    all_targ = torch.cat([eid["target"], eood["target"]])
    rows = bucketed_predictions(all_pred, all_targ,
                                buckets=[10, 30, 60, 100, 150, 200])
    for lo, hi, n, mp, mt in rows:
        bias = mp - mt
        print(f"  ({lo:>3}, {hi:<3}]    {n:<8}{mp:<12.2f}{mt:<12.2f}{bias:+.2f}")

    print("\ndone.")


if __name__ == "__main__":
    main()
