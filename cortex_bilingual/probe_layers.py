"""probe_layers.py — per-layer linear probe for "is this token in a unary run?"

Mechanistic-interpretability tier-A probe (no retraining): forward
the wider-N bilingual LM over a synthetic dataset of unary vs
non-unary byte sequences, capture each layer's residual output via
forward-hooks, and train a small linear probe per layer to predict
the binary label `(this byte position is mid-unary-run)`. The layer
whose probe wins tells us where the unary-mode feature lives — and
thus where a primitive should plug in to read it most cleanly.

Why this is the right next probe.
- The cortex thesis cares about *where* in the residual stream a
  primitive should attach. The current counter primitive plugs in
  at layer 0 by default. If the unary feature is sharpest at L3,
  we're reading from the wrong layer.
- Today's wider-N counter primitive showed a uniform +4 offset
  across all OOD N (10..500). That's calibration, not detection
  failure — meaning the gate fires on the right *byte*. The probe
  asks: does it fire on the right byte at the right *layer*?
- Forward-only, no gradients past the probe head. Runs in minutes
  on M4 Pro.

Run:
    python cortex_bilingual/probe_layers.py \\
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

from cortex_counting import CortexLM, CortexLMConfig


# ─────────────────────────────────────────────────────────────────────
# Synthetic probe dataset
# ─────────────────────────────────────────────────────────────────────

def make_probe_batch(n_seqs: int, seq_len: int, *,
                     n_min: int = 5, n_max: int = 60,
                     unary_p: float = 0.5,
                     seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a batch of byte sequences and per-position labels.

    Each sequence is one of:
      - Unary: random prefix bytes, then '*N:aaaa...a' for some N.
      - Non-unary: random ASCII bytes simulating language.

    Label is 1 at byte positions that are inside the `aaaa...` run
    (between the colon and the newline, exclusive of both); 0
    elsewhere. The per-layer probe learns to predict this from the
    layer's residual.

    Returns:
      ids:    (B, T) int64 byte ids
      labels: (B, T) int64 in {0, 1}
    """
    rng = random.Random(seed)
    ids = torch.zeros((n_seqs, seq_len), dtype=torch.long)
    labels = torch.zeros((n_seqs, seq_len), dtype=torch.long)

    for b in range(n_seqs):
        if rng.random() < unary_p:
            # Build "<prefix>*N:aaaa...a\n" filling to seq_len.
            prefix_len = rng.randint(0, max(0, seq_len // 4))
            prefix = bytes(rng.randint(32, 126) for _ in range(prefix_len))
            n = rng.randint(n_min, min(n_max, seq_len - prefix_len - 8))
            payload = (b"*" + str(n).encode() + b":" + b"a" * n + b"\n")
            seq = prefix + payload
            seq = seq[:seq_len]
            seq = seq + bytes(rng.randint(32, 126) for _ in range(seq_len - len(seq)))
            seq_arr = torch.tensor(list(seq), dtype=torch.long)
            # Label = 1 strictly between ':' and the next '\n'.
            colon_idx = (seq_arr == ord(":")).nonzero(as_tuple=True)[0]
            newline_idx = (seq_arr == ord("\n")).nonzero(as_tuple=True)[0]
            if len(colon_idx) and len(newline_idx):
                c = int(colon_idx[0])
                # First newline after colon
                nl_after = newline_idx[newline_idx > c]
                if len(nl_after):
                    nl = int(nl_after[0])
                    labels[b, c + 1: nl] = 1
            ids[b] = seq_arr
        else:
            # Random language-like bytes.
            seq = bytes(rng.randint(32, 126) for _ in range(seq_len))
            ids[b] = torch.tensor(list(seq), dtype=torch.long)
            # labels remain 0
    return ids, labels


# ─────────────────────────────────────────────────────────────────────
# Direct forward capture (CortexLM uses ModuleDict layers, so forward
# hooks don't fire on the container — we re-implement the loop instead.)
# ─────────────────────────────────────────────────────────────────────

@torch.no_grad()
def forward_with_taps(model, tokens: torch.Tensor) -> dict[int, torch.Tensor]:
    """Run CortexLM forward, capturing each layer's post-residual output.
    Returns {layer_idx: (B, T, D) cpu fp32}.
    Mirrors CortexLM.forward but skips primitives (we want the LM's own
    hidden state, not what a primitive injects)."""
    captures: dict[int, torch.Tensor] = {}
    x = model.embed(tokens)
    x = model.embed_norm(x)
    for i, layer in enumerate(model.layers):
        residual = x
        x = layer["norm"](x)
        x = layer["block"](x)
        x = residual + x
        captures[i] = x.detach().to("cpu", dtype=torch.float32)
    return captures


# ─────────────────────────────────────────────────────────────────────
# Probe (one-vs-rest logistic regression, per layer)
# ─────────────────────────────────────────────────────────────────────

def train_probe(features: torch.Tensor, labels: torch.Tensor,
                lr: float = 1e-2, steps: int = 200,
                weight_decay: float = 1e-4) -> tuple[nn.Linear, float]:
    """Linear logistic regression. Features (N, D), labels (N,) ∈ {0,1}.
    Returns (head, train_accuracy)."""
    N, D = features.shape
    head = nn.Linear(D, 2)
    opt = torch.optim.AdamW(head.parameters(), lr=lr,
                            weight_decay=weight_decay)
    for _ in range(steps):
        logits = head(features)
        loss = nn.functional.cross_entropy(logits, labels)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred = head(features).argmax(-1)
        acc = (pred == labels).float().mean().item()
    return head, acc


def eval_probe(head: nn.Linear, features: torch.Tensor,
               labels: torch.Tensor) -> dict:
    """Held-out accuracy + per-class metrics."""
    with torch.no_grad():
        logits = head(features)
        pred = logits.argmax(-1)
        acc = (pred == labels).float().mean().item()
        # Per-class
        if (labels == 1).any():
            tp = ((pred == 1) & (labels == 1)).sum().item()
            fn = ((pred == 0) & (labels == 1)).sum().item()
            recall_pos = tp / max(1, tp + fn)
        else:
            recall_pos = float("nan")
        if (labels == 0).any():
            tn = ((pred == 0) & (labels == 0)).sum().item()
            fp = ((pred == 1) & (labels == 0)).sum().item()
            recall_neg = tn / max(1, tn + fp)
        else:
            recall_neg = float("nan")
    return {"acc": acc, "recall_unary": recall_pos,
            "recall_lang": recall_neg}


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lm-ckpt", required=True)
    ap.add_argument("--n-train", type=int, default=512,
                    help="number of training sequences for probe")
    ap.add_argument("--n-test", type=int, default=256,
                    help="held-out evaluation sequences")
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--n-min", type=int, default=5)
    ap.add_argument("--n-max", type=int, default=60,
                    help="upper-bound on N for the in-distribution split")
    ap.add_argument("--n-max-ood", type=int, default=200,
                    help="upper-bound for the OOD split (N>60 = OOD here)")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    if args.device == "auto":
        args.device = "mps" if torch.backends.mps.is_available() else "cpu"
    device = args.device

    # ─── load LM ─────────────────────────────────────────────────────
    payload = torch.load(args.lm_ckpt, map_location="cpu", weights_only=False)
    cfg: CortexLMConfig = payload["cfg"]
    cfg.use_counter = False
    model = CortexLM(cfg).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    print(f"loaded LM: {sum(p.numel() for p in model.parameters()):,} params, "
          f"{cfg.n_layers}L d_model={cfg.d_model}")

    print(f"capturing residual at {cfg.n_layers} layers")

    def collect(ids: torch.Tensor) -> dict[int, torch.Tensor]:
        """Run forward, return per-layer (B, T, D) on CPU fp32."""
        return forward_with_taps(model, ids.to(device))

    # ─── ID split (N ≤ n-max, in distribution) ───────────────────────
    print(f"\n=== in-distribution split (N ∈ [{args.n_min}, {args.n_max}]) ===")
    ids_tr, lab_tr = make_probe_batch(args.n_train, args.seq_len,
                                      n_min=args.n_min, n_max=args.n_max,
                                      seed=0)
    ids_te, lab_te = make_probe_batch(args.n_test, args.seq_len,
                                      n_min=args.n_min, n_max=args.n_max,
                                      seed=1)
    feats_tr = collect(ids_tr)
    feats_te = collect(ids_te)

    # Flatten (B, T, D) → (B*T, D); mask out padded random positions
    # by keeping all positions (the probe will learn to ignore noise).
    def flatten(feats_dict, labels):
        out = {}
        L = labels.flatten()
        for li, t in feats_dict.items():
            out[li] = t.flatten(0, 1)
        return out, L

    Ftr, Ltr = flatten(feats_tr, lab_tr)
    Fte, Lte = flatten(feats_te, lab_te)

    # Class balance check
    pos_frac = float(Ltr.float().mean())
    print(f"  train: N={len(Ltr):,}  positive={pos_frac:.3f}")

    print(f"  {'layer':<8}{'train acc':<12}{'test acc':<12}"
          f"{'rec_unary':<12}{'rec_lang':<12}")

    in_dist_results = {}
    heads = {}
    for li in sorted(Ftr.keys()):
        head, tr_acc = train_probe(Ftr[li], Ltr)
        ev = eval_probe(head, Fte[li], Lte)
        in_dist_results[li] = ev
        heads[li] = head
        print(f"  L{li:<7}{tr_acc:<12.3f}{ev['acc']:<12.3f}"
              f"{ev['recall_unary']:<12.3f}{ev['recall_lang']:<12.3f}")

    # ─── OOD split (N > n-max) ───────────────────────────────────────
    print(f"\n=== OOD split (N ∈ ({args.n_max}, {args.n_max_ood}]) ===")
    ids_ood, lab_ood = make_probe_batch(args.n_test, args.seq_len,
                                        n_min=args.n_max + 1,
                                        n_max=args.n_max_ood, seed=2)
    feats_ood = collect(ids_ood)
    Food, Lood = flatten(feats_ood, lab_ood)
    pos_frac_ood = float(Lood.float().mean())
    print(f"  test: N={len(Lood):,}  positive={pos_frac_ood:.3f}")
    print(f"  {'layer':<8}{'OOD acc':<12}{'rec_unary':<12}{'rec_lang':<12}"
          f"{'Δ vs in-dist':<14}")
    for li in sorted(Food.keys()):
        ev = eval_probe(heads[li], Food[li], Lood)
        delta = ev["acc"] - in_dist_results[li]["acc"]
        print(f"  L{li:<7}{ev['acc']:<12.3f}{ev['recall_unary']:<12.3f}"
              f"{ev['recall_lang']:<12.3f}{delta:+.3f}")

    print("\ndone.")


if __name__ == "__main__":
    main()
