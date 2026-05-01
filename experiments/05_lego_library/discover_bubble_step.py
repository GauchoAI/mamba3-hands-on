"""discover_bubble_step — Dreamer-style state-encoding discovery.

The system observes (a, b, did_swap) tuples and learns:
  1. A small discrete codebook (the state encoding).
  2. A decoder that maps each code to the correct action.

NO human-written harvest_pairs. NO told-in-advance encoding. The bubble
sort rule (true: 2 states, "a > b" vs "a <= b") should fall out of the
loss landscape by the system's own pressure to predict actions while
using a finite-codebook bottleneck.

Architecture:
  (a, b) → encoder MLP → logits over K codebook entries
        → Gumbel-softmax with hard=True (straight-through one-hot)
        → decoder linear → action logits (swap / no_swap)

We train with cross-entropy on the observed action. The bottleneck
forces the encoder to compress (a, b) into one of K discrete codes;
to predict perfectly, the partition over (a, b) space must respect
the underlying rule.

Demonstration:
  - Train with K=2 → if it converges, the system discovered the
    minimum-cardinality encoding.
  - Train with K=8 → how many codes actually get used? Should be ~2,
    proving that extra capacity isn't needed.
"""
import argparse, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiscoveryBubble(nn.Module):
    """Dreamer-style discrete bottleneck for bubble-sort discovery."""
    def __init__(self, K: int = 8, d_hidden: int = 16):
        super().__init__()
        self.K = K
        # Encoder: (a, b) → K-way logits
        self.enc = nn.Sequential(
            nn.Linear(2, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, K),
        )
        # Decoder: one-hot code → action logits (2 actions)
        self.dec = nn.Linear(K, 2)

    def forward(self, x: torch.Tensor, tau: float = 1.0):
        logits = self.enc(x)                                       # (N, K)
        # Straight-through Gumbel-softmax: forward returns one-hot,
        # backward uses the soft distribution for gradient flow.
        code = F.gumbel_softmax(logits, tau=tau, hard=True, dim=-1)
        action_logits = self.dec(code)                             # (N, 2)
        return action_logits, code, logits


def sample_batch(rng: np.random.Generator, batch: int):
    """Sample (a, b) ~ uniform[0, 1)^2 and the ground-truth bubble-sort
    action. The system *only* sees (a, b) → action; it never sees the
    'a > b' feature directly."""
    ab = rng.uniform(0, 1, (batch, 2)).astype(np.float32)
    action = (ab[:, 0] > ab[:, 1]).astype(np.int64)
    return ab, action


def train(K: int, steps: int = 2000, batch: int = 512, lr: float = 1e-2,
          device: str = "cpu", verbose: bool = True):
    rng = np.random.default_rng(0)
    model = DiscoveryBubble(K=K).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"K={K}  params={n_params}")
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    for step in range(steps):
        ab, action = sample_batch(rng, batch)
        x = torch.tensor(ab, device=device)
        y = torch.tensor(action, device=device)
        # Anneal Gumbel temperature from 1.0 to ~0.1.
        tau = max(0.1, 1.0 * (1.0 - step / steps))
        action_logits, code, _ = model(x, tau=tau)
        loss = F.cross_entropy(action_logits, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if verbose and (step + 1) % 200 == 0:
            with torch.no_grad():
                action_logits, code, _ = model(x, tau=0.01)
                acc = (action_logits.argmax(-1) == y).float().mean().item()
                code_idx = code.argmax(-1)
                n_used = int(torch.unique(code_idx).numel())
            print(f"  step {step+1:>4}  loss={loss.item():.4f}  acc={acc:.1%}  codes_used={n_used}/{K}")

    # Final eval on a large fresh sample
    eval_ab, eval_action = sample_batch(rng, 10_000)
    x = torch.tensor(eval_ab, device=device)
    y = torch.tensor(eval_action, device=device)
    with torch.no_grad():
        action_logits, code, _ = model(x, tau=0.01)
        acc = (action_logits.argmax(-1) == y).float().mean().item()
        code_idx = code.argmax(-1).cpu().numpy()
        n_used = len(np.unique(code_idx))
        usage = np.bincount(code_idx, minlength=K)

    if verbose:
        print(f"\nEvaluation on 10k fresh samples:")
        print(f"  accuracy: {acc:.2%}")
        print(f"  codes used: {n_used}/{K}")
        print(f"  code usage: {usage.tolist()}")
        # For each used code, decode to action and compare to ground truth
        print(f"\n  Discovered code → action mapping:")
        for c in np.unique(code_idx):
            one_hot = torch.zeros(1, K, device=device)
            one_hot[0, c] = 1.0
            a_pred = int(model.dec(one_hot).argmax(-1).item())
            mask = code_idx == c
            samples_in_code = int(mask.sum())
            true_swap_rate = float(eval_action[mask].mean())
            ab_in_code = eval_ab[mask]
            mean_a = float(ab_in_code[:, 0].mean())
            mean_b = float(ab_in_code[:, 1].mean())
            mean_diff = float((ab_in_code[:, 0] - ab_in_code[:, 1]).mean())
            print(f"    code {c}:  {samples_in_code:>5} samples   "
                  f"decoder→{['no_swap', 'swap'][a_pred]:<7}   "
                  f"true_swap_rate={true_swap_rate:.1%}   "
                  f"mean(a-b)={mean_diff:+.3f}")
        print(f"\n  Interpretation: if codes split cleanly along sign(a-b), "
              f"the system discovered the comparison feature on its own.")
    return acc, n_used, usage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K-list", type=int, nargs="+", default=[2, 4, 8])
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = ap.parse_args()
    print(f"Device: {args.device}")
    print()
    for K in args.K_list:
        print(f"\n══════════ Codebook size K = {K} ══════════")
        train(K=K, steps=args.steps, device=args.device)


if __name__ == "__main__":
    main()
