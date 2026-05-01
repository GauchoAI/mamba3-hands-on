"""discover_gcd_step — same harness, GCD by subtraction.

Observations: (a, b, action) for random integer pairs.
Actions: 0=sub_b_from_a, 1=sub_a_from_b, 2=done

True minimum closed state space:
  - "a > b, both > 0"    → sub_b_from_a
  - "b > a, both > 0"    → sub_a_from_b
  - terminal (a==0 or b==0 or a==b) → done

So 3 actions, with the terminal action triggered by 3 distinct conditions
that all map to the same action. Will the system fold them into one code,
or use one code per condition? Either is valid; we'll see.
"""
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def gcd_action(a: int, b: int) -> int:
    """0=sub_b_from_a, 1=sub_a_from_b, 2=done."""
    if a == 0 or b == 0 or a == b:
        return 2
    if a > b:
        return 0
    return 1


class DiscoveryGCD(nn.Module):
    def __init__(self, K: int = 8, d_hidden: int = 32):
        super().__init__()
        self.K = K
        # Encode (a, b) as floats normalized to [0, 1]. The encoder must
        # discover features like "a > b" and "a == 0" on its own.
        self.enc = nn.Sequential(
            nn.Linear(2, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, K),
        )
        self.dec = nn.Linear(K, 3)

    def forward(self, x: torch.Tensor, tau: float = 1.0):
        logits = self.enc(x)
        code = F.gumbel_softmax(logits, tau=tau, hard=True, dim=-1)
        action_logits = self.dec(code)
        return action_logits, code, logits


def sample_batch(rng: np.random.Generator, batch: int, n_max: int = 50):
    """Sample (a, b) ∈ {0..n_max}². Bias slightly toward terminal cases
    (a==0, b==0, a==b) so they get enough training signal."""
    ab = rng.integers(0, n_max + 1, size=(batch, 2))
    # Force ~10% of samples into terminal cases for training coverage.
    n_term = batch // 10
    if n_term > 0:
        which = rng.integers(0, 3, size=n_term)
        for i, w in enumerate(which):
            if w == 0:    ab[i] = (0, rng.integers(0, n_max + 1))
            elif w == 1:  ab[i] = (rng.integers(0, n_max + 1), 0)
            else:
                v = rng.integers(0, n_max + 1)
                ab[i] = (v, v)
    actions = np.array([gcd_action(int(a), int(b)) for a, b in ab], dtype=np.int64)
    # Normalize to [0, 1] for the encoder
    ab_norm = (ab.astype(np.float32) / n_max)
    return ab_norm, ab, actions


def train(K: int, steps: int = 3000, batch: int = 512, lr: float = 1e-2,
          n_max: int = 50, device: str = "cpu", verbose: bool = True):
    rng = np.random.default_rng(0)
    model = DiscoveryGCD(K=K).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"K={K}  params={n_params}")
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    for step in range(steps):
        ab_norm, _, actions = sample_batch(rng, batch, n_max=n_max)
        x = torch.tensor(ab_norm, device=device)
        y = torch.tensor(actions, device=device)
        tau = max(0.1, 1.0 * (1.0 - step / steps))
        action_logits, code, _ = model(x, tau=tau)
        loss = F.cross_entropy(action_logits, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if verbose and (step + 1) % 300 == 0:
            with torch.no_grad():
                action_logits, code, _ = model(x, tau=0.01)
                acc = (action_logits.argmax(-1) == y).float().mean().item()
                code_idx = code.argmax(-1)
                n_used = int(torch.unique(code_idx).numel())
            print(f"  step {step+1:>4}  loss={loss.item():.4f}  acc={acc:.1%}  codes_used={n_used}/{K}")

    # Eval on a fresh sample
    eval_ab_norm, eval_ab, eval_action = sample_batch(rng, 5000, n_max=n_max)
    x = torch.tensor(eval_ab_norm, device=device)
    y = torch.tensor(eval_action, device=device)
    with torch.no_grad():
        action_logits, code, _ = model(x, tau=0.01)
        acc = (action_logits.argmax(-1) == y).float().mean().item()
        code_idx = code.argmax(-1).cpu().numpy()
        n_used = len(np.unique(code_idx))
        usage = np.bincount(code_idx, minlength=K)

    if verbose:
        print(f"\nEvaluation on 5000 fresh samples:")
        print(f"  accuracy: {acc:.2%}")
        print(f"  codes used: {n_used}/{K}")
        print(f"\n  Code → action + sample-distribution:")
        action_names = ["sub_b_from_a", "sub_a_from_b", "done"]
        for c in np.unique(code_idx):
            mask = code_idx == c
            n_in_code = int(mask.sum())
            ab_in_code = eval_ab[mask]
            actions_in_code = eval_action[mask]
            one_hot = torch.zeros(1, K, device=device)
            one_hot[0, c] = 1.0
            a_pred = int(model.dec(one_hot).argmax(-1).item())
            # Distribution of true actions for samples landing in this code
            action_dist = np.bincount(actions_in_code, minlength=3) / max(n_in_code, 1)
            # Distribution over the (a>b, b>a, a=0, b=0) features
            a_arr, b_arr = ab_in_code[:, 0], ab_in_code[:, 1]
            f_a_gt_b = (a_arr > b_arr).mean()
            f_b_gt_a = (b_arr > a_arr).mean()
            f_a_zero = (a_arr == 0).mean()
            f_b_zero = (b_arr == 0).mean()
            f_eq    = (a_arr == b_arr).mean()
            print(f"    code {c}:  {n_in_code:>5} samples  →  {action_names[a_pred]:<14}")
            print(f"             true action dist: sub_a={action_dist[0]:.2f}  sub_b={action_dist[1]:.2f}  done={action_dist[2]:.2f}")
            print(f"             feature   freq:   a>b={f_a_gt_b:.2f}  b>a={f_b_gt_a:.2f}  a=0={f_a_zero:.2f}  b=0={f_b_zero:.2f}  a==b={f_eq:.2f}")
    return acc, n_used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K-list", type=int, nargs="+", default=[3, 5, 8])
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = ap.parse_args()
    print(f"Device: {args.device}")
    for K in args.K_list:
        print(f"\n══════════ Codebook size K = {K} ══════════")
        train(K=K, steps=args.steps, device=args.device)


if __name__ == "__main__":
    main()
