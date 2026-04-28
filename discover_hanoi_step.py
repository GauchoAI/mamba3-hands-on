"""discover_hanoi_step — Tower of Hanoi rule discovery from raw traces.

Generate optimal Hanoi solutions for n=2..N, observe (state, next_move)
pairs at every step, and train a discrete-bottleneck network to find the
state encoding from scratch. No role-based encoding hint.

Input to encoder: padded peg-positions array — for max disks N_MAX,
state is a length-N_MAX vector where state[i] = peg of disk i (smallest
to largest, 0..2), or -1 if disk doesn't exist for this n. We also
provide n_disks as an extra feature.

Action: one of 6 (src, dst) pairs.

The expected best-case behavior: the system finds an encoding with
~36 codes (matching the role-based encoding's cardinality).
"""
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# Action enumeration: (src, dst) → action_idx
ACTION_PAIRS = [(0,1), (0,2), (1,0), (1,2), (2,0), (2,1)]
ACTION_TO_IDX = {p: i for i, p in enumerate(ACTION_PAIRS)}
N_ACTIONS = len(ACTION_PAIRS)


def hanoi_moves(n: int, src: int = 0, dst: int = 2, aux: int = 1):
    """Yield each (src, dst) move for the standard recursive solution."""
    if n == 0:
        return
    yield from hanoi_moves(n - 1, src, aux, dst)
    yield (src, dst)
    yield from hanoi_moves(n - 1, aux, dst, src)


def generate_traces(n_max: int):
    """For each n in 2..n_max, generate the optimal trace as a list of
    (state, action) pairs. State is a peg-positions array padded with -1
    to length n_max.

    Returns list of (state_padded, n_disks, action_idx) tuples."""
    pairs = []
    for n in range(2, n_max + 1):
        # Initial state: all disks on peg 0
        pegs = [0] * n_max
        for i in range(n):
            pegs[i] = 0
        for i in range(n, n_max):
            pegs[i] = -1
        for src, dst in hanoi_moves(n):
            # Find which disk is moving: smallest disk on src
            disk_to_move = None
            for d in range(n):
                if pegs[d] == src:
                    disk_to_move = d
                    break
            assert disk_to_move is not None, f"No disk on peg {src}"
            # Snapshot pre-move state
            state = pegs.copy()
            action = ACTION_TO_IDX[(src, dst)]
            pairs.append((state, n, action))
            # Apply the move
            pegs[disk_to_move] = dst
    return pairs


class DiscoveryHanoi(nn.Module):
    def __init__(self, n_max: int, K: int = 64, d_emb: int = 8, d_hidden: int = 64):
        super().__init__()
        self.n_max = n_max
        self.K = K
        # Embed each disk's peg (4 values: -1=absent, 0,1,2 = pegs)
        # Shift to 0..3 for embedding lookup.
        self.peg_emb = nn.Embedding(4, d_emb)
        # Embed n_disks (2..n_max)
        self.n_emb = nn.Embedding(n_max + 1, d_emb)
        d_in = n_max * d_emb + d_emb
        self.enc = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, K),
        )
        self.dec = nn.Linear(K, N_ACTIONS)

    def forward(self, state, n_disks, tau=1.0):
        # state: (B, n_max) ints in {-1, 0, 1, 2}; shift to {0..3}.
        peg_e = self.peg_emb(state + 1)                          # (B, n_max, d_emb)
        peg_e_flat = peg_e.flatten(1)                            # (B, n_max·d_emb)
        n_e = self.n_emb(n_disks)                                # (B, d_emb)
        x = torch.cat([peg_e_flat, n_e], dim=-1)
        logits = self.enc(x)
        code = F.gumbel_softmax(logits, tau=tau, hard=True, dim=-1)
        return self.dec(code), code, logits


def train(K: int, n_max: int = 5, steps: int = 4000, batch: int = 256,
          lr: float = 5e-3, device: str = "cpu", verbose: bool = True):
    rng = np.random.default_rng(0)
    pairs = generate_traces(n_max)
    if verbose:
        print(f"Generated {len(pairs)} (state, action) pairs from n=2..{n_max}")
    states = np.array([p[0] for p in pairs], dtype=np.int64)
    n_disks = np.array([p[1] for p in pairs], dtype=np.int64)
    actions = np.array([p[2] for p in pairs], dtype=np.int64)
    N = len(pairs)

    model = DiscoveryHanoi(n_max=n_max, K=K).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"K={K}  params={n_params}")
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    for step in range(steps):
        idx = rng.integers(0, N, size=batch)
        s = torch.tensor(states[idx], device=device)
        nd = torch.tensor(n_disks[idx], device=device)
        y = torch.tensor(actions[idx], device=device)
        tau = max(0.2, 1.0 * (1.0 - step / steps))
        action_logits, code, _ = model(s, nd, tau=tau)
        loss = F.cross_entropy(action_logits, y)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()

        if verbose and (step + 1) % 400 == 0:
            with torch.no_grad():
                action_logits, code, _ = model(s, nd, tau=0.05)
                acc = (action_logits.argmax(-1) == y).float().mean().item()
                code_idx = code.argmax(-1)
                n_used = int(torch.unique(code_idx).numel())
            print(f"  step {step+1:>4}  loss={loss.item():.4f}  acc={acc:.1%}  codes_used={n_used}/{K}")

    # Eval on full dataset
    s = torch.tensor(states, device=device)
    nd = torch.tensor(n_disks, device=device)
    y = torch.tensor(actions, device=device)
    with torch.no_grad():
        action_logits, code, _ = model(s, nd, tau=0.05)
        acc = (action_logits.argmax(-1) == y).float().mean().item()
        code_idx = code.argmax(-1).cpu().numpy()
        n_used = len(np.unique(code_idx))

    if verbose:
        print(f"\nFull-trace evaluation ({N} pairs):")
        print(f"  accuracy: {acc:.2%}")
        print(f"  codes used: {n_used}/{K}")
        # Per-n breakdown
        for n in range(2, n_max + 1):
            mask = n_disks == n
            if mask.sum() == 0:
                continue
            preds = action_logits[torch.tensor(mask)].argmax(-1).cpu().numpy()
            true_acts = actions[mask]
            n_correct = int((preds == true_acts).sum())
            n_total = int(mask.sum())
            print(f"  n={n}: {n_correct}/{n_total} ({100*n_correct/n_total:.1f}%)  trace_len={n_total}")
    return acc, n_used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K-list", type=int, nargs="+", default=[32, 48, 64])
    ap.add_argument("--n-max", type=int, default=5)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = ap.parse_args()
    print(f"Device: {args.device}")
    for K in args.K_list:
        print(f"\n══════════ Codebook size K = {K} ══════════")
        train(K=K, n_max=args.n_max, steps=args.steps, device=args.device)


if __name__ == "__main__":
    main()
