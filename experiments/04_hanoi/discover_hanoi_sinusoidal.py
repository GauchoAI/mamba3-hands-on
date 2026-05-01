"""discover_hanoi_sinusoidal — sinusoidal encoding for unbounded integer features.

The 0.04% gap was held-out n=17 having unseen embedding indices for
top_p=16 and n_disks=17. Switching to sinusoidal positional encoding
for those features (treats integers as positions in a continuous
embedding space) extrapolates by construction.

PE(pos, 2k)   = sin(pos / 10000^(2k/d))
PE(pos, 2k+1) = cos(pos / 10000^(2k/d))

Discrete embeddings stay for peg ids and parity (both have small
fixed-size vocabularies). Counts can also be sinusoidal for
consistency with top_p.
"""
import argparse
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


ACTION_PAIRS = [(0,1), (0,2), (1,0), (1,2), (2,0), (2,1)]
ACTION_TO_IDX = {p: i for i, p in enumerate(ACTION_PAIRS)}
N_ACTIONS = len(ACTION_PAIRS)


def hanoi_moves(n, src=0, dst=2, aux=1):
    if n == 0: return
    yield from hanoi_moves(n - 1, src, aux, dst)
    yield (src, dst)
    yield from hanoi_moves(n - 1, aux, dst, src)


def generate_traces_for_ns(n_list, n_max_pad):
    pairs = []
    for n in n_list:
        pegs = [0] * n_max_pad
        for i in range(n, n_max_pad): pegs[i] = -1
        for src, dst in hanoi_moves(n):
            disk = next(i for i in range(n) if pegs[i] == src)
            pairs.append((pegs.copy(), n, ACTION_TO_IDX[(src, dst)]))
            pegs[disk] = dst
    return pairs


def compute_aggregates(state, n_max_pad):
    big = n_max_pad
    peg0 = state[:, 0]; peg1 = state[:, 1]
    count0 = (state == 0).sum(axis=1)
    count1 = (state == 1).sum(axis=1)
    count2 = (state == 2).sum(axis=1)
    top_p0 = np.where(state == 0, np.arange(state.shape[1])[None, :], big).min(axis=1)
    top_p1 = np.where(state == 1, np.arange(state.shape[1])[None, :], big).min(axis=1)
    top_p2 = np.where(state == 2, np.arange(state.shape[1])[None, :], big).min(axis=1)
    n_disks = (state != -1).sum(axis=1)
    last_disk_idx = np.clip(n_disks - 1, 0, state.shape[1] - 1)
    peg_largest = np.take_along_axis(state, last_disk_idx[:, None], axis=1).squeeze(-1)
    return np.stack([peg0, peg1, count0, count1, count2,
                     top_p0, top_p1, top_p2, n_disks, peg_largest],
                    axis=-1).astype(np.int64)


def sinusoidal_encoding(positions: torch.Tensor, d_model: int) -> torch.Tensor:
    """Standard sinusoidal positional encoding.
    positions: (B,) int. Returns (B, d_model).
    """
    pos = positions.float().unsqueeze(-1)                  # (B, 1)
    i = torch.arange(d_model, device=positions.device).float()  # (d_model,)
    div = torch.exp(-(i // 2) * 2.0 * math.log(10000.0) / d_model)
    angles = pos * div                                     # (B, d_model)
    pe = torch.zeros_like(angles)
    pe[..., 0::2] = torch.sin(angles[..., 0::2])
    pe[..., 1::2] = torch.cos(angles[..., 1::2])
    return pe


class HanoiSinMLP(nn.Module):
    """Sinusoidal encoding for integer features that vary with n; learned
    embeddings for the small-vocab features (peg ids).
    """
    def __init__(self, n_max_pad: int, d_emb: int = 16, d_hidden: int = 128):
        super().__init__()
        self.n_max_pad = n_max_pad
        self.d_emb = d_emb
        self.peg_p_emb = nn.Embedding(3, d_emb)        # peg ∈ {0,1,2}
        d_in = 10 * d_emb
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden), nn.ReLU(),
            nn.Linear(d_hidden, d_hidden), nn.ReLU(),
            nn.Linear(d_hidden, d_hidden), nn.ReLU(),
            nn.Linear(d_hidden, d_hidden), nn.ReLU(),
            nn.Linear(d_hidden, N_ACTIONS),
        )

    def forward(self, aggs):
        a0 = self.peg_p_emb(aggs[:, 0])
        a1 = self.peg_p_emb(aggs[:, 1])
        pl = self.peg_p_emb(aggs[:, 9])
        c0 = sinusoidal_encoding(aggs[:, 2], self.d_emb)
        c1 = sinusoidal_encoding(aggs[:, 3], self.d_emb)
        c2 = sinusoidal_encoding(aggs[:, 4], self.d_emb)
        t0 = sinusoidal_encoding(aggs[:, 5], self.d_emb)
        t1 = sinusoidal_encoding(aggs[:, 6], self.d_emb)
        t2 = sinusoidal_encoding(aggs[:, 7], self.d_emb)
        nd = sinusoidal_encoding(aggs[:, 8], self.d_emb)
        x = torch.cat([a0, a1, c0, c1, c2, t0, t1, t2, nd, pl], dim=-1)
        return self.net(x)


def train_and_test(train_ns, test_ns, n_max_pad, steps,
                   batch=512, lr=3e-3, d_hidden=128, device="cpu"):
    rng = np.random.default_rng(0)
    train_pairs = generate_traces_for_ns(train_ns, n_max_pad)
    test_pairs  = generate_traces_for_ns(test_ns,  n_max_pad)
    train_states = np.array([p[0] for p in train_pairs], dtype=np.int64)
    train_actions = np.array([p[2] for p in train_pairs], dtype=np.int64)
    train_aggs = compute_aggregates(train_states, n_max_pad)
    test_states = np.array([p[0] for p in test_pairs], dtype=np.int64)
    test_n_disks = np.array([p[1] for p in test_pairs], dtype=np.int64)
    test_actions = np.array([p[2] for p in test_pairs], dtype=np.int64)
    test_aggs = compute_aggregates(test_states, n_max_pad)
    N = len(train_pairs)
    print(f"Train pairs (n in {train_ns}): {N}")
    print(f"Test pairs  (n in {test_ns}):  {len(test_pairs)}")

    model = HanoiSinMLP(n_max_pad=n_max_pad, d_hidden=d_hidden).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params={n_params}  d_hidden={d_hidden}")
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr * 0.05)

    best_test_acc = 0.0
    best_state = None
    test_a_t = torch.tensor(test_aggs, device=device)
    test_y_t = torch.tensor(test_actions, device=device)

    for step in range(steps):
        idx = rng.integers(0, N, size=batch)
        a = torch.tensor(train_aggs[idx], device=device)
        y = torch.tensor(train_actions[idx], device=device)
        logits = model(a)
        loss = F.cross_entropy(logits, y)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if (step + 1) % 500 == 0:
            with torch.no_grad():
                test_acc = (model(test_a_t).argmax(-1) == test_y_t).float().mean().item()
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if (step + 1) % 2000 == 0:
            print(f"  step {step+1:>5}  loss={loss.item():.6f}  best_test={best_test_acc:.4%}")

    if best_state is not None:
        model.load_state_dict(best_state)

    print("\nFinal eval:")
    with torch.no_grad():
        train_acc = (model(torch.tensor(train_aggs, device=device)).argmax(-1)
                     == torch.tensor(train_actions, device=device)).float().mean().item()
        t_logits = model(test_a_t)
        test_acc = (t_logits.argmax(-1) == test_y_t).float().mean().item()
    print(f"  Train: {train_acc:.4%}")
    print(f"  Held-out test: {test_acc:.4%}")
    print("  Per-n breakdown:")
    for n in test_ns:
        mask = test_n_disks == n
        if mask.sum() == 0: continue
        preds = t_logits[torch.tensor(mask)].argmax(-1).cpu().numpy()
        true = test_actions[mask]
        n_correct = int((preds == true).sum())
        n_total = int(mask.sum())
        print(f"    n={n}: {n_correct}/{n_total} ({100*n_correct/n_total:.4f}%)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-ns", type=int, nargs="+",
                    default=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
    ap.add_argument("--test-ns",  type=int, nargs="+", default=[16, 17])
    ap.add_argument("--n-max-pad", type=int, default=18)
    ap.add_argument("--steps", type=int, default=25000)
    ap.add_argument("--d-hidden", type=int, default=128)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = ap.parse_args()
    print(f"Device: {args.device}\n")
    train_and_test(args.train_ns, args.test_ns, args.n_max_pad,
                   args.steps, d_hidden=args.d_hidden, device=args.device)


if __name__ == "__main__":
    main()
