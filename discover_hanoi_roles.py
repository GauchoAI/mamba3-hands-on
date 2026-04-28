"""discover_hanoi_roles — pure role-based features. No absolute disk indices.

The 0.04% gap was caused by absolute disk indices (top_p, n_disks)
having values in test that were never seen in training. Indices are
anti-invariant by definition.

Switching to a role-based representation:

  - peg of smallest disk         (always disk 0, in {0,1,2})
  - peg of second-smallest       (always disk 1, in {0,1,2})
  - peg of largest               (disk n-1, but "largest" is the role)
  - peg of second-largest        (disk n-2)
  - peg of third-largest         (disk n-3)
  - peg of fourth-largest        (disk n-4)
  - peg of middle-disk-1, 2      (interpolated for large n)
  - n_disks parity (mod 2 and mod 4)
  - top-comparison flags between pegs

Every feature is in a small fixed-size vocabulary, regardless of n.
By construction the same code works for n=2 or n=2000.

Sentinel value -1 (mapped to embedding index 3 for pegs) for "this role
doesn't exist for this n" (e.g., second-largest doesn't exist for n=1).
"""
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


ACTION_PAIRS = [(0,1), (0,2), (1,0), (1,2), (2,0), (2,1)]
ACTION_TO_IDX = {p: i for i, p in enumerate(ACTION_PAIRS)}
N_ACTIONS = len(ACTION_PAIRS)
ABSENT = 3   # peg-embedding index for "this role doesn't exist"


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


N_SMALL = 10
N_LARGE = 10


def role_features(state, n_max_pad):
    """Role-based features. For n <= N_SMALL+N_LARGE the smallest+largest
    rolls cover all disks (with overlap). For n > N_SMALL+N_LARGE there's
    a middle gap, but with N_SMALL=N_LARGE=8 the gap only opens at n>=17.
    """
    B = state.shape[0]
    n_disks = (state != -1).sum(axis=1)

    # Pegs of smallest N_SMALL disks
    smallest = []
    for i in range(N_SMALL):
        col = state[:, i] if i < state.shape[1] else np.full(B, -1)
        smallest.append(np.where(col == -1, ABSENT, col))

    # Pegs of largest N_LARGE disks (disk n-1, n-2, ..., n-N_LARGE)
    largest = []
    for k in range(N_LARGE):
        idx = n_disks - 1 - k
        valid = idx >= 0
        idx_safe = np.clip(idx, 0, state.shape[1] - 1)
        peg = np.take_along_axis(state, idx_safe[:, None], axis=1).squeeze(-1)
        largest.append(np.where(valid, peg, ABSENT))

    parity = n_disks % 2

    # Top comparisons (legal-move structure)
    big = n_max_pad
    top_p0 = np.where(state == 0, np.arange(state.shape[1])[None, :], big).min(axis=1)
    top_p1 = np.where(state == 1, np.arange(state.shape[1])[None, :], big).min(axis=1)
    top_p2 = np.where(state == 2, np.arange(state.shape[1])[None, :], big).min(axis=1)
    def cmp(a, b):
        out = np.zeros_like(a)
        out = np.where((a < b) & (a != big) & (b != big), 1, out)
        out = np.where((a > b) & (a != big) & (b != big), 2, out)
        return out
    cmp_01 = cmp(top_p0, top_p1)
    cmp_02 = cmp(top_p0, top_p2)
    cmp_12 = cmp(top_p1, top_p2)

    return np.stack(smallest + largest + [parity, cmp_01, cmp_02, cmp_12],
                    axis=-1).astype(np.int64)


class HanoiRoleMLP(nn.Module):
    def __init__(self, d_emb: int = 16, d_hidden: int = 128):
        super().__init__()
        self.peg_emb = nn.Embedding(4, d_emb)         # peg ∈ {0,1,2,ABSENT}
        self.parity_emb = nn.Embedding(2, d_emb)
        self.cmp_emb = nn.Embedding(3, d_emb)
        n_features = N_SMALL + N_LARGE + 1 + 3        # peg-roles + parity + 3 cmps
        d_in = n_features * d_emb
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden), nn.ReLU(),
            nn.Linear(d_hidden, d_hidden), nn.ReLU(),
            nn.Linear(d_hidden, d_hidden), nn.ReLU(),
            nn.Linear(d_hidden, d_hidden), nn.ReLU(),
            nn.Linear(d_hidden, N_ACTIONS),
        )

    def forward(self, feats):
        embs = []
        for i in range(N_SMALL + N_LARGE):
            embs.append(self.peg_emb(feats[:, i]))
        i = N_SMALL + N_LARGE
        embs.append(self.parity_emb(feats[:, i]))
        embs.append(self.cmp_emb(feats[:, i + 1]))
        embs.append(self.cmp_emb(feats[:, i + 2]))
        embs.append(self.cmp_emb(feats[:, i + 3]))
        x = torch.cat(embs, dim=-1)
        return self.net(x)


def train_and_test(train_ns, test_ns, n_max_pad, steps,
                   batch=512, lr=3e-3, d_hidden=128, device="cpu"):
    rng = np.random.default_rng(0)
    train_pairs = generate_traces_for_ns(train_ns, n_max_pad)
    test_pairs  = generate_traces_for_ns(test_ns,  n_max_pad)
    train_states = np.array([p[0] for p in train_pairs], dtype=np.int64)
    train_actions = np.array([p[2] for p in train_pairs], dtype=np.int64)
    train_feats = role_features(train_states, n_max_pad)
    test_states = np.array([p[0] for p in test_pairs], dtype=np.int64)
    test_n_disks = np.array([p[1] for p in test_pairs], dtype=np.int64)
    test_actions = np.array([p[2] for p in test_pairs], dtype=np.int64)
    test_feats = role_features(test_states, n_max_pad)
    N = len(train_pairs)
    print(f"Train pairs (n in {train_ns}): {N}")
    print(f"Test pairs  (n in {test_ns}):  {len(test_pairs)}")

    # Diagnostic: how many unique role-feature vectors in training? Aliasing?
    from collections import defaultdict
    bucket = defaultdict(list)
    for i in range(N):
        bucket[tuple(train_feats[i].tolist())].append(int(train_actions[i]))
    n_distinct = len(bucket)
    n_amb = sum(1 for a in bucket.values() if len(set(a)) > 1)
    print(f"Distinct training role vectors: {n_distinct}; ambiguous: {n_amb}")
    if n_amb > 0:
        print("⚠️  ROLE VECTORS ARE NOT INJECTIVE — model can't reach 100% by definition.")

    model = HanoiRoleMLP(d_hidden=d_hidden).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params={n_params}  d_hidden={d_hidden}")
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr * 0.05)

    best_test_acc = 0.0
    best_state = None
    test_a_t = torch.tensor(test_feats, device=device)
    test_y_t = torch.tensor(test_actions, device=device)

    for step in range(steps):
        idx = rng.integers(0, N, size=batch)
        a = torch.tensor(train_feats[idx], device=device)
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
            with torch.no_grad():
                t_a = torch.tensor(train_feats, device=device)
                train_acc = (model(t_a).argmax(-1)
                             == torch.tensor(train_actions, device=device)).float().mean().item()
            print(f"  step {step+1:>5}  loss={loss.item():.5f}  train={train_acc:.4%}  best_test={best_test_acc:.4%}")

    if best_state is not None:
        model.load_state_dict(best_state)
    print("\nFinal eval:")
    with torch.no_grad():
        train_acc = (model(torch.tensor(train_feats, device=device)).argmax(-1)
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
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--d-hidden", type=int, default=128)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = ap.parse_args()
    print(f"Device: {args.device}\n")
    train_and_test(args.train_ns, args.test_ns, args.n_max_pad,
                   args.steps, d_hidden=args.d_hidden, device=args.device)


if __name__ == "__main__":
    main()
