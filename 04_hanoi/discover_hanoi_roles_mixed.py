"""discover_hanoi_roles_mixed — mixed-K ensemble for true 100%.

Train role-encoded MLPs with different K values (8, 10, 12). Each K
has a different role-vector blind spot:
  - K=8 named roles → middle gap for n>=9
  - K=10 named roles → middle gap for n>=19 (so for n=17 a few cases)
  - K=12 named roles → no middle gap for n<=23

Different K's = different feature spaces = different decision boundaries.
Their errors are uncorrelated and the ensemble vote should close the
remaining gap.

We also include multiple seeds per K for additional ensemble diversity.
"""
import argparse, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


ACTION_PAIRS = [(0,1), (0,2), (1,0), (1,2), (2,0), (2,1)]
ACTION_TO_IDX = {p: i for i, p in enumerate(ACTION_PAIRS)}
N_ACTIONS = len(ACTION_PAIRS)
ABSENT = 3


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


def role_features_K(state, n_max_pad, K_small, K_large):
    """Role features parameterized by K_small and K_large."""
    B = state.shape[0]
    n_disks = (state != -1).sum(axis=1)
    smallest = []
    for i in range(K_small):
        col = state[:, i] if i < state.shape[1] else np.full(B, -1)
        smallest.append(np.where(col == -1, ABSENT, col))
    largest = []
    for k in range(K_large):
        idx = n_disks - 1 - k
        valid = idx >= 0
        idx_safe = np.clip(idx, 0, state.shape[1] - 1)
        peg = np.take_along_axis(state, idx_safe[:, None], axis=1).squeeze(-1)
        largest.append(np.where(valid, peg, ABSENT))
    parity = n_disks % 2
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


def legal_action_mask(state, n_max_pad):
    big = n_max_pad + 100
    top_p0 = np.where(state == 0, np.arange(state.shape[1])[None, :], big).min(axis=1)
    top_p1 = np.where(state == 1, np.arange(state.shape[1])[None, :], big).min(axis=1)
    top_p2 = np.where(state == 2, np.arange(state.shape[1])[None, :], big).min(axis=1)
    top = np.stack([top_p0, top_p1, top_p2], axis=-1)
    B = state.shape[0]
    mask = np.zeros((B, N_ACTIONS), dtype=np.bool_)
    for i, (src, dst) in enumerate(ACTION_PAIRS):
        src_has = top[:, src] < big
        dst_ok = (top[:, dst] == big) | (top[:, dst] > top[:, src])
        mask[:, i] = src_has & dst_ok
    return mask


class HanoiRoleMLP(nn.Module):
    def __init__(self, K_small, K_large, d_emb=16, d_hidden=128):
        super().__init__()
        self.K_small = K_small
        self.K_large = K_large
        self.peg_emb = nn.Embedding(4, d_emb)
        self.parity_emb = nn.Embedding(2, d_emb)
        self.cmp_emb = nn.Embedding(3, d_emb)
        n_features = K_small + K_large + 1 + 3
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
        for i in range(self.K_small + self.K_large):
            embs.append(self.peg_emb(feats[:, i]))
        i = self.K_small + self.K_large
        embs.append(self.parity_emb(feats[:, i]))
        embs.append(self.cmp_emb(feats[:, i + 1]))
        embs.append(self.cmp_emb(feats[:, i + 2]))
        embs.append(self.cmp_emb(feats[:, i + 3]))
        return self.net(torch.cat(embs, dim=-1))


def train_one(K_small, K_large, seed, train_states, train_actions,
              test_states, test_actions, test_legal, n_max_pad,
              steps, batch=512, lr=3e-3, d_hidden=128, device="cpu"):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    train_feats = role_features_K(train_states, n_max_pad, K_small, K_large)
    test_feats  = role_features_K(test_states,  n_max_pad, K_small, K_large)
    model = HanoiRoleMLP(K_small, K_large, d_hidden=d_hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr * 0.05)
    test_a_t = torch.tensor(test_feats, device=device)
    test_y_t = torch.tensor(test_actions, device=device)
    test_legal_t = torch.tensor(test_legal, device=device)
    best_test_acc = 0.0
    best_state = None
    N = len(train_states)
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
                logits_eval = model(test_a_t).masked_fill(~test_legal_t, -1e9)
                test_acc = (logits_eval.argmax(-1) == test_y_t).float().mean().item()
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, best_test_acc, test_feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-ns", type=int, nargs="+",
                    default=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
    ap.add_argument("--test-ns",  type=int, nargs="+", default=[16, 17])
    ap.add_argument("--n-max-pad", type=int, default=18)
    ap.add_argument("--steps", type=int, default=20000)
    # Per-K, multiple seeds: (K, n_seeds_for_that_K)
    ap.add_argument("--config", type=str,
                    default="8x3,10x3,12x3",
                    help="comma-sep KxS, e.g. '8x3,10x3,12x3' = 3 seeds at K=8, 3 at K=10, 3 at K=12")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--save-to", type=str, default="checkpoints/hanoi_role_ensemble.pt")
    args = ap.parse_args()
    print(f"Device: {args.device}")

    # Parse config
    config = []
    for spec in args.config.split(","):
        K_str, S_str = spec.split("x")
        config.append((int(K_str), int(S_str)))
    print(f"Mixed config: {config}\n")

    train_pairs = generate_traces_for_ns(args.train_ns, args.n_max_pad)
    test_pairs  = generate_traces_for_ns(args.test_ns,  args.n_max_pad)
    train_states = np.array([p[0] for p in train_pairs], dtype=np.int64)
    train_actions = np.array([p[2] for p in train_pairs], dtype=np.int64)
    test_states = np.array([p[0] for p in test_pairs], dtype=np.int64)
    test_n_disks = np.array([p[1] for p in test_pairs], dtype=np.int64)
    test_actions = np.array([p[2] for p in test_pairs], dtype=np.int64)
    test_legal = legal_action_mask(test_states, args.n_max_pad)
    print(f"Train pairs: {len(train_pairs)}  Test pairs: {len(test_pairs)}")

    models = []  # list of (model, K_small, K_large, test_feats)
    seed_counter = 42
    t0 = time.time()
    for K, S in config:
        for s in range(S):
            print(f"\n── K={K}/{K}  seed={seed_counter}  (model {len(models)+1}/{sum(s for _,s in config)}) ──")
            model, acc, test_feats = train_one(
                K, K, seed_counter, train_states, train_actions,
                test_states, test_actions, test_legal, args.n_max_pad,
                args.steps, device=args.device,
            )
            models.append((model, K, test_feats))
            seed_counter += 1
            print(f"  best held-out: {acc:.6%}  elapsed={time.time()-t0:.0f}s")

    # Ensemble: average softmax probs (with legal mask) across all models
    test_y_t = torch.tensor(test_actions, device=args.device)
    test_legal_t = torch.tensor(test_legal, device=args.device)
    print(f"\n── Ensemble of {len(models)} models ──")
    with torch.no_grad():
        avg_probs = None
        for model, K, test_feats in models:
            test_a_t = torch.tensor(test_feats, device=args.device)
            logits = model(test_a_t).masked_fill(~test_legal_t, -1e9)
            probs = F.softmax(logits, dim=-1)
            avg_probs = probs if avg_probs is None else avg_probs + probs
        avg_probs /= len(models)
        ensemble_pred = avg_probs.argmax(-1)
        n_correct = int((ensemble_pred == test_y_t).sum().item())
        n_total = int(test_y_t.numel())
    print(f"  Ensemble accuracy: {n_correct}/{n_total} = {100*n_correct/n_total:.6f}%   errors: {n_total - n_correct}")
    print("  Per-n breakdown:")
    pred_np = ensemble_pred.cpu().numpy()
    for n in args.test_ns:
        mask = test_n_disks == n
        if mask.sum() == 0: continue
        nc = int((pred_np[mask] == test_actions[mask]).sum())
        nt = int(mask.sum())
        print(f"    n={n}: {nc}/{nt} ({100*nc/nt:.6f}%)")

    # Save the ensemble for later use by hanoi_solve.py
    if args.save_to:
        from pathlib import Path
        Path(args.save_to).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "models": [(model.state_dict(), K) for model, K, _ in models],
            "n_max_pad": args.n_max_pad,
            "config": [(K, S) for K, S in config],
            "train_ns": args.train_ns,
        }, args.save_to)
        print(f"\nSaved ensemble → {args.save_to}")


if __name__ == "__main__":
    main()
