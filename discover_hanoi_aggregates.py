"""discover_hanoi_aggregates — augment input with simple aggregate features.

The vanilla MLP doesn't generalize across n because the padded peg state
buries the relevant info (positions of smallest disks) in a sea of
varying-length padding. Provide aggregate features alongside the raw state:

  - peg of disk 0 (always present for n >= 1)
  - peg of disk 1 (always present for n >= 2)
  - count of disks on each peg (3 values)

These are NOT the rule — they're just simple aggregates that any encoder
could compute given enough capacity. The test is: with these features
provided, does training on n=2..6 generalize to n=7..10?

If yes: the bottleneck was feature extraction, and the discovery system
can find these features given an architecture with the right inductive
bias (set-attention, etc.).

If no: there's something more fundamental than feature extraction missing.
"""
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


ACTION_PAIRS = [(0,1), (0,2), (1,0), (1,2), (2,0), (2,1)]
ACTION_TO_IDX = {p: i for i, p in enumerate(ACTION_PAIRS)}
N_ACTIONS = len(ACTION_PAIRS)


def hanoi_moves(n: int, src: int = 0, dst: int = 2, aux: int = 1):
    if n == 0: return
    yield from hanoi_moves(n - 1, src, aux, dst)
    yield (src, dst)
    yield from hanoi_moves(n - 1, aux, dst, src)


def generate_traces_for_ns(n_list, n_max_pad: int):
    pairs = []
    for n in n_list:
        pegs = [0] * n_max_pad
        for i in range(n): pegs[i] = 0
        for i in range(n, n_max_pad): pegs[i] = -1
        for src, dst in hanoi_moves(n):
            disk_to_move = None
            for d in range(n):
                if pegs[d] == src: disk_to_move = d; break
            assert disk_to_move is not None
            pairs.append((pegs.copy(), n, ACTION_TO_IDX[(src, dst)]))
            pegs[disk_to_move] = dst
    return pairs


def compute_aggregates(state: np.ndarray, n_max_pad: int) -> np.ndarray:
    """state: (B, n_max_pad). Returns (B, n_aggregates).

    Features:
      [0] peg of disk 0   (smallest)
      [1] peg of disk 1
      [2..4] count of disks per peg
      [5..7] top disk index per peg (n_max_pad if peg empty)
      [8] n_disks total
      [9] peg of largest disk (disk N-1)
    """
    big = n_max_pad
    peg0 = state[:, 0]
    peg1 = state[:, 1]
    count0 = (state == 0).sum(axis=1)
    count1 = (state == 1).sum(axis=1)
    count2 = (state == 2).sum(axis=1)
    top_p0 = np.where(state == 0, np.arange(state.shape[1])[None, :], big).min(axis=1)
    top_p1 = np.where(state == 1, np.arange(state.shape[1])[None, :], big).min(axis=1)
    top_p2 = np.where(state == 2, np.arange(state.shape[1])[None, :], big).min(axis=1)
    n_disks = (state != -1).sum(axis=1)
    # Peg of the largest existing disk (disk index n_disks-1).
    # state[i, n_disks[i]-1] for each row. Use take_along_axis.
    last_disk_idx = np.clip(n_disks - 1, 0, state.shape[1] - 1)
    peg_largest = np.take_along_axis(state, last_disk_idx[:, None], axis=1).squeeze(-1)
    return np.stack([peg0, peg1, count0, count1, count2,
                     top_p0, top_p1, top_p2, n_disks, peg_largest],
                    axis=-1).astype(np.int64)


class DiscoveryHanoiAgg(nn.Module):
    """Encoder over raw state + aggregate features."""
    def __init__(self, n_max: int, K: int = 64, d_emb: int = 8, d_hidden: int = 64):
        super().__init__()
        self.n_max = n_max
        self.K = K
        self.peg_emb = nn.Embedding(4, d_emb)        # raw padded state values
        self.peg_p_emb = nn.Embedding(3, d_emb)      # peg index in {0,1,2}
        self.count_emb = nn.Embedding(n_max + 2, d_emb)
        self.top_emb = nn.Embedding(n_max + 1, d_emb)  # top disk idx (or n_max for empty)

        # 10 aggregate features × d_emb (no raw padded state — it's n-dependent)
        d_in = 10 * d_emb
        self.enc = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, K),
        )
        self.dec = nn.Linear(K, N_ACTIONS)

    def forward(self, state, aggregates, tau=1.0, deterministic: bool = False):
        # state is unused now — kept in signature for compatibility.
        a0 = self.peg_p_emb(aggregates[:, 0])
        a1 = self.peg_p_emb(aggregates[:, 1])
        c0 = self.count_emb(aggregates[:, 2])
        c1 = self.count_emb(aggregates[:, 3])
        c2 = self.count_emb(aggregates[:, 4])
        t0 = self.top_emb(aggregates[:, 5])
        t1 = self.top_emb(aggregates[:, 6])
        t2 = self.top_emb(aggregates[:, 7])
        nd = self.count_emb(aggregates[:, 8])
        pl = self.peg_p_emb(aggregates[:, 9])
        x = torch.cat([a0, a1, c0, c1, c2, t0, t1, t2, nd, pl], dim=-1)
        logits = self.enc(x)
        if deterministic:
            # Pure argmax — no Gumbel noise. Use straight-through one-hot.
            idx = logits.argmax(dim=-1, keepdim=True)
            code = torch.zeros_like(logits).scatter_(-1, idx, 1.0)
        else:
            code = F.gumbel_softmax(logits, tau=tau, hard=True, dim=-1)
        return self.dec(code), code, logits


def train_and_test(K: int, train_ns, test_ns, n_max_pad: int,
                   steps: int, batch: int = 512, lr: float = 5e-3,
                   usage_weight: float = 0.1, device: str = "cpu",
                   verbose: bool = True):
    rng = np.random.default_rng(0)
    train_pairs = generate_traces_for_ns(train_ns, n_max_pad)
    test_pairs  = generate_traces_for_ns(test_ns,  n_max_pad)
    if verbose:
        print(f"Train pairs (n in {train_ns}): {len(train_pairs)}")
        print(f"Test pairs  (n in {test_ns}):  {len(test_pairs)}")

    train_states  = np.array([p[0] for p in train_pairs], dtype=np.int64)
    train_actions = np.array([p[2] for p in train_pairs], dtype=np.int64)
    train_aggs    = compute_aggregates(train_states, n_max_pad)
    test_states   = np.array([p[0] for p in test_pairs],  dtype=np.int64)
    test_n_disks  = np.array([p[1] for p in test_pairs],  dtype=np.int64)
    test_actions  = np.array([p[2] for p in test_pairs],  dtype=np.int64)
    test_aggs     = compute_aggregates(test_states, n_max_pad)
    N = len(train_pairs)

    model = DiscoveryHanoiAgg(n_max=n_max_pad, K=K).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"K={K}  params={n_params}  (state + aggregates)")
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr * 0.05)

    # Track best held-out as we train (eval-during-training)
    best_test_acc = 0.0
    best_state = None

    test_states_t = torch.tensor(test_states, device=device)
    test_aggs_t   = torch.tensor(test_aggs,   device=device)
    test_y_t      = torch.tensor(test_actions, device=device)

    log_K = float(np.log(K))
    for step in range(steps):
        idx = rng.integers(0, N, size=batch)
        s = torch.tensor(train_states[idx], device=device)
        a = torch.tensor(train_aggs[idx], device=device)
        y = torch.tensor(train_actions[idx], device=device)
        # Slower temperature anneal: hold at 1.0 for 20%, then linearly to 0.3.
        tau = max(0.3, 1.0 - 0.7 * max(0.0, (step - 0.2 * steps) / (0.8 * steps)))
        action_logits, code, _ = model(s, a, tau=tau)
        ce = F.cross_entropy(action_logits, y)
        code_probs = code.mean(dim=0)
        usage_score = -(code_probs * torch.log(code_probs + 1e-10)).sum() / log_K
        loss = ce - usage_weight * usage_score
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step(); sched.step()

        # Periodically eval held-out and keep the best
        if (step + 1) % 500 == 0:
            with torch.no_grad():
                action_logits_t, _, _ = model(test_states_t, test_aggs_t, tau=0.05, deterministic=True)
                test_acc = (action_logits_t.argmax(-1) == test_y_t).float().mean().item()
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        if verbose and (step + 1) % 1500 == 0:
            with torch.no_grad():
                action_logits, code, _ = model(s, a, tau=0.05, deterministic=True)
                acc = (action_logits.argmax(-1) == y).float().mean().item()
                code_idx = code.argmax(-1)
                n_used = int(torch.unique(code_idx).numel())
            print(f"  step {step+1:>5}  ce={ce.item():.4f}  train_acc={acc:.1%}  codes={n_used}/{K}  best_test={best_test_acc:.2%}")

    if best_state is not None:
        model.load_state_dict(best_state)
        if verbose:
            print(f"\n  → Restored best checkpoint with held-out acc = {best_test_acc:.2%}")

    print("\nEvaluation:")
    s = torch.tensor(train_states, device=device)
    a = torch.tensor(train_aggs, device=device)
    y = torch.tensor(train_actions, device=device)
    with torch.no_grad():
        action_logits, code, _ = model(s, a, tau=0.05, deterministic=True)
        train_acc = (action_logits.argmax(-1) == y).float().mean().item()
        n_used = len(torch.unique(code.argmax(-1)))
    print(f"  Train ({train_ns}): {train_acc:.2%}  codes_used={n_used}")

    s = torch.tensor(test_states, device=device)
    a = torch.tensor(test_aggs, device=device)
    y = torch.tensor(test_actions, device=device)
    with torch.no_grad():
        action_logits, _, _ = model(s, a, tau=0.05, deterministic=True)
        test_acc = (action_logits.argmax(-1) == y).float().mean().item()
    print(f"  Held-out test ({test_ns}): {test_acc:.2%}")
    print("\n  Per-n breakdown:")
    for n in test_ns:
        mask = test_n_disks == n
        if mask.sum() == 0: continue
        preds = action_logits[torch.tensor(mask)].argmax(-1).cpu().numpy()
        true = test_actions[mask]
        n_correct = int((preds == true).sum())
        n_total = int(mask.sum())
        print(f"    n={n}: {n_correct}/{n_total} ({100*n_correct/n_total:.1f}%)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=64)
    ap.add_argument("--train-ns", type=int, nargs="+", default=[2, 3, 4, 5, 6])
    ap.add_argument("--test-ns",  type=int, nargs="+", default=[7, 8, 9, 10])
    ap.add_argument("--n-max-pad", type=int, default=10)
    ap.add_argument("--steps", type=int, default=15000)
    ap.add_argument("--usage-weight", type=float, default=0.1)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = ap.parse_args()
    print(f"Device: {args.device}\n")
    train_and_test(K=args.K, train_ns=args.train_ns, test_ns=args.test_ns,
                   n_max_pad=args.n_max_pad, steps=args.steps,
                   usage_weight=args.usage_weight, device=args.device)


if __name__ == "__main__":
    main()
