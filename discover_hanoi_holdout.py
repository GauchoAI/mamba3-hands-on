"""discover_hanoi_holdout — does the discovered encoding generalize?

Train on n=2..6 traces only. Test on n=7..10 traces (the model has never
seen them). The model gets the padded peg-state but NO n_disks input —
it has to infer n from the count of non-(-1) entries on its own.

If the discovered Lego achieves >0% on held-out n, it means the
encoding learned a structural pattern that transfers — perfect-extension
in the Lego sense, learned from data alone with no role-encoding hint.
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
    if n == 0:
        return
    yield from hanoi_moves(n - 1, src, aux, dst)
    yield (src, dst)
    yield from hanoi_moves(n - 1, aux, dst, src)


def generate_traces_for_ns(n_list, n_max_pad: int):
    pairs = []
    for n in n_list:
        pegs = [0] * n_max_pad
        for i in range(n):
            pegs[i] = 0
        for i in range(n, n_max_pad):
            pegs[i] = -1
        for src, dst in hanoi_moves(n):
            disk_to_move = None
            for d in range(n):
                if pegs[d] == src:
                    disk_to_move = d
                    break
            assert disk_to_move is not None
            pairs.append((pegs.copy(), n, ACTION_TO_IDX[(src, dst)]))
            pegs[disk_to_move] = dst
    return pairs


class DiscoveryHanoiNoN(nn.Module):
    """Same architecture but WITHOUT n_disks input — encoder must infer
    n from padded state alone."""
    def __init__(self, n_max: int, K: int = 64, d_emb: int = 8, d_hidden: int = 64):
        super().__init__()
        self.n_max = n_max
        self.K = K
        self.peg_emb = nn.Embedding(4, d_emb)         # -1, 0, 1, 2
        d_in = n_max * d_emb
        self.enc = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, K),
        )
        self.dec = nn.Linear(K, N_ACTIONS)

    def forward(self, state, tau=1.0):
        peg_e = self.peg_emb(state + 1)
        x = peg_e.flatten(1)
        logits = self.enc(x)
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
    test_states   = np.array([p[0] for p in test_pairs],  dtype=np.int64)
    test_n_disks  = np.array([p[1] for p in test_pairs],  dtype=np.int64)
    test_actions  = np.array([p[2] for p in test_pairs],  dtype=np.int64)
    N = len(train_pairs)

    model = DiscoveryHanoiNoN(n_max=n_max_pad, K=K).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"K={K}  params={n_params}  (usage_weight={usage_weight})")
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    log_K = float(np.log(K))
    for step in range(steps):
        idx = rng.integers(0, N, size=batch)
        s = torch.tensor(train_states[idx], device=device)
        y = torch.tensor(train_actions[idx], device=device)
        tau = max(0.3, 1.0 * (1.0 - step / steps))
        action_logits, code, _ = model(s, tau=tau)
        ce = F.cross_entropy(action_logits, y)
        code_probs = code.mean(dim=0)
        usage_score = -(code_probs * torch.log(code_probs + 1e-10)).sum() / log_K
        loss = ce - usage_weight * usage_score
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()

        if verbose and (step + 1) % 1000 == 0:
            with torch.no_grad():
                action_logits, code, _ = model(s, tau=0.05)
                acc = (action_logits.argmax(-1) == y).float().mean().item()
                code_idx = code.argmax(-1)
                n_used = int(torch.unique(code_idx).numel())
            print(f"  step {step+1:>5}  ce={ce.item():.4f}  train_acc={acc:.1%}  codes={n_used}/{K}")

    # Evaluate train and held-out test
    print("\nEvaluation:")
    s = torch.tensor(train_states, device=device)
    y = torch.tensor(train_actions, device=device)
    with torch.no_grad():
        action_logits, code, _ = model(s, tau=0.05)
        train_acc = (action_logits.argmax(-1) == y).float().mean().item()
        n_used = len(torch.unique(code.argmax(-1)))
    print(f"  Train ({train_ns}): {train_acc:.2%}  codes_used={n_used}")

    s = torch.tensor(test_states, device=device)
    y = torch.tensor(test_actions, device=device)
    with torch.no_grad():
        action_logits, _, _ = model(s, tau=0.05)
        test_acc = (action_logits.argmax(-1) == y).float().mean().item()
    print(f"  Held-out test ({test_ns}): {test_acc:.2%}")
    print(f"\n  Per-n breakdown:")
    for n in test_ns:
        mask = test_n_disks == n
        if mask.sum() == 0: continue
        preds = action_logits[torch.tensor(mask)].argmax(-1).cpu().numpy()
        true = test_actions[mask]
        n_correct = int((preds == true).sum())
        n_total = int(mask.sum())
        print(f"    n={n}: {n_correct}/{n_total} ({100*n_correct/n_total:.1f}%)  trace_len={n_total}")
    return train_acc, test_acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=64)
    ap.add_argument("--train-ns", type=int, nargs="+", default=[2, 3, 4, 5, 6])
    ap.add_argument("--test-ns",  type=int, nargs="+", default=[7, 8, 9, 10])
    ap.add_argument("--n-max-pad", type=int, default=10)
    ap.add_argument("--steps", type=int, default=10_000)
    ap.add_argument("--usage-weight", type=float, default=0.1)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = ap.parse_args()
    print(f"Device: {args.device}")
    print()
    train_and_test(K=args.K, train_ns=args.train_ns, test_ns=args.test_ns,
                   n_max_pad=args.n_max_pad, steps=args.steps,
                   usage_weight=args.usage_weight, device=args.device)


if __name__ == "__main__":
    main()
