"""discover_hanoi_invariant — order-invariant GRU over disk-peg sequence.

Replaces the fixed-feature MLP with a GRU that processes the disk-peg
sequence from largest to smallest. Architecture has zero dependence on n:
weights are shared per position. By construction, the function it learns
is defined for any sequence length, not just the lengths seen in training.

Hypothesis: this fixes the "fingerprint-set grows with n" problem.
The MLP only knew responses to fingerprints it saw during training.
The GRU learns a recurrence; whatever it learns at depth-k applies at
any depth.

Input encoding: peg sequence reversed (position 0 = largest disk's peg,
position n-1 = smallest disk's peg). ABSENT disks (positions >= n) become
ABSENT-token (=3). The GRU consumes the full padded sequence; absent
disks come first (since reversed), then real disks largest→smallest.

Training on n=2..15 canonical traces should suffice for any n at inference.
"""
import argparse, time, gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from discover_hanoi_roles_mixed import (
    legal_action_mask, generate_traces_for_ns,
    ACTION_PAIRS, ACTION_TO_IDX, N_ACTIONS, ABSENT,
)


class HanoiInvariantGRU(nn.Module):
    """Process disk-peg sequence largest→smallest with a small GRU."""

    def __init__(self, d_emb=16, d_hidden=64, n_layers=2):
        super().__init__()
        self.peg_emb = nn.Embedding(4, d_emb)  # 0, 1, 2, ABSENT
        self.gru = nn.GRU(d_emb, d_hidden, num_layers=n_layers,
                          batch_first=True, dropout=0.0)
        self.head = nn.Sequential(
            nn.Linear(d_hidden, d_hidden), nn.ReLU(),
            nn.Linear(d_hidden, N_ACTIONS),
        )

    def forward(self, pegs):
        """pegs: (B, max_len) int64, -1 = absent. Returns logits (B, N_ACTIONS)."""
        # Map -1 → ABSENT (=3), keep 0/1/2 as is
        pegs_clean = torch.where(pegs == -1, torch.full_like(pegs, ABSENT), pegs)
        # Reverse so position 0 of GRU sees largest (or absent) first,
        # position n-1 sees smallest disk last
        pegs_rev = pegs_clean.flip(-1)
        x = self.peg_emb(pegs_rev)  # (B, max_len, d_emb)
        h, _ = self.gru(x)  # (B, max_len, d_hidden)
        return self.head(h[:, -1])  # last position = smallest disk


def train(model, train_states, train_actions, n_max_pad,
          steps, batch=512, lr=3e-3, device="cpu",
          test_states=None, test_actions=None, test_legal=None):
    rng = np.random.default_rng(0)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr * 0.05)
    N = len(train_states)
    print(f"Training on {N} states, {steps} steps, batch={batch}")
    t0 = time.time()
    best_acc = 0.0
    best_state = None
    test_a_t = torch.tensor(test_states, device=device) if test_states is not None else None
    test_y_t = torch.tensor(test_actions, device=device) if test_actions is not None else None
    test_legal_t = torch.tensor(test_legal, device=device) if test_legal is not None else None
    for step in range(steps):
        idx = rng.integers(0, N, size=batch)
        a = torch.tensor(train_states[idx], device=device)
        y = torch.tensor(train_actions[idx], device=device)
        logits = model(a)
        loss = F.cross_entropy(logits, y)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if (step + 1) % 1000 == 0:
            if test_a_t is not None:
                with torch.no_grad():
                    eval_logits = model(test_a_t).masked_fill(~test_legal_t, -1e9)
                    acc = (eval_logits.argmax(-1) == test_y_t).float().mean().item()
                if acc > best_acc:
                    best_acc = acc
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                print(f"  step {step+1}/{steps}  loss={loss.item():.4f}  "
                      f"test_acc={acc:.4%}  elapsed={time.time()-t0:.0f}s")
            else:
                print(f"  step {step+1}/{steps}  loss={loss.item():.4f}  "
                      f"elapsed={time.time()-t0:.0f}s")
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def eval_canonical_n(model, n, n_max_pad, device, chunk_size=65536):
    """Stream canonical trace, return (n_total, n_correct)."""
    from probe_invariance import trace_chunks
    n_total, n_correct = 0, 0
    for states, actions in trace_chunks(n, n_max_pad, chunk_size):
        legal = legal_action_mask(states, n_max_pad)
        legal_t = torch.tensor(legal, device=device)
        y_t = torch.tensor(actions, device=device)
        states_t = torch.tensor(states, device=device)
        with torch.no_grad():
            logits = model(states_t).masked_fill(~legal_t, -1e9)
            pred = logits.argmax(-1)
            n_correct += int((pred == y_t).sum().item())
        n_total += len(actions)
        del states, actions, legal, legal_t, y_t, states_t, logits, pred
        gc.collect()
    return n_total, n_correct


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-ns", type=int, nargs="+",
                    default=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
    ap.add_argument("--test-ns", type=int, nargs="+", default=[16, 17])
    ap.add_argument("--n-max-pad", type=int, default=24)
    ap.add_argument("--steps", type=int, default=15000)
    ap.add_argument("--d-hidden", type=int, default=64)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--probe-ns", type=int, nargs="+",
                    default=[15, 16, 17, 18, 19, 20, 21, 22, 23])
    ap.add_argument("--save-to", type=str, default="checkpoints/hanoi_invariant_gru.pt")
    args = ap.parse_args()

    print(f"Device: {args.device}")
    print(f"Building canonical traces for n={args.train_ns}...")
    train_pairs = generate_traces_for_ns(args.train_ns, args.n_max_pad)
    train_states = np.array([p[0] for p in train_pairs], dtype=np.int64)
    train_actions = np.array([p[2] for p in train_pairs], dtype=np.int64)
    print(f"  train states: {len(train_states)}")

    test_pairs = generate_traces_for_ns(args.test_ns, args.n_max_pad)
    test_states = np.array([p[0] for p in test_pairs], dtype=np.int64)
    test_actions = np.array([p[2] for p in test_pairs], dtype=np.int64)
    test_legal = legal_action_mask(test_states, args.n_max_pad)
    print(f"  test states: {len(test_states)} (n={args.test_ns})")

    model = HanoiInvariantGRU(d_hidden=args.d_hidden, n_layers=args.n_layers).to(args.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: HanoiInvariantGRU  d_hidden={args.d_hidden}  layers={args.n_layers}  params={n_params}")

    model = train(model, train_states, train_actions, args.n_max_pad,
                  args.steps, device=args.device,
                  test_states=test_states, test_actions=test_actions,
                  test_legal=test_legal)

    del train_states, train_actions, test_states, test_actions, test_legal
    gc.collect()

    print("\n── Per-n canonical-trace prediction accuracy ──")
    print(f"{'n':>3} | {'states':>10} | {'correct':>10} | {'acc':>11} | {'verdict'}")
    print("-" * 58)
    for n in args.probe_ns:
        if n + 1 > args.n_max_pad:
            print(f"  n={n} skipped"); continue
        n_total, n_correct = eval_canonical_n(model, n, args.n_max_pad, args.device)
        acc = 100 * n_correct / n_total
        verdict = "✓" if n_correct == n_total else f"✗ {n_total - n_correct} wrong"
        print(f"{n:>3} | {n_total:>10} | {n_correct:>10} | {acc:>10.4f}% | {verdict}")

    if args.save_to:
        from pathlib import Path
        Path(args.save_to).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": model.state_dict(),
            "config": {"d_hidden": args.d_hidden, "n_layers": args.n_layers},
            "n_max_pad": args.n_max_pad,
            "train_ns": args.train_ns,
        }, args.save_to)
        print(f"\nSaved → {args.save_to}")


if __name__ == "__main__":
    main()
