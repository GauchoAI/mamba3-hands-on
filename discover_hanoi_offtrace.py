"""discover_hanoi_offtrace — augment training with off-trace reachable states.

Hypothesis: the n=18+ errors are all on novel role-fingerprints unseen
during n=2..15 canonical-trace training. If we add OFF-TRACE reachable
states from the same n range (random configurations, labeled with the
optimal next move), we expand the fingerprint set the MLP learns.

Key: any peg assignment of n disks is a reachable Hanoi state, and the
optimal next move toward "all on peg 2" is computable in O(n) via:
  - find largest disk not on target peg
  - if all smaller disks are on the auxiliary peg: move it to target
  - else: recurse on the subproblem (move smallest disks to aux first)

Memory-safe: off-trace samples are bounded (200k total across n=2..15).
Trains a single K=12 model to test the hypothesis quickly.
"""
import argparse, time, gc
import numpy as np
import torch
import torch.nn.functional as F

from discover_hanoi_roles_mixed import (
    HanoiRoleMLP, role_features_K, legal_action_mask,
    generate_traces_for_ns, ACTION_PAIRS, ACTION_TO_IDX, N_ACTIONS,
)


def optimal_move_from_state(pegs, n, target=2):
    """Optimal next (src, dst) to reach all-on-target. None if solved.

    pegs: list of length >= n; pegs[i] is current peg of disk i (0..2).
    """
    # Largest disk not yet on target
    k = -1
    for i in range(n - 1, -1, -1):
        if pegs[i] != target:
            k = i
            break
    if k == -1:
        return None
    src = pegs[k]
    aux = 3 - src - target
    # If all smaller disks are on aux, we can move disk k now
    if all(pegs[i] == aux for i in range(k)):
        return (src, target)
    # Otherwise, the next move is determined by the subproblem of moving
    # disks 0..k-1 to aux
    return optimal_move_from_state(pegs, k, target=aux)


def sample_offtrace_states(n_list, n_max_pad, samples_per_n, rng):
    """Random reachable states (uniform peg assignments) per n, labeled
    with the optimal next move. Returns (states, actions) numpy arrays."""
    all_states = []
    all_actions = []
    for n in n_list:
        for _ in range(samples_per_n):
            pegs = [int(p) for p in rng.integers(0, 3, size=n)]
            full = pegs + [-1] * (n_max_pad - n)
            move = optimal_move_from_state(pegs, n)
            if move is None:
                continue  # already solved, no action
            all_states.append(full)
            all_actions.append(ACTION_TO_IDX[move])
    return (np.array(all_states, dtype=np.int64),
            np.array(all_actions, dtype=np.int64))


def train_one_K(K, seed, train_states, train_actions, n_max_pad,
                steps, batch=512, lr=3e-3, d_hidden=128, device="cpu"):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    train_feats = role_features_K(train_states, n_max_pad, K, K)
    model = HanoiRoleMLP(K, K, d_hidden=d_hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr * 0.05)
    N = len(train_states)
    print(f"  K={K} seed={seed}: training on {N} states for {steps} steps")
    t0 = time.time()
    for step in range(steps):
        idx = rng.integers(0, N, size=batch)
        a = torch.tensor(train_feats[idx], device=device)
        y = torch.tensor(train_actions[idx], device=device)
        logits = model(a)
        loss = F.cross_entropy(logits, y)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if (step + 1) % 2000 == 0:
            print(f"    step {step+1}/{steps}  loss={loss.item():.4f}  "
                  f"elapsed={time.time()-t0:.0f}s")
    print(f"  K={K} done in {time.time()-t0:.0f}s")
    return model


def eval_model_on_n_canonical(model, K, n, n_max_pad, device, chunk_size=65536):
    """Stream canonical trace at n, return (n_total, n_correct) for this single model."""
    from probe_invariance import trace_chunks
    n_total, n_correct = 0, 0
    for states, actions in trace_chunks(n, n_max_pad, chunk_size):
        legal = legal_action_mask(states, n_max_pad)
        legal_t = torch.tensor(legal, device=device)
        y_t = torch.tensor(actions, device=device)
        feats = role_features_K(states, n_max_pad, K, K)
        feats_t = torch.tensor(feats, device=device)
        with torch.no_grad():
            logits = model(feats_t).masked_fill(~legal_t, -1e9)
            pred = logits.argmax(-1)
            n_correct += int((pred == y_t).sum().item())
        n_total += len(actions)
        del states, actions, legal, legal_t, y_t, feats, feats_t, logits, pred
        gc.collect()
    return n_total, n_correct


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-ns", type=int, nargs="+",
                    default=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
    ap.add_argument("--n-max-pad", type=int, default=24)
    ap.add_argument("--samples-per-n", type=int, default=15000)
    ap.add_argument("--K", type=int, default=12)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--probe-ns", type=int, nargs="+", default=[15, 17, 18, 19, 20])
    args = ap.parse_args()

    print(f"Device: {args.device}")
    print(f"Generating canonical traces for n={args.train_ns}...")
    train_pairs = generate_traces_for_ns(args.train_ns, args.n_max_pad)
    canonical_states = np.array([p[0] for p in train_pairs], dtype=np.int64)
    canonical_actions = np.array([p[2] for p in train_pairs], dtype=np.int64)
    print(f"  canonical states: {len(canonical_states)}")

    print(f"Sampling {args.samples_per_n} off-trace states per n in {args.train_ns}...")
    rng = np.random.default_rng(args.seed)
    offtrace_states, offtrace_actions = sample_offtrace_states(
        args.train_ns, args.n_max_pad, args.samples_per_n, rng)
    print(f"  off-trace states: {len(offtrace_states)}")

    train_states = np.concatenate([canonical_states, offtrace_states], axis=0)
    train_actions = np.concatenate([canonical_actions, offtrace_actions], axis=0)
    print(f"  TOTAL training states: {len(train_states)}")
    del canonical_states, canonical_actions, offtrace_states, offtrace_actions
    gc.collect()

    print(f"\nTraining single K={args.K} model with off-trace augmentation...")
    model = train_one_K(args.K, args.seed, train_states, train_actions,
                        args.n_max_pad, args.steps, device=args.device)

    del train_states, train_actions
    gc.collect()

    print("\n── Per-n canonical-trace prediction accuracy ──")
    print(f"{'n':>3} | {'states':>10} | {'correct':>10} | {'acc':>11} | {'verdict'}")
    print("-" * 58)
    for n in args.probe_ns:
        if n + 1 > args.n_max_pad:
            print(f"  n={n} skipped"); continue
        n_total, n_correct = eval_model_on_n_canonical(
            model, args.K, n, args.n_max_pad, args.device)
        acc = 100 * n_correct / n_total
        verdict = "✓" if n_correct == n_total else f"✗ {n_total - n_correct} wrong"
        print(f"{n:>3} | {n_total:>10} | {n_correct:>10} | {acc:>10.4f}% | {verdict}")


if __name__ == "__main__":
    main()
