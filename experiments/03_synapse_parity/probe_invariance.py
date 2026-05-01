"""probe_invariance — does prediction accuracy hold at every n, or decay?

If the role features are truly invariant, prediction accuracy on canonical
traces should be ~constant from n=15 (trained) up to n=23 (n_max_pad-1).
If it decays with n, there's a hidden index leak (or a training-distribution
gap that grows combinatorially with n).

Memory-safe: streams the canonical trace as a generator, evaluates in
fixed-size chunks (no full trace ever held in memory). For n=23 (8.4M
states) this stays well under 1 GB.
"""
import argparse, gc
import numpy as np, torch, torch.nn.functional as F
from discover_hanoi_roles_mixed import (
    role_features_K, legal_action_mask,
    ACTION_PAIRS, ACTION_TO_IDX, hanoi_moves,
)
from hanoi_solve import load_ensemble


def trace_chunks(n, n_max_pad, chunk_size):
    """Yield (states, actions) numpy chunks from the canonical trace of n."""
    pegs = [0] * n_max_pad
    for i in range(n, n_max_pad):
        pegs[i] = -1
    buf_states, buf_actions = [], []
    for src, dst in hanoi_moves(n):
        disk = next(i for i in range(n) if pegs[i] == src)
        buf_states.append(pegs.copy())
        buf_actions.append(ACTION_TO_IDX[(src, dst)])
        pegs[disk] = dst
        if len(buf_states) >= chunk_size:
            yield (np.array(buf_states, dtype=np.int64),
                   np.array(buf_actions, dtype=np.int64))
            buf_states.clear(); buf_actions.clear()
    if buf_states:
        yield (np.array(buf_states, dtype=np.int64),
               np.array(buf_actions, dtype=np.int64))


def eval_n(models, n, n_max_pad, device, chunk_size):
    """Stream + score canonical trace of n. Returns (n_total, n_correct)."""
    n_total = 0
    n_correct = 0
    for states, actions in trace_chunks(n, n_max_pad, chunk_size):
        legal = legal_action_mask(states, n_max_pad)
        legal_t = torch.tensor(legal, device=device)
        y_t = torch.tensor(actions, device=device)
        avg_probs = None
        with torch.no_grad():
            for model, K in models:
                feats = role_features_K(states, n_max_pad, K, K)
                feats_t = torch.tensor(feats, device=device)
                logits = model(feats_t).masked_fill(~legal_t, -1e9)
                probs = F.softmax(logits, dim=-1)
                avg_probs = probs if avg_probs is None else avg_probs + probs
                del feats, feats_t, logits, probs
            avg_probs /= len(models)
            pred = avg_probs.argmax(-1)
            n_correct += int((pred == y_t).sum().item())
        n_total += len(actions)
        del states, actions, legal, legal_t, y_t, avg_probs, pred
        gc.collect()
    return n_total, n_correct


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/hanoi_role_ensemble.pt")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n-list", type=int, nargs="+",
                    default=[15, 16, 17, 18, 19, 20])
    ap.add_argument("--chunk-size", type=int, default=65536,
                    help="states per chunk (memory bound)")
    args = ap.parse_args()

    models, n_max_pad = load_ensemble(args.ckpt, args.device)
    print(f"Loaded {len(models)} models, n_max_pad={n_max_pad}")
    print(f"Training was on n=2..15. Probing n={args.n_list}")
    print(f"Chunk size: {args.chunk_size}\n")

    print(f"{'n':>3} | {'states':>10} | {'correct':>10} | {'acc':>11} | {'verdict'}")
    print("-" * 60)
    for n in args.n_list:
        if n + 1 > n_max_pad:
            print(f"  n={n} skipped (exceeds n_max_pad={n_max_pad})")
            continue
        n_total, n_correct = eval_n(models, n, n_max_pad,
                                     args.device, args.chunk_size)
        acc = 100 * n_correct / n_total
        verdict = "✓" if n_correct == n_total else f"✗ {n_total - n_correct} wrong"
        print(f"{n:>3} | {n_total:>10} | {n_correct:>10} | {acc:>10.4f}% | {verdict}")


if __name__ == "__main__":
    main()
