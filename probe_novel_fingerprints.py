"""probe_novel_fingerprints — are high-n errors on novel role-fingerprints?

For each error at n=18..20, check whether its role-fingerprint (per K)
appeared in any n=2..15 canonical trace.

If errors are dominated by NOVEL fingerprints → the issue is a
training-distribution gap (combinatorially novel feature combos at high n).
If errors are dominated by SEEN fingerprints → the MLP is just locally
noisy on rare states; more ensemble seeds would help.

Memory-safe: streams traces in chunks. Builds training fingerprint set
ONCE (n=2..15 totals 65k states, tiny). Then streams n=18..20.
"""
import argparse, gc
import numpy as np, torch, torch.nn.functional as F
from discover_hanoi_roles_mixed import (
    role_features_K, legal_action_mask, hanoi_moves, ACTION_TO_IDX,
)
from hanoi_solve import load_ensemble


def trace_chunks(n, n_max_pad, chunk_size):
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


def build_training_fingerprints(train_ns, n_max_pad, K_values):
    """For each K, build set of fingerprint-tuples seen in n=2..15."""
    print(f"Building training fingerprint sets for K's={K_values}...")
    fp_sets = {K: set() for K in K_values}
    n_total = 0
    for n in train_ns:
        for states, _ in trace_chunks(n, n_max_pad, 65536):
            for K in K_values:
                feats = role_features_K(states, n_max_pad, K, K)
                for row in feats:
                    fp_sets[K].add(tuple(row.tolist()))
            n_total += len(states)
    for K, s in fp_sets.items():
        print(f"  K={K}: {len(s)} unique fingerprints from {n_total} states")
    return fp_sets


def analyze_n(models, n, n_max_pad, device, chunk_size, fp_sets, K_values):
    """Stream n's trace, find errors, classify each error by fingerprint
    novelty per K."""
    n_total = 0
    n_correct = 0
    # For each K, count: error has fingerprint in training? yes/no
    err_seen = {K: 0 for K in K_values}
    err_novel = {K: 0 for K in K_values}
    err_examples = []  # for first few errors

    for states, actions in trace_chunks(n, n_max_pad, chunk_size):
        legal = legal_action_mask(states, n_max_pad)
        legal_t = torch.tensor(legal, device=device)
        y_t = torch.tensor(actions, device=device)
        avg_probs = None
        # Also keep per-K features for membership test on errors
        feats_per_K = {}
        with torch.no_grad():
            for model, K in models:
                feats = role_features_K(states, n_max_pad, K, K)
                feats_per_K.setdefault(K, feats)
                feats_t = torch.tensor(feats, device=device)
                logits = model(feats_t).masked_fill(~legal_t, -1e9)
                probs = F.softmax(logits, dim=-1)
                avg_probs = probs if avg_probs is None else avg_probs + probs
            avg_probs /= len(models)
            pred = avg_probs.argmax(-1)
            wrong = (pred != y_t).cpu().numpy()
        n_total += len(actions)
        n_correct += int((~wrong).sum())
        # For each error, check fingerprint novelty per K
        err_idx = np.where(wrong)[0]
        for ei in err_idx:
            for K in K_values:
                fp = tuple(feats_per_K[K][ei].tolist())
                if fp in fp_sets[K]:
                    err_seen[K] += 1
                else:
                    err_novel[K] += 1
            if len(err_examples) < 3:
                err_examples.append({
                    "state": states[ei][:n].tolist(),
                    "true_action": int(actions[ei]),
                    "pred_action": int(pred[ei].item()),
                    "fp_seen_per_K": {K: tuple(feats_per_K[K][ei].tolist()) in fp_sets[K]
                                      for K in K_values},
                })
        del states, actions, legal, legal_t, y_t, avg_probs, pred, feats_per_K
        gc.collect()

    return n_total, n_correct, err_seen, err_novel, err_examples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/hanoi_role_ensemble.pt")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n-list", type=int, nargs="+", default=[18, 19, 20])
    ap.add_argument("--chunk-size", type=int, default=65536)
    args = ap.parse_args()

    models, n_max_pad = load_ensemble(args.ckpt, args.device)
    K_values = sorted(set(K for _, K in models))
    print(f"Loaded {len(models)} models, K's={K_values}, n_max_pad={n_max_pad}\n")

    train_ns = list(range(2, 16))
    fp_sets = build_training_fingerprints(train_ns, n_max_pad, K_values)
    print()

    for n in args.n_list:
        if n + 1 > n_max_pad:
            print(f"n={n} skipped"); continue
        print(f"── n={n} ──")
        n_total, n_correct, err_seen, err_novel, examples = analyze_n(
            models, n, n_max_pad, args.device, args.chunk_size, fp_sets, K_values)
        n_err = n_total - n_correct
        print(f"  total: {n_total}, correct: {n_correct}, errors: {n_err}")
        for K in K_values:
            seen = err_seen[K]
            novel = err_novel[K]
            print(f"  K={K}: errors with fp seen-in-train: {seen}, novel: {novel}")
        print(f"  first {len(examples)} errors:")
        for i, ex in enumerate(examples):
            print(f"    [{i}] state={ex['state']}")
            print(f"        true={ex['true_action']} pred={ex['pred_action']}  fp_seen_per_K={ex['fp_seen_per_K']}")
        print()


if __name__ == "__main__":
    main()
