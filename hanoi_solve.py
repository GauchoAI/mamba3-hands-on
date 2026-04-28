"""hanoi_solve — load discovered Hanoi ensemble and solve any-n Hanoi.

Demonstrates "perfect extension at scale": train on n=2..15, then drop
this script in front of any n and let it run. Reports:
  - did it solve it?
  - did it use the optimal 2^n - 1 moves?
  - how long did it take?

If the ensemble is truly invariant, n=20 (1M moves), n=25 (33M moves)
should all run optimally despite never seeing those n's in training.
"""
import argparse, time
import numpy as np
import torch
import torch.nn.functional as F

from discover_hanoi_roles_mixed import (
    HanoiRoleMLP, role_features_K, legal_action_mask,
    ACTION_PAIRS, N_ACTIONS, ABSENT,
)


def load_ensemble(path: str, device: str):
    ck = torch.load(path, map_location=device, weights_only=False)
    models = []
    for state_dict, K in ck["models"]:
        m = HanoiRoleMLP(K, K).to(device)
        m.load_state_dict(state_dict)
        m.eval()
        models.append((m, K))
    return models, ck["n_max_pad"]


def solve_n(n: int, models, n_max_pad: int, device: str,
            verbose: bool = False, max_steps_factor: int = 4):
    """Run the ensemble on a fresh n-disk Hanoi until solved or stuck."""
    pegs = np.array([0] * n_max_pad, dtype=np.int64)
    for i in range(n, n_max_pad):
        pegs[i] = -1
    optimal = (1 << n) - 1
    max_steps = optimal * max_steps_factor + 16
    step_count = 0
    visited = set()

    while step_count < max_steps:
        # Goal check
        if all(pegs[i] == 2 for i in range(n)):
            return {"solved": True, "steps": step_count, "optimal": optimal}

        # Stuck detection: revisited state means we're cycling
        st_tuple = tuple(pegs[:n])
        if st_tuple in visited:
            return {"solved": False, "steps": step_count, "optimal": optimal,
                    "reason": "cycle"}
        visited.add(st_tuple)

        # Compute legal mask for current state
        state_2d = pegs[None, :]   # (1, n_max_pad)
        legal = legal_action_mask(state_2d, n_max_pad)
        if not legal.any():
            return {"solved": False, "steps": step_count, "optimal": optimal,
                    "reason": "no legal moves"}
        legal_t = torch.tensor(legal, device=device)

        # Ensemble vote: each K computes its own features, then softmax-avg
        avg_probs = None
        with torch.no_grad():
            for model, K in models:
                feats = role_features_K(state_2d, n_max_pad, K, K)
                feats_t = torch.tensor(feats, device=device)
                logits = model(feats_t).masked_fill(~legal_t, -1e9)
                probs = F.softmax(logits, dim=-1)
                avg_probs = probs if avg_probs is None else avg_probs + probs
            avg_probs /= len(models)
            action = int(avg_probs.argmax(-1).item())

        src, dst = ACTION_PAIRS[action]
        # Apply move: smallest disk on src goes to dst
        disk = next(i for i in range(n_max_pad) if pegs[i] == src)
        pegs[disk] = dst
        step_count += 1

        if verbose and step_count % 1000 == 0:
            print(f"  step {step_count} / {optimal}  ({100*step_count/optimal:.1f}%)")

    return {"solved": False, "steps": step_count, "optimal": optimal,
            "reason": "max_steps"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/hanoi_role_ensemble.pt")
    ap.add_argument("--n-list", type=int, nargs="+", default=[2, 5, 10, 15, 17, 20, 22])
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = ap.parse_args()
    print(f"Device: {args.device}")
    print(f"Loading ensemble from {args.ckpt}…")
    models, n_max_pad = load_ensemble(args.ckpt, args.device)
    print(f"  {len(models)} models loaded, K values: {[K for _, K in models]}")
    print(f"  n_max_pad: {n_max_pad}\n")

    print(f"{'n':>3} | {'optimal':>10} | {'achieved':>10} | {'ratio':>6} | "
          f"{'time':>8} | {'verdict':<25}")
    print("-" * 80)
    for n in args.n_list:
        if n + 1 > n_max_pad:
            print(f"  n={n} skipped (exceeds n_max_pad={n_max_pad})")
            continue
        t0 = time.time()
        result = solve_n(n, models, n_max_pad, args.device)
        dt = time.time() - t0
        if result["solved"]:
            ratio = result["steps"] / result["optimal"]
            verdict = "✓ optimal" if result["steps"] == result["optimal"] else f"✓ {ratio:.2f}x"
        else:
            ratio = float("inf")
            verdict = f"✗ {result['reason']}"
        print(f"{n:>3} | {result['optimal']:>10} | {result['steps']:>10} | "
              f"{ratio:>6.2f} | {dt:>7.1f}s | {verdict}")


if __name__ == "__main__":
    main()
