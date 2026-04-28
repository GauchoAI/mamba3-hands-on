"""hanoi_parallel_solve — batched lockstep solver for the GRU.

Runs N independent (n, start_state) solver instances in parallel by
batching one GRU forward per tick. Smaller-n runs finish first and drop
out; larger n keep going. Tests robustness across many off-canonical
starts simultaneously.

Each instance:
  - random n in [n_min, n_max]
  - random reachable start state (uniform peg assignment)
  - target: all-on-peg-2
  - tracks success, step count, optimal count
"""
import argparse, time
import numpy as np
import torch

from discover_hanoi_invariant import HanoiInvariantGRU
from discover_hanoi_offtrace import optimal_move_from_state
from discover_hanoi_roles_mixed import (
    legal_action_mask, ACTION_PAIRS, N_ACTIONS,
)


def load_gru(path: str, device: str):
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = ck["config"]
    model = HanoiInvariantGRU(d_hidden=cfg["d_hidden"], n_layers=cfg["n_layers"]).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck["n_max_pad"]


def optimal_count_from_state(pegs, n, target=2):
    """Number of optimal moves to reach all-on-target from this state."""
    pegs = list(pegs)
    count = 0
    while True:
        m = optimal_move_from_state(pegs, n, target=target)
        if m is None:
            return count
        src, dst = m
        # apply: smallest disk on src moves to dst
        disk = next(i for i in range(n) if pegs[i] == src)
        pegs[disk] = dst
        count += 1


def init_run(n, n_max_pad, rng):
    """Random reachable start: uniform peg assignment over n disks."""
    pegs = np.full(n_max_pad, -1, dtype=np.int64)
    pegs[:n] = rng.integers(0, 3, size=n)
    optimal = optimal_count_from_state(pegs[:n].tolist(), n)
    return {"n": n, "pegs": pegs, "optimal": optimal,
            "steps": 0, "done": False, "solved": False, "reason": None}


def all_on_target(pegs, n, target=2):
    return bool(np.all(pegs[:n] == target))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/hanoi_invariant_gru.pt")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n-runs", type=int, default=50)
    ap.add_argument("--n-min", type=int, default=10)
    ap.add_argument("--n-max", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-steps-factor", type=int, default=4,
                    help="Cap each run at optimal * this factor")
    args = ap.parse_args()

    print(f"Device: {args.device}")
    model, n_max_pad = load_gru(args.ckpt, args.device)
    print(f"Loaded GRU, n_max_pad={n_max_pad}")

    rng = np.random.default_rng(args.seed)
    runs = []
    for _ in range(args.n_runs):
        n = int(rng.integers(args.n_min, args.n_max + 1))
        runs.append(init_run(n, n_max_pad, rng))
    print(f"\nLaunching {args.n_runs} parallel runs, n in [{args.n_min}, {args.n_max}]")
    print(f"  optimal steps total: {sum(r['optimal'] for r in runs)}")
    print(f"  longest single run optimal: {max(r['optimal'] for r in runs)}")

    t0 = time.time()
    last_report = t0
    while True:
        active_idx = [i for i, r in enumerate(runs) if not r["done"]]
        if not active_idx:
            break
        # Batch active states
        batch_states = np.stack([runs[i]["pegs"] for i in active_idx], axis=0)
        legal = legal_action_mask(batch_states, n_max_pad)
        legal_t = torch.tensor(legal, device=args.device)
        states_t = torch.tensor(batch_states, device=args.device)
        with torch.no_grad():
            logits = model(states_t).masked_fill(~legal_t, -1e9)
            actions = logits.argmax(-1).cpu().numpy()
        # Apply each action
        for j, i in enumerate(active_idx):
            r = runs[i]
            n = r["n"]
            if all_on_target(r["pegs"], n):
                r["done"] = True; r["solved"] = True; continue
            cap = r["optimal"] * args.max_steps_factor + 16
            if r["steps"] >= cap:
                r["done"] = True; r["solved"] = False; r["reason"] = "max_steps"; continue
            src, dst = ACTION_PAIRS[int(actions[j])]
            # apply: smallest disk on src
            pegs = r["pegs"]
            disk = next(d for d in range(n_max_pad) if pegs[d] == src)
            pegs[disk] = dst
            r["steps"] += 1
            if all_on_target(pegs, n):
                r["done"] = True; r["solved"] = True

        if time.time() - last_report > 5:
            n_done = sum(1 for r in runs if r["done"])
            n_solved = sum(1 for r in runs if r["solved"])
            tick = sum(r["steps"] for r in runs)
            print(f"  [{time.time()-t0:.0f}s] active={len(active_idx)} done={n_done} solved={n_solved} total_ticks={tick}")
            last_report = time.time()

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s.\n")
    n_solved = sum(1 for r in runs if r["solved"])
    n_optimal = sum(1 for r in runs if r["solved"] and r["steps"] == r["optimal"])
    print(f"Solved: {n_solved}/{len(runs)}")
    print(f"Optimal: {n_optimal}/{len(runs)}")
    print(f"\nPer-run detail (sorted by n):")
    print(f"{'n':>3} | {'optimal':>8} | {'achieved':>8} | {'verdict'}")
    print("-" * 50)
    for r in sorted(runs, key=lambda r: (r["n"], r["optimal"])):
        if r["solved"]:
            v = "✓ optimal" if r["steps"] == r["optimal"] else f"✓ {r['steps']/r['optimal']:.2f}x"
        else:
            v = f"✗ {r['reason'] or 'unknown'}"
        print(f"{r['n']:>3} | {r['optimal']:>8} | {r['steps']:>8} | {v}")


if __name__ == "__main__":
    main()
