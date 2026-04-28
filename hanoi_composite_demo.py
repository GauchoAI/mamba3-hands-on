"""hanoi_composite_demo — discovered Hanoi ensemble in composite tasks.

Demonstrates that the role-encoded Hanoi ensemble works as a drop-in
primitive for composite Lego tasks, not just standalone Hanoi.

Composites:
  GCDHANOI a b: solve Hanoi(a), solve Hanoi(b), gcd(2^a-1, 2^b-1)
  CHAIN n_list: solve Hanoi for each n in list, count total moves

Validates: the Lego-pattern says "discovery + composition + language".
We've done discovery (✓ 100% via mixed-K ensemble). This demo shows
composition (chain the discovered Lego with itself or with other Legos).
"""
import argparse, math, time
import numpy as np
import torch

from hanoi_solve import load_ensemble, solve_n


def task_hanoi(n: int, models, n_max_pad: int, device: str):
    """Direct Hanoi: solve, return move count."""
    return solve_n(n, models, n_max_pad, device)


def task_gcdhanoi(a: int, b: int, models, n_max_pad: int, device: str):
    """Compose Hanoi + GCD: gcd(2^a-1, 2^b-1).

    The discovered Hanoi ensemble produces the optimal move count
    (2^n - 1) for each input. We then compute gcd of those two counts
    using plain Python (not a discovered Lego, but easy to swap in).
    """
    res_a = solve_n(a, models, n_max_pad, device)
    res_b = solve_n(b, models, n_max_pad, device)
    if not (res_a["solved"] and res_b["solved"]):
        return {"ok": False, "reason": "Hanoi failed"}
    return {
        "ok": True,
        "a": a, "moves_a": res_a["steps"],
        "b": b, "moves_b": res_b["steps"],
        "gcd": math.gcd(res_a["steps"], res_b["steps"]),
    }


def task_chain(ns: list, models, n_max_pad: int, device: str):
    """Solve Hanoi for each n in list, sum total moves."""
    total = 0
    detail = []
    for n in ns:
        r = solve_n(n, models, n_max_pad, device)
        if not r["solved"]:
            return {"ok": False, "reason": f"failed at n={n}"}
        detail.append((n, r["steps"], r["optimal"]))
        total += r["steps"]
    return {"ok": True, "total_moves": total, "detail": detail}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/hanoi_role_ensemble.pt")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    print(f"Device: {args.device}")
    print(f"Loading discovered Hanoi ensemble…")
    models, n_max_pad = load_ensemble(args.ckpt, args.device)
    print(f"  {len(models)} models, K's: {[K for _, K in models]}, n_max_pad: {n_max_pad}")
    print()

    print("══ Composite tasks using the DISCOVERED ensemble ══\n")

    print("HANOI 5:")
    r = task_hanoi(5, models, n_max_pad, args.device)
    print(f"  result: {r}\n")

    print("HANOI 10:")
    r = task_hanoi(10, models, n_max_pad, args.device)
    print(f"  result: {r}\n")

    print("GCDHANOI 4 6  (gcd of 2^4-1, 2^6-1 = gcd(15, 63) = 3)")
    r = task_gcdhanoi(4, 6, models, n_max_pad, args.device)
    print(f"  result: {r}")
    print(f"  (math: gcd(2^4-1, 2^6-1) = {math.gcd(15, 63)})\n")

    print("GCDHANOI 6 9  (gcd of 2^6-1, 2^9-1 = gcd(63, 511) = 7)")
    r = task_gcdhanoi(6, 9, models, n_max_pad, args.device)
    print(f"  result: {r}")
    print(f"  (math: gcd(2^6-1, 2^9-1) = {math.gcd(63, 511)})\n")

    print("CHAIN [3, 5, 7] — solve each Hanoi, total move count")
    r = task_chain([3, 5, 7], models, n_max_pad, args.device)
    print(f"  result: {r}")
    optimal_total = (2**3 - 1) + (2**5 - 1) + (2**7 - 1)
    print(f"  (math: sum of optimal = {optimal_total})\n")


if __name__ == "__main__":
    main()
