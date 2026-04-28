"""gcd_step_function — Euclidean GCD by subtraction as a step function.

Algorithm: while a != b: if a > b: a -= b else: b -= a. When a == b,
it's the GCD.

Role-encoded state (4 small ints, value-invariant):
    a_gt_b : 1 if a > b else 0
    b_gt_a : 1 if b > a else 0
    a_zero : 1 if a == 0 else 0
    b_zero : 1 if b == 0 else 0

(at any time, exactly one of {a_gt_b, b_gt_a, a==b} is true. The
explicit zero flags handle the degenerate inputs.)

Action (3 classes):
    0: sub_b_from_a  (a -= b; valid when a > b)
    1: sub_a_from_b  (b -= a; valid when b > a)
    2: done          (a == b; emit the answer)

Training data: (state, action) pairs from running GCD on a curriculum
of (a, b) pairs in small range. Validate perfect extension on large
pairs that share no values with training.
"""
from typing import List, Tuple


# Action enumeration
ACTIONS = ["sub_b_from_a", "sub_a_from_b", "done"]
N_ACTIONS = len(ACTIONS)
ACTION_TO_IDX = {a: i for i, a in enumerate(ACTIONS)}


def state_for_pair(a: int, b: int) -> Tuple[int, int, int, int]:
    """4-tuple state for the current (a, b) pair."""
    return (
        1 if a > b else 0,
        1 if b > a else 0,
        1 if a == 0 else 0,
        1 if b == 0 else 0,
    )


def correct_action(a: int, b: int) -> int:
    """Reference action for the iterative subtraction algorithm."""
    if a == b:
        return ACTION_TO_IDX["done"]
    if a > b:
        return ACTION_TO_IDX["sub_b_from_a"]
    return ACTION_TO_IDX["sub_a_from_b"]


def gcd_step(a: int, b: int) -> Tuple[int, int]:
    """Apply one step of subtraction GCD. Returns new (a, b)."""
    if a == b:
        return a, b  # done
    if a > b:
        return a - b, b
    return a, b - a


def gcd_trajectory(a: int, b: int, max_steps: int = 1000) -> List[Tuple[Tuple[int,...], int]]:
    """Generate (state, action) pairs for solving GCD(a, b)."""
    pairs = []
    for _ in range(max_steps):
        s = state_for_pair(a, b)
        act = correct_action(a, b)
        pairs.append((s, act))
        if act == ACTION_TO_IDX["done"]:
            break
        a, b = gcd_step(a, b)
    return pairs


def harvest_pairs(curriculum_max: int) -> List[Tuple[Tuple[int,...], int]]:
    """For each (a, b) with 1 <= a, b <= curriculum_max, generate the
    full trajectory's (state, action) pairs.

    Many trajectories share states — collapse and dedupe? We keep
    duplicates because the relative frequency in training data
    matches frequency in real GCD problems.
    """
    pairs = []
    for a in range(1, curriculum_max + 1):
        for b in range(1, curriculum_max + 1):
            pairs.extend(gcd_trajectory(a, b))
    return pairs


if __name__ == "__main__":
    pairs = harvest_pairs(curriculum_max=10)
    print(f"Pairs from (a,b) in 1..10²: {len(pairs)}")
    distinct = set(s for s, _ in pairs)
    print(f"Distinct states: {len(distinct)}")
    print(f"States: {sorted(distinct)}")
    print(f"\nFirst 8 pairs:")
    for s, a in pairs[:8]:
        print(f"  state={s} -> {ACTIONS[a]}")
    # Verify: GCD(12, 8) = 4
    traj = gcd_trajectory(12, 8)
    print(f"\nGCD(12, 8) trajectory: {len(traj)} steps")
    a, b = 12, 8
    for s, act in traj:
        print(f"  ({a}, {b})  state={s}  action={ACTIONS[act]}")
        if act == ACTION_TO_IDX["done"]:
            print(f"  GCD = {a}")
            break
        a, b = gcd_step(a, b)
