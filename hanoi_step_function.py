"""hanoi_step_function — the clean step function for Tower of Hanoi.

One forward pass = one move. No byte rendering.

State (5 small ints, n-invariant):
    n_parity     : n & 1
    move_parity  : move_index & 1
    top_A        : smallest disk on peg A (0 = empty, 1..n = disk id)
    top_B        : same for B
    top_C        : same for C

Action (6 classes, dense enumeration):
    0: A→B   1: A→C   2: B→A   3: B→C   4: C→A   5: C→B

The function f(state) → action is deterministic for the iterative
Hanoi algorithm and computable from these 5 inputs alone. If the
model learns it as a function, perfect extension to any n follows.

Training data = (state, action) pairs harvested from running Hanoi(n)
for each n in the curriculum. n=2..6 yields 119 unique pairs total.
"""
from typing import List, Tuple
from hanoi_tool import HanoiTool, hanoi_moves


# Action enumeration: (src_peg, dst_peg) pairs in fixed order.
ACTIONS = [
    (0, 1),  # 0: A→B
    (0, 2),  # 1: A→C
    (1, 0),  # 2: B→A
    (1, 2),  # 3: B→C
    (2, 0),  # 4: C→A
    (2, 1),  # 5: C→B
]
N_ACTIONS = len(ACTIONS)
ACTION_TO_IDX = {a: i for i, a in enumerate(ACTIONS)}


def state_for_step(tool: HanoiTool) -> Tuple[int, int, int, int, int]:
    """Produce the 5-tuple state for the current step of `tool`.

    Tops are encoded as ROLES, not disk IDs — independent of n:
        0 = empty
        1 = holds the globally smallest existing disk
        2 = holds the next-smallest "visible" disk (smallest of the
            other two pegs)
        3 = holds the largest visible disk (the third peg's top)

    With this encoding the state space is fixed and tiny (≤ 3^3 = 27
    distinct top configurations) regardless of n. Training on n=2..6
    sees every reachable configuration, so OOD n behaves identically.
    """
    raw_tops = [0, 0, 0]
    for disk_idx, peg in enumerate(tool.peg):
        disk_id = disk_idx + 1
        if raw_tops[peg] == 0 or disk_id < raw_tops[peg]:
            raw_tops[peg] = disk_id

    # Rank non-empty pegs by their top (smaller disk id = higher rank).
    non_empty = [(disk, peg) for peg, disk in enumerate(raw_tops) if disk > 0]
    non_empty.sort()  # ascending by disk id
    role = [0, 0, 0]
    for rank, (_, peg) in enumerate(non_empty):
        role[peg] = rank + 1  # 1 = smallest, 2 = next, 3 = largest

    return (
        tool.n & 1,
        tool.move_index & 1,
        role[0],
        role[1],
        role[2],
    )


def harvest_pairs(n_max_curr: List[int]) -> List[Tuple[Tuple[int, ...], int]]:
    """For each n in `n_max_curr`, run Hanoi(n) move-by-move and emit
    (state, action_idx) pairs at the start of each move.

    Returns list of (state_tuple, action_idx) — total = sum_{n} (2^n - 1).
    """
    pairs = []
    for n in n_max_curr:
        tool = HanoiTool(n)
        for k, src, dst in tool.moves:
            state = state_for_step(tool)
            action_idx = ACTION_TO_IDX[(src, dst)]
            pairs.append((state, action_idx))
            # Apply the move so next state is correct.
            tool.peg[k - 1] = dst
            tool.move_index += 1
    return pairs


def expected_action_sequence(n: int) -> List[int]:
    """Reference sequence of action indices for Hanoi(n)."""
    moves = hanoi_moves(n)
    return [ACTION_TO_IDX[(src, dst)] for k, src, dst in moves]


if __name__ == "__main__":
    pairs = harvest_pairs([2, 3, 4, 5, 6])
    print(f"Total (state, action) pairs for n=2..6: {len(pairs)}")
    # Should be 3+7+15+31+63 = 119
    assert len(pairs) == 119
    print(f"First 5 pairs:")
    for s, a in pairs[:5]:
        src, dst = ACTIONS[a]
        print(f"  state={s} -> action={a} ({'ABC'[src]}→{'ABC'[dst]})")

    # Sanity: number of distinct (state, action) tuples
    distinct_states = set(s for s, _ in pairs)
    print(f"Distinct states across n=2..6: {len(distinct_states)}")

    print("\n✓ Step-function dataset ready")
