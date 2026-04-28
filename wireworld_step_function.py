"""wireworld_step_function — WireWorld CA as a per-cell step function.

States: 0=empty, 1=conductor, 2=electron_head, 3=electron_tail.

Transition rules (deterministic):
  empty           → empty
  electron_head   → electron_tail
  electron_tail   → conductor
  conductor       → electron_head iff 1 or 2 of its 8 neighbors are heads,
                    else stays conductor.

Per-cell input feature: (cell_state, n_head_neighbors), n_head_neighbors ∈ 0..8.
4 states × 9 counts = 36 distinct (state, n) pairs. Train sees them all.
The rule per cell has 4 branches — exactly the kind of thing that takes
NumPy several boolean masks but the MLP collapses into one forward.
"""
from typing import List, Tuple

EMPTY, CONDUCTOR, HEAD, TAIL = 0, 1, 2, 3
ACTIONS = ["empty", "conductor", "head", "tail"]
N_ACTIONS = len(ACTIONS)


def correct_action(cell: int, n_heads: int) -> int:
    if cell == EMPTY:
        return EMPTY
    if cell == HEAD:
        return TAIL
    if cell == TAIL:
        return CONDUCTOR
    # cell == CONDUCTOR
    return HEAD if n_heads in (1, 2) else CONDUCTOR


def harvest_pairs() -> List[Tuple[Tuple[int, int], int]]:
    pairs = []
    for cell in range(4):
        for n in range(9):
            pairs.append(((cell, n), correct_action(cell, n)))
    return pairs


def step_grid(grid: List[List[int]]) -> List[List[int]]:
    h = len(grid); w = len(grid[0])
    out = [[0] * w for _ in range(h)]
    for r in range(h):
        for c in range(w):
            n = sum(
                1 for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                if not (dr == 0 and dc == 0)
                and grid[(r + dr) % h][(c + dc) % w] == HEAD
            )
            out[r][c] = correct_action(grid[r][c], n)
    return out


if __name__ == "__main__":
    pairs = harvest_pairs()
    print(f"All {len(pairs)} (cell, n_heads) → next pairs:")
    for s, a in pairs:
        cell, n = s
        print(f"  cell={ACTIONS[cell]:<10} n_heads={n} → {ACTIONS[a]}")
