"""conway_step_function — Conway's Game of Life as a cell-level step function.

Rules:
  - A live cell with 2 or 3 live neighbors stays alive.
  - A dead cell with exactly 3 live neighbors becomes alive.
  - All other cells die / stay dead.

State (2 ints): (alive, neighbor_count) where neighbor_count ∈ 0..8.
Action (2 classes): next state — 0 (dead) or 1 (alive).

18 distinct states (2 alive × 9 counts). Train sees them all.
Generalizes to any grid size by construction (the rule is per-cell).
"""
from typing import List, Tuple

ACTIONS = ["dead", "alive"]
N_ACTIONS = len(ACTIONS)


def correct_action(alive: int, n_alive: int) -> int:
    """The Conway transition rule."""
    if alive == 1 and n_alive in (2, 3):
        return 1
    if alive == 0 and n_alive == 3:
        return 1
    return 0


def harvest_pairs() -> List[Tuple[Tuple[int, int], int]]:
    """All 18 (state, action) pairs."""
    pairs = []
    for alive in (0, 1):
        for n_alive in range(9):
            state = (alive, n_alive)
            pairs.append((state, correct_action(alive, n_alive)))
    return pairs


def step_grid(grid: List[List[int]]) -> List[List[int]]:
    """Reference: apply Conway rules to a full grid (with wraparound)."""
    h = len(grid)
    w = len(grid[0])
    out = [[0] * w for _ in range(h)]
    for r in range(h):
        for c in range(w):
            n = sum(
                grid[(r + dr) % h][(c + dc) % w]
                for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                if not (dr == 0 and dc == 0)
            )
            out[r][c] = correct_action(grid[r][c], n)
    return out


if __name__ == "__main__":
    pairs = harvest_pairs()
    print(f"All distinct (alive, n_alive) -> next pairs ({len(pairs)}):")
    for s, a in pairs:
        alive, n = s
        print(f"  alive={alive}, neighbors={n} -> {ACTIONS[a]}")
