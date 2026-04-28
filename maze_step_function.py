"""maze_step_function — greedy grid navigation step.

Open grid (or wall-checking can be layered in by orchestrator).

State (2 ints, encoded as 0..2 for -1, 0, +1):
    dx_sign: 0 = goal is to the west, 1 = same column, 2 = east
    dy_sign: 0 = goal is to the north, 1 = same row, 2 = south

Action (5):
    0: N   1: S   2: E   3: W   4: done

Greedy: prioritize y-axis movement, then x. Done iff both zero.

Total reachable states: 9 (3 × 3). Trivially generalizes to any
grid size since the function depends on signs of dx, dy — not
their magnitudes.
"""
from typing import List, Tuple

ACTIONS = ["N", "S", "E", "W", "done"]
A_N, A_S, A_E, A_W, A_DONE = 0, 1, 2, 3, 4
N_ACTIONS = len(ACTIONS)


def correct_action(dx_sign: int, dy_sign: int) -> int:
    """Greedy: vertical first, then horizontal. dx_sign/dy_sign in 0..2
    representing -1, 0, +1."""
    if dx_sign == 1 and dy_sign == 1:
        return A_DONE
    if dy_sign == 0:
        return A_N  # goal is north
    if dy_sign == 2:
        return A_S
    # dy_sign == 1: y is aligned, decide on x
    if dx_sign == 0:
        return A_W
    return A_E  # dx_sign == 2


def harvest_pairs() -> List[Tuple[Tuple[int, int], int]]:
    pairs = []
    for dx in range(3):
        for dy in range(3):
            pairs.append(((dx, dy), correct_action(dx, dy)))
    return pairs


def navigate(start: Tuple[int, int], goal: Tuple[int, int],
             model, device: str = "cpu",
             max_steps: int = 100000) -> Tuple[Tuple[int, int], List[int], int]:
    """Step the model from start toward goal on an open grid.
    Returns (final_position, action_history, n_steps)."""
    import torch
    pos = list(start)
    history = []
    for step in range(max_steps):
        dx = goal[0] - pos[0]
        dy = goal[1] - pos[1]
        dx_sign = 0 if dx < 0 else (1 if dx == 0 else 2)
        dy_sign = 0 if dy < 0 else (1 if dy == 0 else 2)
        s_t = torch.tensor([[dx_sign, dy_sign]], dtype=torch.long, device=device)
        with torch.no_grad():
            logits = model(s_t)
        act = int(logits[0].argmax().item())
        history.append(act)
        if act == A_DONE:
            return tuple(pos), history, step + 1
        if act == A_N:
            pos[1] -= 1
        elif act == A_S:
            pos[1] += 1
        elif act == A_E:
            pos[0] += 1
        elif act == A_W:
            pos[0] -= 1
    return tuple(pos), history, max_steps


if __name__ == "__main__":
    pairs = harvest_pairs()
    print(f"Maze step: {len(pairs)} (state, action) pairs")
    for s, a in pairs:
        dx, dy = s
        dx_str = ("←", " ", "→")[dx]
        dy_str = ("↑", " ", "↓")[dy]
        print(f"  dx={dx_str} dy={dy_str} -> {ACTIONS[a]}")
