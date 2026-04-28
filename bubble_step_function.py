"""bubble_step_function — bubble sort one-comparison decision.

Within a pass, the only per-step decision is "swap this adjacent
pair or not?" The orchestrator handles iteration, end-of-pass
detection, and termination.

State (1 bit, list-size-invariant): (a_gt_b,) — the comparison.
Action (2): swap, no_swap.

This is the simplest non-trivial step function — captures the
heart of comparison-based sorting in a single neural decision.
The same Lego works for any list length and any comparable type.
"""
from typing import List, Tuple

ACTIONS = ["swap", "no_swap"]
N_ACTIONS = len(ACTIONS)


def correct_action(a_gt_b: int) -> int:
    return 0 if a_gt_b else 1


def harvest_pairs() -> List[Tuple[Tuple[int], int]]:
    return [((0,), 1), ((1,), 0)]


def bubble_sort_with_step(arr: List, model, device: str = "cpu",
                          max_passes: int = 1000) -> Tuple[List, int]:
    """Bubble sort using the model as the comparison-decision step.
    Returns (sorted_arr, n_swaps)."""
    import torch
    arr = list(arr)
    n = len(arr)
    n_swaps = 0
    for p in range(max_passes):
        dirty = False
        for i in range(n - 1):
            a, b = arr[i], arr[i + 1]
            state = (1 if a > b else 0,)
            s_t = torch.tensor([list(state)], dtype=torch.long, device=device)
            with torch.no_grad():
                logits = model(s_t)
            act = int(logits[0].argmax().item())
            if act == 0:  # swap
                arr[i], arr[i + 1] = arr[i + 1], arr[i]
                dirty = True
                n_swaps += 1
        if not dirty:
            return arr, n_swaps
    return arr, n_swaps


if __name__ == "__main__":
    print(f"Bubble step pairs: {harvest_pairs()}")
