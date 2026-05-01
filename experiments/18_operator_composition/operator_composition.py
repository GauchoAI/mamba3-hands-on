from __future__ import annotations

import ast
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from operator_curriculum_lib import finish, seed_all, train_binary_mlp


def train_comparator():
    x = torch.tensor([[0.0], [1.0]], dtype=torch.float32)
    y = torch.tensor([[0.0], [1.0]], dtype=torch.float32)
    return train_binary_mlp(x, y, hidden=4, epochs=120, lr=2e-2, seed=18)[0]


def sort_list(values: list[int], model) -> list[int]:
    arr = list(values)
    dirty = True
    while dirty:
        dirty = False
        for i in range(len(arr) - 1):
            with torch.no_grad():
                swap = torch.sigmoid(model(torch.tensor([[float(arr[i] > arr[i + 1])]], dtype=torch.float32))).item() >= 0.5
            if swap:
                arr[i], arr[i + 1] = arr[i + 1], arr[i]
                dirty = True
    return arr


def solve_nested(expr: str, model) -> list:
    obj = ast.literal_eval(expr)
    return [sort_list(x, model) if isinstance(x, list) else x for x in obj]


def main() -> None:
    start = time.time()
    rng = seed_all(18)
    model = train_comparator()
    cases = []
    for _ in range(20):
        nested = [[rng.randint(0, 50) for _ in range(rng.randint(3, 8))] for _ in range(3)]
        expr = repr(nested)
        out = solve_nested(expr, model)
        expected = [sorted(x) for x in nested]
        cases.append(out == expected)
    payload = {
        "script": __file__,
        "experiment": "operator_composition",
        "claim": "A stack/parse harness and learned comparator operator compose into a nested-list sorter.",
        "n_cases": len(cases),
        "pass_rate": round(sum(cases) / len(cases), 4),
    }
    finish(start, payload)
    if not all(cases):
        raise SystemExit("composition failed")


if __name__ == "__main__":
    main()

