from __future__ import annotations

import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from operator_curriculum_lib import finish, seed_all, train_binary_mlp


def sort_with_policy(items: list, key, model) -> list:
    arr = list(items)
    dirty = True
    while dirty:
        dirty = False
        for i in range(len(arr) - 1):
            a_gt_b = float(key(arr[i]) > key(arr[i + 1]))
            with torch.no_grad():
                swap = torch.sigmoid(model(torch.tensor([[a_gt_b]], dtype=torch.float32))).item() >= 0.5
            if swap:
                arr[i], arr[i + 1] = arr[i + 1], arr[i]
                dirty = True
    return arr


def main() -> None:
    start = time.time()
    rng = seed_all(15)
    x = torch.tensor([[0.0], [1.0]], dtype=torch.float32)
    y = torch.tensor([[0.0], [1.0]], dtype=torch.float32)
    model, stats = train_binary_mlp(x, y, hidden=4, epochs=120, lr=2e-2, seed=15)
    tests = {
        "numbers": [rng.randint(0, 999) for _ in range(80)],
        "letters": list("sortingoperator"),
        "cards": [(rank, suit) for rank, suit in [(9, "H"), (2, "S"), (14, "D"), (7, "C"), (7, "H")]],
    }
    results = {}
    for name, values in tests.items():
        key = (lambda x: x[0]) if name == "cards" else (lambda x: x)
        out = sort_with_policy(values, key, model)
        results[name] = {"ok": out == sorted(values, key=key), "n": len(values)}
    payload = {
        "script": __file__,
        "experiment": "comparator_transfer",
        "claim": "A learned compare/swap decision transfers across surfaces once values are mapped to an ordering role.",
        "train": stats,
        "rollouts": results,
    }
    finish(start, payload)
    if not all(r["ok"] for r in results.values()):
        raise SystemExit("comparator transfer failed")


if __name__ == "__main__":
    main()

