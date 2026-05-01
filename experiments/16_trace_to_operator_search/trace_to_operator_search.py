from __future__ import annotations

import itertools
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from operator_curriculum_lib import finish, train_binary_mlp


def labels(a: int, b: int) -> int:
    return int(a > b)


def feature_bank(a: int, b: int) -> dict[str, float]:
    return {
        "a_raw": a / 9,
        "b_raw": b / 9,
        "a_gt_b": float(a > b),
        "diff_sign": float(a - b > 0),
        "same_parity": float((a & 1) == (b & 1)),
    }


def build(names: tuple[str, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    xs, ys = [], []
    for a in range(10):
        for b in range(10):
            fb = feature_bank(a, b)
            xs.append([fb[n] for n in names])
            ys.append([float(labels(a, b))])
    return torch.tensor(xs, dtype=torch.float32), torch.tensor(ys, dtype=torch.float32)


def rollout_ok(model, names: tuple[str, ...]) -> bool:
    for a in range(50, -1, -1):
        for b in range(51):
            fb = feature_bank(a % 10, b % 10) | {"a_gt_b": float(a > b), "diff_sign": float(a - b > 0)}
            x = torch.tensor([[fb[n] for n in names]], dtype=torch.float32)
            with torch.no_grad():
                pred = torch.sigmoid(model(x)).item() >= 0.5
            if pred != (a > b):
                return False
    return True


def main() -> None:
    start = time.time()
    candidates = ["a_raw", "b_raw", "a_gt_b", "diff_sign", "same_parity"]
    tried = []
    winner = None
    for r in range(1, 4):
        for names in itertools.combinations(candidates, r):
            x, y = build(names)
            model, stats = train_binary_mlp(x, y, hidden=6, epochs=100, lr=2e-2, seed=16)
            ok = stats["train_acc"] == 1.0 and rollout_ok(model, names)
            tried.append({"features": names, "train_acc": stats["train_acc"], "rollout_ok": ok})
            if ok:
                winner = {"features": names, "stats": stats}
                break
        if winner:
            break
    payload = {
        "script": __file__,
        "experiment": "trace_to_operator_search",
        "claim": "A tiny search can select the smallest state encoding whose learned policy verifies by rollout.",
        "winner": winner,
        "tried": tried,
    }
    finish(start, payload)
    if not winner:
        raise SystemExit("no verified operator found")


if __name__ == "__main__":
    main()

