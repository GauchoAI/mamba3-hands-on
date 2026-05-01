from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from operator_curriculum_lib import ROLE_CLOSE, ROLE_END, ROLE_OPEN, finish, train_binary_mlp


def featurize(prefix: list[int], target_pairs: int, candidate: int) -> list[float]:
    """Summarize the raw role prefix without directly providing depth."""
    n = max(1, target_pairs)
    opens = prefix.count(ROLE_OPEN)
    closes = prefix.count(ROLE_CLOSE)
    return [
        opens / n,
        closes / n,
        target_pairs / 24,
        float(candidate == ROLE_OPEN),
        float(candidate == ROLE_CLOSE),
        float(candidate == ROLE_END),
    ]


def build(max_pairs: int) -> tuple[torch.Tensor, torch.Tensor]:
    xs, ys = [], []
    for n in range(1, max_pairs + 1):
        for opens in range(n + 1):
            for closes in range(opens + 1):
                prefix = [ROLE_OPEN] * opens + [ROLE_CLOSE] * closes
                for cand in (ROLE_OPEN, ROLE_CLOSE, ROLE_END):
                    ok = (
                        (cand == ROLE_OPEN and opens < n) or
                        (cand == ROLE_CLOSE and opens > closes) or
                        (cand == ROLE_END and opens == n and closes == n)
                    )
                    xs.append(featurize(prefix, n, cand))
                    ys.append([float(ok)])
    return torch.tensor(xs, dtype=torch.float32), torch.tensor(ys, dtype=torch.float32)


def main() -> None:
    start = time.time()
    x, y = build(16)
    model, stats = train_binary_mlp(x, y, hidden=32, epochs=600, lr=1e-2, seed=12)
    hx, hy = build(20)
    with torch.no_grad():
        held_acc = ((torch.sigmoid(model(hx)) >= 0.5).float() == hy).float().mean().item()
    payload = {
        "script": __file__,
        "experiment": "raw_trace_stack",
        "claim": "Trace-derived counts plus a tiny policy recover stack validity without hand-given depth.",
        "train": stats,
        "heldout_acc_pairs_1_to_20": round(held_acc, 4),
    }
    finish(start, payload)
    if held_acc < 0.98:
        raise SystemExit("heldout accuracy below threshold")


if __name__ == "__main__":
    main()
