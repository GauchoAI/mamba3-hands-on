from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from operator_curriculum_lib import ROLE_CLOSE, ROLE_END, ROLE_OPEN, finish, seed_all, train_binary_mlp


SURFACES = {"paren": ("(", ")"), "bracket": ("[", "]"), "brace": ("{", "}"), "call": ("CALL(", ")")}


def features(depth: int, opens: int, closes: int, target: int, candidate: int, surface_id: int) -> list[float]:
    n = max(1, target)
    return [
        depth / n, opens / n, closes / n,
        float(depth == 0), float(opens == target), float(closes == target),
        float(candidate == ROLE_OPEN), float(candidate == ROLE_CLOSE), float(candidate == ROLE_END),
        surface_id / max(1, len(SURFACES) - 1),
    ]


def build(train_surfaces: list[str], max_pairs: int) -> tuple[torch.Tensor, torch.Tensor]:
    xs, ys = [], []
    surface_ids = {s: i for i, s in enumerate(SURFACES)}
    for s in train_surfaces:
        sid = surface_ids[s]
        for n in range(1, max_pairs + 1):
            for opens in range(n + 1):
                for closes in range(opens + 1):
                    depth = opens - closes
                    for cand in (ROLE_OPEN, ROLE_CLOSE, ROLE_END):
                        ok = (cand == ROLE_OPEN and opens < n) or (cand == ROLE_CLOSE and depth > 0) or (cand == ROLE_END and opens == n and closes == n)
                        xs.append(features(depth, opens, closes, n, cand, sid))
                        ys.append([1.0 if ok else 0.0])
    return torch.tensor(xs, dtype=torch.float32), torch.tensor(ys, dtype=torch.float32)


def main() -> None:
    start = time.time()
    seed_all(13)
    x, y = build(["paren", "bracket"], 6)
    model, stats = train_binary_mlp(x, y, hidden=18, epochs=180, lr=1e-2, seed=13)
    eval_rows = {}
    for surface in SURFACES:
        ex, ey = build([surface], 16)
        with torch.no_grad():
            acc = ((torch.sigmoid(model(ex)) >= 0.5).float() == ey).float().mean().item()
        eval_rows[surface] = round(acc, 4)
    payload = {
        "script": __file__,
        "experiment": "multi_surface_stack",
        "claim": "The same learned stack-validity policy transfers across remapped token surfaces.",
        "train_surfaces": ["paren", "bracket"],
        "heldout_surfaces": ["brace", "call"],
        "train": stats,
        "surface_acc": eval_rows,
    }
    finish(start, payload)
    if min(eval_rows.values()) < 0.98:
        raise SystemExit("surface transfer below threshold")


if __name__ == "__main__":
    main()

