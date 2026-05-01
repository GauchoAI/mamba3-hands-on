from __future__ import annotations

import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from operator_curriculum_lib import ROLE_CLOSE, ROLE_END, ROLE_OPEN, finish, seed_all, train_binary_mlp, valid_stack_sequence


def feat(opens: int, closes: int, target: int, cand: int) -> list[float]:
    depth = opens - closes
    n = max(1, target)
    return [opens / n, closes / n, depth / n, float(depth == 0), float(opens == target), float(cand == ROLE_OPEN), float(cand == ROLE_CLOSE), float(cand == ROLE_END)]


def build(max_pairs: int) -> tuple[torch.Tensor, torch.Tensor]:
    xs, ys = [], []
    for n in range(1, max_pairs + 1):
        for opens in range(n + 1):
            for closes in range(opens + 1):
                for cand in (ROLE_OPEN, ROLE_CLOSE, ROLE_END):
                    ok = (cand == ROLE_OPEN and opens < n) or (cand == ROLE_CLOSE and opens > closes) or (cand == ROLE_END and opens == n and closes == n)
                    xs.append(feat(opens, closes, n, cand))
                    ys.append([float(ok)])
    return torch.tensor(xs, dtype=torch.float32), torch.tensor(ys, dtype=torch.float32)


def decode(rng: random.Random, target: int, model=None) -> list[int]:
    opens = closes = 0
    roles = []
    for _ in range(target * 4 + 4):
        proposal = rng.choices([ROLE_OPEN, ROLE_CLOSE, ROLE_END], [0.42, 0.42, 0.16], k=1)[0]
        cand_order = [proposal, ROLE_CLOSE, ROLE_OPEN, ROLE_END]
        if model is None:
            chosen = proposal
        else:
            chosen = ROLE_END
            for cand in cand_order:
                with torch.no_grad():
                    p = torch.sigmoid(model(torch.tensor([feat(opens, closes, target, cand)], dtype=torch.float32))).item()
                if p >= 0.5:
                    chosen = cand
                    break
        roles.append(chosen)
        if chosen == ROLE_END:
            break
        if chosen == ROLE_OPEN:
            opens += 1
        elif chosen == ROLE_CLOSE:
            closes += 1
    return roles


def pass_rate(model, trials: int, target: int, seed: int) -> float:
    rng = random.Random(seed)
    return sum(valid_stack_sequence(decode(rng, target, model), target) for _ in range(trials)) / trials


def main() -> None:
    start = time.time()
    seed_all(14)
    x, y = build(6)
    model, stats = train_binary_mlp(x, y, hidden=16, epochs=160, lr=1e-2, seed=14)
    payload = {
        "script": __file__,
        "experiment": "operator_guided_decoding",
        "claim": "A learned validity operator can steer a noisy token-role decoder during generation, not after generation.",
        "train": stats,
        "baseline_pass_rate": round(pass_rate(None, 100, 12, 1), 4),
        "guided_pass_rate_n12": round(pass_rate(model, 100, 12, 2), 4),
        "guided_pass_rate_n20": round(pass_rate(model, 100, 20, 3), 4),
    }
    finish(start, payload)
    if payload["guided_pass_rate_n20"] < 0.95:
        raise SystemExit("guided decoding below threshold")


if __name__ == "__main__":
    main()

