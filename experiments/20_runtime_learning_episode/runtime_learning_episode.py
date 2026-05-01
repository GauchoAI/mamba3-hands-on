from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from operator_curriculum_lib import finish, train_binary_mlp


def main() -> None:
    start = time.time()
    # New task introduced at runtime: keep a parity bit while reading toggles.
    # State is current parity plus observed symbol class; action is next parity.
    xs, ys = [], []
    for parity in (0, 1):
        for symbol_is_toggle in (0, 1):
            xs.append([float(parity), float(symbol_is_toggle)])
            ys.append([float(parity ^ symbol_is_toggle)])
    x = torch.tensor(xs, dtype=torch.float32)
    y = torch.tensor(ys, dtype=torch.float32)
    model, stats = train_binary_mlp(x, y, hidden=6, epochs=160, lr=2e-2, seed=20)
    tests = []
    for seq in [[1, 0, 1, 1], [0, 0, 1], [1] * 11, [0, 1, 0, 1, 0]]:
        p = 0
        for symbol in seq:
            with torch.no_grad():
                p = int(torch.sigmoid(model(torch.tensor([[float(p), float(symbol)]], dtype=torch.float32))).item() >= 0.5)
        tests.append({"seq": seq, "pred": p, "expected": sum(seq) % 2, "ok": p == (sum(seq) % 2)})
    payload = {
        "script": __file__,
        "experiment": "runtime_learning_episode",
        "claim": "A new transition rule can be learned and verified at runtime under the one-minute rule.",
        "train": stats,
        "tests": tests,
        "pass_rate": round(sum(t["ok"] for t in tests) / len(tests), 4),
    }
    finish(start, payload)
    if payload["pass_rate"] < 1.0:
        raise SystemExit("runtime learning failed")


if __name__ == "__main__":
    main()

