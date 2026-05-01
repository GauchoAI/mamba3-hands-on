from __future__ import annotations

import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from operator_curriculum_lib import ROLE_CLOSE, ROLE_END, ROLE_OPEN, finish, valid_stack_sequence


def parse_request(text: str) -> dict:
    lower = text.lower()
    n_match = re.search(r"(\d+)", lower)
    n = int(n_match.group(1)) if n_match else 3
    surface = "bracket" if "bracket" in lower else "block" if "begin" in lower or "block" in lower else "paren"
    return {"target_pairs": n, "surface": surface}


def synthesize(task: dict) -> list[int]:
    return [ROLE_OPEN] * task["target_pairs"] + [ROLE_CLOSE] * task["target_pairs"] + [ROLE_END]


def main() -> None:
    start = time.time()
    prompts = [
        "Make a balanced parenthesis expression with 4 pairs.",
        "Generate 6 matching bracket pairs.",
        "Create a BEGIN END block with 5 nested pairs.",
    ]
    rows = []
    for prompt in prompts:
        task = parse_request(prompt)
        roles = synthesize(task)
        rows.append({"prompt": prompt, "task": task, "valid": valid_stack_sequence(roles, task["target_pairs"])})
    payload = {
        "script": __file__,
        "experiment": "language_to_role_trace",
        "claim": "Language is a thin adapter into role traces; exactness is owned by the role/operator layer.",
        "rows": rows,
        "pass_rate": round(sum(r["valid"] for r in rows) / len(rows), 4),
    }
    finish(start, payload)
    if payload["pass_rate"] < 1.0:
        raise SystemExit("language adapter failed")


if __name__ == "__main__":
    main()

