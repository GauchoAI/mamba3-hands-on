from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from operator_curriculum_lib import finish


ROOT = Path(__file__).resolve().parents[2]


def load(rel: str) -> dict:
    return json.loads((ROOT / rel).read_text())


def main() -> None:
    start = time.time()
    evidence_paths = [
        "experiments/11_stack_operator_transfer/artifacts/stack_operator_results.json",
        "experiments/16_trace_to_operator_search/artifacts/result.json",
        "experiments/17_operator_registry/artifacts/operator_registry.json",
        "experiments/20_runtime_learning_episode/artifacts/result.json",
    ]
    evidence = [load(p) for p in evidence_paths if (ROOT / p).exists()]
    demo = {
        "language_adapter": "bounded role/task parser",
        "operator_induction": "runtime parity rule + trace search",
        "registry_entries": load("experiments/17_operator_registry/artifacts/operator_registry.json").get("n_entries", 0),
        "all_one_minute": all(e.get("one_minute_rule", True) for e in evidence),
    }
    payload = {
        "script": __file__,
        "experiment": "lab_organ_demo",
        "claim": "The Lab can run a small language-to-role/operator/registry loop without long training.",
        "evidence_count": len(evidence),
        "demo": demo,
    }
    finish(start, payload)
    if len(evidence) < 4 or not demo["all_one_minute"]:
        raise SystemExit("organ demo lacks evidence")


if __name__ == "__main__":
    main()

