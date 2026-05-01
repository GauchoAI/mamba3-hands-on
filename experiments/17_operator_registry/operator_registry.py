from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from operator_curriculum_lib import finish


ROOT = Path(__file__).resolve().parents[2]
CHAPTERS = [
    ("stack_validity", "experiments/11_stack_operator_transfer/artifacts/stack_operator_results.json"),
    ("raw_trace_stack", "experiments/12_raw_trace_stack/artifacts/result.json"),
    ("multi_surface_stack", "experiments/13_multi_surface_stack/artifacts/result.json"),
    ("guided_decoding", "experiments/14_operator_guided_decoding/artifacts/result.json"),
    ("comparator", "experiments/15_comparator_transfer/artifacts/result.json"),
]


def main() -> None:
    start = time.time()
    entries = []
    for name, rel in CHAPTERS:
        path = ROOT / rel
        if not path.exists():
            continue
        raw = path.read_bytes()
        data = json.loads(raw)
        entries.append({
            "operator": name,
            "source": rel,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "one_minute_rule": data.get("one_minute_rule"),
            "elapsed_s": data.get("elapsed_s"),
        })
    registry = {
        "schema": "lab.operator_registry.v1",
        "entries": entries,
        "n_entries": len(entries),
    }
    out = Path(__file__).resolve().parent / "artifacts" / "operator_registry.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(registry, indent=2) + "\n")
    payload = {
        "script": __file__,
        "experiment": "operator_registry",
        "claim": "Operators become reusable only when their contracts and evidence are registered.",
        "registry": registry,
    }
    finish(start, payload)
    if len(entries) < 3:
        raise SystemExit("not enough operator evidence to register")


if __name__ == "__main__":
    main()

