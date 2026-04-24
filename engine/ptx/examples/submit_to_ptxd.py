#!/usr/bin/env python3
"""
Minimal example: submit a batch of training jobs to `ptxd` via subprocess
stdin/stdout.  This is the integration pattern `three_populations.py` should
use to replace its current `spawn_worker` (one CPU specialist_trainer process
per task) pattern.

Usage:
    cd engine/ptx
    cargo build --release --bin ptxd
    python examples/submit_to_ptxd.py
"""

import json
import subprocess
import sys
import time
from pathlib import Path

PTXD = Path(__file__).resolve().parent.parent / "target/release/ptxd"

jobs = [
    {"id": f"parity_{i:02d}", "task": "parity",
     "d_model": 32, "n_layers": 1,
     "steps": 1000, "batch_size": 16}
    for i in range(4)
]

if not PTXD.exists():
    print(f"ptxd binary not found: {PTXD}")
    print("Run: cargo build --release --bin ptxd")
    sys.exit(1)

print(f"Submitting {len(jobs)} jobs to ptxd...")
t0 = time.time()

lines = "\n".join(json.dumps(j) for j in jobs) + "\n"
result = subprocess.run(
    [str(PTXD)],
    input=lines,
    capture_output=True, text=True,
)

if result.returncode != 0:
    print("ptxd exited with error:")
    print(result.stderr)
    sys.exit(1)

t = time.time() - t0
print(f"\nCompleted in {t:.2f}s total wall time.\n")

print("Per-job results:")
print(f"  {'id':<12} {'status':<14} {'loss':>8} {'best_acc':>9} {'ms/step':>9}")
for line in result.stdout.strip().split("\n"):
    r = json.loads(line)
    print(f"  {r['id']:<12} {r['status']:<14} "
          f"{r['final_loss']:>8.4f} {r['best_acc']:>9.2%} "
          f"{r['ms_per_step']:>8.2f}ms")
