"""Show winning config for each graduated teacher."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from state_db import StateDB

db = StateDB("three_pop/training.db")
teachers = db.get_teachers()

print(f"{'task':25s} {'d':>4s} {'L':>2s} {'dS':>3s} {'backend':>8s} {'device':>6s} {'acc':>5s}")
print("-" * 60)

for task in sorted(teachers.keys()):
    lin = db.get_lineage(task)
    winners = [e for e in lin if e["accuracy"] >= 0.95]
    if winners:
        w = winners[0]
        cfg = w.get("config", {})
        backend = cfg.get("scan_backend", "triton")
        device = cfg.get("device", "cuda")
        layers = cfg.get("n_kernel_layers", 3)
        d = cfg.get("d_model", 64)
        ds = cfg.get("d_state", 16)
        acc = w["accuracy"]
        print(f"{task:25s} {d:>4d} {layers:>2d} {ds:>3d} {backend:>8s} {device:>6s} {acc:>4.0%}")
    else:
        print(f"{task:25s} — no winning entry in lineage")

db.close()
