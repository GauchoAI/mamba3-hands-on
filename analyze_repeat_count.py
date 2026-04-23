"""Analyze repeat_count mutations — which configs scored best?"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from state_db import StateDB
from collections import defaultdict

db = StateDB("three_pop/training.db")
lin = db.get_lineage("repeat_count")

print(f"repeat_count: {len(lin)} lineage entries")
print()

# All entries sorted by accuracy
entries = []
for e in lin:
    cfg = e.get("config", {})
    entries.append({
        "acc": e["accuracy"],
        "round": e["round"],
        "role": e.get("role", "?"),
        "d": cfg.get("d_model", 64),
        "L": cfg.get("n_kernel_layers", 3),
        "dS": cfg.get("d_state", 16),
        "lr": cfg.get("lr", 1e-3),
        "wd": cfg.get("weight_decay", 0.1),
        "opt": cfg.get("optimizer", "adamw"),
        "loss": cfg.get("loss_fn", "ce"),
        "backend": cfg.get("scan_backend", "triton"),
        "device": cfg.get("device", "cuda"),
        "hd": cfg.get("headdim", 16),
    })

# Top configs by accuracy
entries.sort(key=lambda x: -x["acc"])
print("=== TOP 15 by accuracy ===")
print(f"{'acc':>5s} {'r':>4s} {'role':>10s} {'d':>4s} {'L':>2s} {'dS':>3s} {'hd':>3s} {'lr':>8s} {'wd':>4s} {'opt':>6s} {'loss':>12s} {'backend':>7s} {'device':>5s}")
print("-" * 90)
for e in entries[:15]:
    print(f"{e['acc']:>4.0%} r{e['round']:>3d} {e['role']:>10s} {e['d']:>4d} {e['L']:>2d} {e['dS']:>3d} {e['hd']:>3d} {e['lr']:>8.1e} {e['wd']:>4.2f} {e['opt']:>6s} {e['loss']:>12s} {e['backend']:>7s} {e['device']:>5s}")

# What d_model values have been tried?
print()
print("=== Accuracy by d_model ===")
by_d = defaultdict(list)
for e in entries:
    by_d[e["d"]].append(e["acc"])
for d in sorted(by_d.keys()):
    accs = by_d[d]
    print(f"  d={d:>3d}: max={max(accs):.0%} avg={sum(accs)/len(accs):.0%} n={len(accs)}")

# By layers
print()
print("=== Accuracy by n_layers ===")
by_L = defaultdict(list)
for e in entries:
    by_L[e["L"]].append(e["acc"])
for L in sorted(by_L.keys()):
    accs = by_L[L]
    print(f"  L={L}: max={max(accs):.0%} avg={sum(accs)/len(accs):.0%} n={len(accs)}")

# By d_state
print()
print("=== Accuracy by d_state ===")
by_dS = defaultdict(list)
for e in entries:
    by_dS[e["dS"]].append(e["acc"])
for dS in sorted(by_dS.keys()):
    accs = by_dS[dS]
    print(f"  dS={dS:>2d}: max={max(accs):.0%} avg={sum(accs)/len(accs):.0%} n={len(accs)}")

# By backend
print()
print("=== Accuracy by backend ===")
by_be = defaultdict(list)
for e in entries:
    by_be[e["backend"]].append(e["acc"])
for be in sorted(by_be.keys()):
    accs = by_be[be]
    print(f"  {be:>7s}: max={max(accs):.0%} avg={sum(accs)/len(accs):.0%} n={len(accs)}")

# By device
print()
print("=== Accuracy by device ===")
by_dev = defaultdict(list)
for e in entries:
    by_dev[e["device"]].append(e["acc"])
for dev in sorted(by_dev.keys()):
    accs = by_dev[dev]
    print(f"  {dev:>5s}: max={max(accs):.0%} avg={sum(accs)/len(accs):.0%} n={len(accs)}")

db.close()
