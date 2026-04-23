"""Timeline of repeat_count mutations — what's winning over time?"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from state_db import StateDB

db = StateDB("three_pop/training.db")
lin = db.get_lineage("repeat_count")

print(f"repeat_count — full timeline ({len(lin)} entries)")
print(f"{'r':>4s} {'acc':>5s} {'role':>10s} {'backend':>7s} {'device':>5s} {'d':>4s} {'L':>2s} {'dS':>3s} {'opt':>6s} {'loss':>12s}")
print("-" * 75)

# Track which backend/device is the current champion
champion_backend = "triton"
champion_device = "cuda"

for e in lin:
    cfg = e.get("config", {})
    r = e["round"]
    a = e["accuracy"]
    role = e.get("role", "?")
    be = cfg.get("scan_backend", "triton")
    dev = cfg.get("device", "cuda")
    d = cfg.get("d_model", 64)
    L = cfg.get("n_kernel_layers", 3)
    ds = cfg.get("d_state", 16)
    opt = cfg.get("optimizer", "adamw")
    loss = cfg.get("loss_fn", "ce")

    marker = ""
    if role == "champion":
        champion_backend = be
        champion_device = dev
        marker = " << CHAMPION"
    elif a >= 0.70:
        marker = " *"

    print(f"r{r:>3d} {a:>4.0%} {role:>10s} {be:>7s} {dev:>5s} {d:>4d} {L:>2d} {ds:>3d} {opt:>6s} {loss:>12s}{marker}")

print()
print(f"Current champion: backend={champion_backend} device={champion_device}")

# Count backend/device distribution over time
print()
print("=== Backend/Device distribution over rounds ===")
early = [e for e in lin if e["round"] <= 10]
mid = [e for e in lin if 10 < e["round"] <= 100]
late = [e for e in lin if e["round"] > 100]

for label, group in [("Early (r1-10)", early), ("Mid (r11-100)", mid), ("Late (r100+)", late)]:
    if not group:
        continue
    n = len(group)
    jit = sum(1 for e in group if e.get("config", {}).get("scan_backend") == "jit")
    cpu = sum(1 for e in group if e.get("config", {}).get("device") == "cpu")
    above_60 = sum(1 for e in group if e["accuracy"] >= 0.60)
    print(f"  {label:15s}: {n:>3d} entries, {jit:>3d} jit ({jit*100//max(n,1)}%), {cpu:>2d} cpu ({cpu*100//max(n,1)}%), {above_60:>3d} above 60%")

db.close()
