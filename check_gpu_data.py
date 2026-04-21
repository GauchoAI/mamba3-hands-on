"""Check GPU history data in the DB."""
from metrics_db import MetricsReader

r = MetricsReader()
gpu = r.get_gpu_history(limit=20)
print(f"GPU history: {len(gpu)} entries")
for g in gpu[-10:]:
    print(f"  gpu={g['gpu_pct']:.0f}%  vram={g['mem_pct']:.0f}%  "
          f"workers={g['n_workers']}  exps={g['n_experiments']}")

# Check what the renderer would see
exps = r.get_experiments()
print(f"\nExperiments: {len(exps)}")
running = [e for e in exps if e["status"] == "running"]
print(f"Running: {len(running)}")

# Try rendering
print("\nAttempting render...")
try:
    from render import render_once
    render_once()
    print("Render OK")
except Exception as e:
    import traceback
    print(f"Render FAILED: {e}")
    traceback.print_exc()
