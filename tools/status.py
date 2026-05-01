"""Quick status report from metrics.db."""
import json
from metrics_db import MetricsReader

r = MetricsReader()

exps = r.get_experiments()
running = [e for e in exps if e["status"] == "running"]
print(f"Experiments: {len(exps)} total, {len(running)} running")

print("\nTop 10:")
for i, e in enumerate(exps[:10]):
    cfg = json.loads(e["config_json"]) if e.get("config_json") else {}
    wd = cfg.get("weight_decay", 0)
    bk = cfg.get("backend", "pt")
    method = f"wd={wd}" if wd > 0 else "perp"
    if bk == "tinygrad":
        method += "/tg"
    peak = e.get("peak_fresh", 0) or 0
    cyc = e.get("max_cycle", 0) or 0
    dm = e.get("d_model", "?")
    nl = e.get("n_kernel_layers", "?")
    eid = e["exp_id"]
    st = e["status"]
    print(f"  {i+1}. {eid}: {peak:.1%} d={dm} L={nl} [{method}] cyc={cyc} [{st}]")

tasks = r.get_all_task_latest()
best = {}
for t in tasks:
    tt = t["task_type"]
    if tt not in best or t["accuracy"] > best[tt][1]:
        best[tt] = (t["exp_id"], t["accuracy"])

print("\nBest per task:")
for task in sorted(best, key=lambda t: -best[t][1]):
    eid, acc = best[task]
    if acc > 0:
        print(f"  {task}: {acc:.0%} ({eid})")

events = r.get_events(limit=5)
print("\nLatest events:")
for e in events:
    print(f"  {e['event_type']}: {e.get('exp_id','')} {e.get('details','')}")

gpu = r.get_gpu_history(limit=1)
if gpu:
    g = gpu[0]
    print(f"\nGPU: {g['gpu_pct']:.0f}%  VRAM: {g['mem_pct']:.0f}%  Workers: {g['n_workers']}")
