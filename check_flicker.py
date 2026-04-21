"""Check for flickering in fresh accuracy."""
from metrics_db import MetricsReader

r = MetricsReader()
exps = r.get_experiments()

# Check top 3 experiments
for exp in exps[:3]:
    eid = exp["exp_id"]
    history = r.get_cycle_history(eid)
    if not history:
        continue
    print(f"\n{eid} (best={exp.get('peak_fresh',0) or 0:.1%}):")
    for h in history[-15:]:
        fresh = h["fresh_acc"] or 0
        loss = h["loss"] or 0
        bar = "█" * int(fresh * 100) + "░" * (30 - int(fresh * 100))
        print(f"  cycle {h['cycle']:4d}  fresh={fresh:.1%}  {bar}")
