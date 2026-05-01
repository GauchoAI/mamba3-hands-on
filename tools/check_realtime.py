"""Check real-time DB state."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from state_db import StateDB

db = StateDB("three_pop/training.db")

runs = db.get_active_runs()
print(f"Active runs: {len(runs)}")
for r in runs:
    t = r["task"]
    c = r["cycle"]
    a = r["accuracy"]
    print(f"  {t}: cycle={c} acc={a}")

hist = db.conn.execute("SELECT COUNT(*) FROM cycle_history").fetchone()[0]
print(f"\nCycle history: {hist} entries")
if hist > 0:
    last = db.conn.execute(
        "SELECT task, cycle, accuracy, loss, grad_norm, param_norm, gpu_mem_mb "
        "FROM cycle_history ORDER BY id DESC LIMIT 5"
    ).fetchall()
    for r in last:
        t, c, a, l, g, p, m = r
        print(f"  {t} c{c}: acc={a} loss={l:.3f} grad={g:.1f} params={p:.0f} gpu={m:.0f}MB")

db.close()
