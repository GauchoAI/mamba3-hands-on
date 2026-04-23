"""Check full history for a task."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from state_db import StateDB

task = sys.argv[1] if len(sys.argv) > 1 else "repeat_count"
db = StateDB("three_pop/training.db")

lin = db.get_lineage(task)
print(f"{task}: {len(lin)} lineage entries")
for e in lin:
    r = e["round"]
    a = e["accuracy"]
    b = e["best_accuracy"]
    role = e.get("role", "?")
    mut = e.get("mutation", "")
    if mut and len(mut) > 60:
        mut = mut[:60] + "..."
    print(f"  r{r} {role:10s}: acc={a:.0%} best={b:.0%}  {mut}")

status = db.get_task_status(task)
if status:
    ba = status["best_accuracy"]
    print(f"\ntask_status: best={ba:.0%} status={status['status']}")

# Check cycle_history for max accuracy ever
cur = db.conn.execute(
    "SELECT MAX(accuracy) FROM cycle_history WHERE task=?", (task,))
row = cur.fetchone()
if row and row[0]:
    print(f"cycle_history max accuracy: {row[0]:.0%}")

db.close()
