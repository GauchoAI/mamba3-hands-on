"""Audit the state database — what do we have?"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from state_db import StateDB
import json

db = StateDB("three_pop/training.db")

# Tables and counts
cur = db.conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = [r[0] for r in cur.fetchall()]
print("=== TABLES ===")
for t in tables:
    count = db.conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
    cols = [d[1] for d in db.conn.execute(f"PRAGMA table_info({t})").fetchall()]
    print(f"  {t}: {count} rows — columns: {cols}")

# Latest lineage entry
print("\n=== LATEST LINEAGE ENTRY ===")
cur = db.conn.execute("SELECT * FROM lineage ORDER BY id DESC LIMIT 1")
cols = [d[0] for d in cur.description]
row = cur.fetchone()
if row:
    for c, v in zip(cols, row):
        val = str(v)[:100] if v else "NULL"
        print(f"  {c}: {val}")

# Latest experiment
print("\n=== LATEST EXPERIMENT ===")
cur = db.conn.execute("SELECT * FROM experiments ORDER BY id DESC LIMIT 1")
cols = [d[0] for d in cur.description]
row = cur.fetchone()
if row:
    for c, v in zip(cols, row):
        val = str(v)[:100] if v else "NULL"
        print(f"  {c}: {val}")

# Model cards for stuck tasks
print("\n=== MODEL CARDS ===")
for task in ["parity", "arithmetic_next", "binary_pattern_next"]:
    card = db.build_model_card(task)
    n_teachers = len(card.get("teachers", []))
    best = card.get("best_accuracy", 0)
    teachers_str = ", ".join(t["model"] for t in card.get("teachers", []))
    print(f"  {task}: best={best:.0%}, {n_teachers} teachers [{teachers_str}]")

# Teacher eval cache
print("\n=== TEACHER EVAL CACHE ===")
all_evals = db.get_all_teacher_scores()
for e in all_evals:
    print(f"  {e['teacher']} → {e['task']}: {e['accuracy']:.0%}")

# Runtime config
print("\n=== RUNTIME CONFIG ===")
for k, v in db.get_all_config().items():
    print(f"  {k}: {v}")

# What's MISSING
print("\n=== WHAT'S MISSING ===")
missing = []

# No real-time "currently training" status
cur = db.conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='active_runs'")
if not cur.fetchone():
    missing.append("active_runs table — no real-time tracking of what's training NOW")

# No per-cycle granularity in lineage (only per-round)
missing.append("per-cycle accuracy history — lineage only records end-of-round, not cycle-by-cycle progress")

# No parent_id pointer in lineage
cur = db.conn.execute("PRAGMA table_info(lineage)")
cols = [r[1] for r in cur.fetchall()]
if "parent_id" not in cols:
    missing.append("parent_id in lineage — have to infer parent from round order, not explicit pointer")

# No explicit won/lost in lineage
if "won" not in cols:
    missing.append("won column in lineage — have to compute from accuracy vs previous best")

for m in missing:
    print(f"  - {m}")

db.close()
