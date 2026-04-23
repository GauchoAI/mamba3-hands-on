# Stateless Orchestrator — Implementation Plan

## Goal

The orchestrator becomes a thin scheduler with zero in-memory state.
All state lives in SQLite + checkpoints. Workers are self-sufficient.
Kill and replace the orchestrator at any time with zero consequences.

## Current State (what the orchestrator does today)

1. Reads teachers from DB (✅ already stateless)
2. Tracks `task_config` in memory (❌ should be in DB)
3. Tracks `task_best` in memory (❌ should be in DB)
4. Tracks `task_best_round` in memory (❌ should be in DB)
5. Runs diagnostician (❌ should be in workers)
6. Builds `on_cycle` callback for Firebase (❌ workers could push directly)
7. Runs champion-challenger comparison (❌ workers could self-compare)
8. Logs lineage (✅ workers already log their own too)
9. Pushes Firebase snapshot (could be separate)
10. PID lock (✅ keep this)

## Target State

**Orchestrator responsibilities (thin):**
- Read `runtime_config` from DB
- Query DB: "which tasks need training?"
- Spawn subprocess workers
- Wait for them to finish
- Repeat

**Worker responsibilities (fat):**
- Load checkpoint (resume)
- Train N cycles
- Evaluate with error analysis
- Run diagnostician on self
- Log cycle_history, error_analysis, lineage to DB
- Push to Firebase directly
- If mastered: register teacher in DB, save cache
- If plateaued: create own challenger (self-mutate)
- Save checkpoint
- Clear active_run

**DB responsibilities (source of truth):**
- `task_status` table: what state each task is in (waiting/training/mastered)
- `task_config`: current config per task (not in orchestrator memory)
- Everything else already exists

## Implementation Sequence

### Commit 1: Add `task_status` table to DB

**Files:** `state_db.py`

New table:
```sql
CREATE TABLE IF NOT EXISTS task_status (
    task TEXT PRIMARY KEY,
    status TEXT NOT NULL,      -- waiting|training|mastered
    current_config TEXT,       -- JSON config
    best_accuracy REAL,
    best_round INTEGER,
    total_cycles INTEGER,
    updated_at REAL
);
```

Methods:
- `get_task_status(task)` → dict
- `update_task_status(task, status, config, best_acc, ...)`
- `get_tasks_needing_training()` → list of tasks not mastered
- `get_task_config(task)` → config dict from DB

Migration: populate from existing lineage on first run.

**Deploy:** `git push` + `git pull` on H100. Workers pick up new
state_db.py automatically. No restart.

### Commit 2: Workers update task_status

**Files:** `specialist_trainer.py`

After training completes, worker updates DB:
```python
_db.update_task_status(task, "training", config, best_acc, cycle)
if best_acc >= target_acc:
    _db.update_task_status(task, "mastered", config, best_acc, cycle)
    _db.register_teacher(task, best_acc, cycle, config, ...)
```

**Deploy:** `git push` + `git pull`. Workers pick up automatically.
No restart.

### Commit 3: Workers run diagnostician on themselves

**Files:** `specialist_trainer.py`

After evaluation, worker checks its own diagnostics:
```python
from diagnostician import Diagnostician
diag = Diagnostician(_db)
signals = diag.diagnose(task)
# Store signals in active_run for orchestrator to read
_db.update_active_run(task, cycle, ..., signals=json.dumps(signals))
```

This is observation only — workers don't mutate themselves.
The diagnostic signals are recorded for the next mutation decision.

**Deploy:** `git push` + `git pull`. No restart.

### Commit 4: Workers handle champion-challenger

**Files:** `specialist_trainer.py` (add `--mode champion|challenger`)

When launched as champion: train normally, save checkpoint.
When launched as challenger: train with mutated config, compare
against champion's best_accuracy from DB.

```python
if args.mode == "challenger":
    champion_best = _db.get_task_status(task)["best_accuracy"]
    if best_acc > champion_best:
        # Challenger wins — update task_status with new config
        _db.update_task_status(task, "training", config, best_acc, cycle)
        print(f"✓ Challenger wins: {best_acc:.0%} > {champion_best:.0%}")
    else:
        # Champion holds — restore champion checkpoint
        print(f"✗ Champion holds: {champion_best:.0%} >= {best_acc:.0%}")
        restore_champion_checkpoint(task)
```

**Deploy:** `git push` + `git pull`. No restart.

### Commit 5: Slim orchestrator reads everything from DB

**Files:** `three_populations.py` (major rewrite)

```python
def run(args):
    lock = _acquire_lock()
    db = StateDB("three_pop/training.db")

    while True:
        # Hot-reload config
        cfg = db.get_all_config()
        pool_size = cfg.get("max_concurrent", 4)
        cycles = cfg.get("cycles_per_round", 10)

        # What needs training?
        tasks = db.get_tasks_needing_training()
        if not tasks:
            print("All mastered!")
            break

        # Spawn batch
        for batch in chunks(tasks, pool_size):
            procs = {}
            for task in batch:
                task_cfg = db.get_task_config(task)
                # Check if diagnostician flagged anything
                status = db.get_task_status(task)
                signals = json.loads(status.get("signals", "[]"))

                if should_run_challenger(task, db):
                    # Spawn champion + challenger
                    procs[task] = spawn_worker(task, task_cfg, mode="champion")
                    challenger_cfg = create_challenger(task_cfg, signals, db)
                    procs[f"{task}_challenger"] = spawn_worker(
                        task, challenger_cfg, mode="challenger")
                else:
                    procs[task] = spawn_worker(task, task_cfg, mode="champion")

            # Wait for all — workers handle their own state
            for name, proc in procs.items():
                proc.wait(timeout=600)

        # Firebase sync (could be separate process)
        db.sync_to_firebase()

        time.sleep(1)
```

The orchestrator has NO in-memory state. Everything comes from DB.
Kill it at any point — nothing lost.

**Deploy:** `git push` + `git pull` + restart (PID lock). Workers
in flight finish and save their own lineage. New orchestrator reads
DB and continues.

### Commit 6: Firebase push as separate concern

**Files:** `firebase_sync_worker.py` (new)

Optional: a tiny process that polls the DB every 5 seconds and
pushes snapshots to Firebase. Completely independent of orchestrator.

```python
while True:
    db = StateDB("three_pop/training.db")
    snapshot = build_snapshot_from_db(db)
    fb._put("mamba3/snapshot", snapshot)
    db.close()
    time.sleep(5)
```

This means Firebase stays fresh even between orchestrator restarts.

**Deploy:** Start alongside orchestrator. Independent lifecycle.

## Dependency Graph

```
Commit 1: task_status table (foundation)
    ↓
Commit 2: workers update task_status
    ↓
Commit 3: workers run diagnostician (uses task_status)
    ↓
Commit 4: workers handle champion-challenger (uses task_status)
    ↓
Commit 5: slim orchestrator (reads task_status, spawns workers)
    ↓
Commit 6: firebase sync worker (optional, independent)
```

Each commit is independently deployable. Commits 1-4 require NO
orchestrator restart. Only commit 5 requires a restart (and that's
the final one that makes restarts painless forever after).

## Migration Path

The transition is gradual:
- Commits 1-4: workers get smarter, orchestrator still works the old way
- Commit 5: orchestrator becomes thin, relies on smart workers
- At no point does training stop or state get lost
- The DB is always the source of truth
- Old orchestrator and new orchestrator produce the same results

## Verification

After each commit:
1. Check DB has the new data (`audit_db.py`)
2. Check workers are writing to new tables
3. Check orchestrator reads from DB (not memory)
4. Kill orchestrator mid-batch — verify no data lost
5. Restart — verify it continues seamlessly

## End State

```
┌─────────────────────┐
│   Orchestrator      │  Stateless. Reads DB.
│   (disposable)      │  Spawns workers. Waits.
└────────┬────────────┘
         │ subprocess.Popen
         ▼
┌─────────────────────┐
│   Worker            │  Self-sufficient.
│   (specialist_      │  Trains, evaluates, diagnoses,
│    trainer.py)      │  logs lineage, pushes Firebase,
│                     │  handles champion-challenger.
└────────┬────────────┘
         │ writes to
         ▼
┌─────────────────────┐
│   SQLite DB         │  Source of truth.
│   (training.db)     │  Teachers, lineage, diagnostics,
│                     │  task_status, cycle_history,
│                     │  error_analysis, runtime_config.
└─────────────────────┘
         │ read by
         ▼
┌─────────────────────┐
│   Firebase Sync     │  Independent process.
│   (optional)        │  Polls DB, pushes to Firebase.
└─────────────────────┘
```

Kill the orchestrator → workers finish, DB is safe.
Kill Firebase sync → UI goes stale, training continues.
Kill a worker → checkpoint saved, next batch picks it up.
Nothing is a single point of failure.
