"""
Metrics database — SQLite backend for all training telemetry.

Workers write, coordinator writes, renderers read. No coupling.

Tables:
  experiments  — one row per experiment config
  cycles       — one row per (experiment, cycle) with loss, accuracy, etc.
  tasks        — one row per (experiment, cycle, task_type) with per-task accuracy
  gpu          — one row per coordinator heartbeat with GPU/VRAM usage
  events       — mastery, unlock, evolve, prune events with timestamps
  teacher      — teacher state snapshots
"""
import sqlite3
import json
import time
from pathlib import Path
from contextlib import contextmanager


DB_PATH = "metrics.db"


def get_db(path=None):
    """Get a database connection with WAL mode for concurrent access."""
    db = sqlite3.connect(path or DB_PATH, timeout=10)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=NORMAL")
    db.row_factory = sqlite3.Row
    return db


def init_db(path=None):
    """Create all tables if they don't exist."""
    db = get_db(path)

    db.executescript("""
    CREATE TABLE IF NOT EXISTS experiments (
        exp_id TEXT PRIMARY KEY,
        d_model INTEGER,
        d_state INTEGER,
        headdim INTEGER,
        n_kernel_layers INTEGER,
        batch_size INTEGER,
        lr REAL,
        weight_decay REAL,
        n_params INTEGER,
        status TEXT DEFAULT 'running',
        created_at REAL,
        config_json TEXT
    );

    CREATE TABLE IF NOT EXISTS cycles (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        exp_id TEXT,
        cycle INTEGER,
        loss REAL,
        fresh_acc REAL,
        best_fresh REAL,
        train_acc REAL,
        elapsed_s REAL,
        timestamp REAL,
        FOREIGN KEY (exp_id) REFERENCES experiments(exp_id)
    );

    CREATE TABLE IF NOT EXISTS tasks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        exp_id TEXT,
        cycle INTEGER,
        task_type TEXT,
        accuracy REAL,
        difficulty REAL DEFAULT 0,
        status TEXT,
        timestamp REAL,
        FOREIGN KEY (exp_id) REFERENCES experiments(exp_id)
    );

    CREATE TABLE IF NOT EXISTS gpu (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        gpu_pct REAL,
        mem_pct REAL,
        mem_used_mb REAL,
        mem_total_mb REAL,
        n_workers INTEGER,
        n_experiments INTEGER,
        timestamp REAL
    );

    CREATE TABLE IF NOT EXISTS events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        event_type TEXT,
        exp_id TEXT,
        details TEXT,
        timestamp REAL
    );

    CREATE TABLE IF NOT EXISTS teacher (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        exp_id TEXT,
        cycle INTEGER,
        unlocked_tasks TEXT,
        status_text TEXT,
        mastery_log TEXT,
        learning_report TEXT,
        timestamp REAL
    );

    CREATE INDEX IF NOT EXISTS idx_cycles_exp ON cycles(exp_id, cycle);
    CREATE INDEX IF NOT EXISTS idx_tasks_exp ON tasks(exp_id, cycle);
    CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type, timestamp);
    CREATE INDEX IF NOT EXISTS idx_gpu_time ON gpu(timestamp);
    """)

    db.commit()
    return db


# ── Write API (used by workers and coordinator) ────────────────────

class MetricsWriter:
    """Thread-safe writer for training metrics."""

    def __init__(self, path=None):
        self.path = path or DB_PATH
        init_db(self.path)

    def _db(self):
        return get_db(self.path)

    def register_experiment(self, exp_id, config, n_params):
        db = self._db()
        db.execute("""
            INSERT OR REPLACE INTO experiments
            (exp_id, d_model, d_state, headdim, n_kernel_layers,
             batch_size, lr, weight_decay, n_params, status, created_at, config_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'running', ?, ?)
        """, (
            exp_id, config.get("d_model"), config.get("d_state"),
            config.get("headdim"), config.get("n_kernel_layers"),
            config.get("batch_size"), config.get("lr"),
            config.get("weight_decay", 0), n_params,
            time.time(), json.dumps(config),
        ))
        db.commit()
        db.close()

    def log_cycle(self, exp_id, cycle, loss, fresh_acc, best_fresh,
                  train_acc=None, elapsed_s=None):
        db = self._db()
        db.execute("""
            INSERT INTO cycles
            (exp_id, cycle, loss, fresh_acc, best_fresh, train_acc, elapsed_s, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (exp_id, cycle, loss, fresh_acc, best_fresh, train_acc,
              elapsed_s, time.time()))
        db.commit()
        db.close()

    def log_tasks(self, exp_id, cycle, type_accs, difficulties=None):
        db = self._db()
        now = time.time()
        for task_type, acc in type_accs.items():
            diff = difficulties.get(task_type, 0) if difficulties else 0
            status = "mastered" if acc >= 0.9 else ("learning" if acc >= 0.4 else "struggling")
            db.execute("""
                INSERT INTO tasks (exp_id, cycle, task_type, accuracy, difficulty, status, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (exp_id, cycle, task_type, acc, diff, status, now))
        db.commit()
        db.close()

    def log_gpu(self, gpu_pct, mem_pct, mem_used_mb, mem_total_mb,
                n_workers, n_experiments):
        db = self._db()
        db.execute("""
            INSERT INTO gpu (gpu_pct, mem_pct, mem_used_mb, mem_total_mb,
                           n_workers, n_experiments, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (gpu_pct, mem_pct, mem_used_mb, mem_total_mb,
              n_workers, n_experiments, time.time()))
        db.commit()
        db.close()

    def log_event(self, event_type, exp_id=None, details=None):
        db = self._db()
        db.execute("""
            INSERT INTO events (event_type, exp_id, details, timestamp)
            VALUES (?, ?, ?, ?)
        """, (event_type, exp_id, details, time.time()))
        db.commit()
        db.close()

    def log_teacher(self, exp_id, cycle, teacher):
        db = self._db()
        db.execute("""
            INSERT INTO teacher
            (exp_id, cycle, unlocked_tasks, status_text, mastery_log, learning_report, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            exp_id, cycle,
            json.dumps(list(teacher.unlocked_tasks)),
            teacher.get_status(),
            json.dumps(teacher.mastery_log),
            teacher.get_learning_report() if teacher.mastery_log else "",
            time.time(),
        ))
        db.commit()
        db.close()

    def update_status(self, exp_id, status):
        db = self._db()
        db.execute("UPDATE experiments SET status = ? WHERE exp_id = ?",
                   (status, exp_id))
        db.commit()
        db.close()


# ── Read API (used by renderers) ───────────────────────────────────

class MetricsReader:
    """Read-only access to metrics database."""

    def __init__(self, path=None):
        self.path = path or DB_PATH

    def _db(self):
        return get_db(self.path)

    def get_experiments(self):
        """All experiments, sorted by best fresh accuracy."""
        db = self._db()
        rows = db.execute("""
            SELECT e.*, MAX(c.best_fresh) as peak_fresh, MAX(c.cycle) as max_cycle
            FROM experiments e
            LEFT JOIN cycles c ON e.exp_id = c.exp_id
            GROUP BY e.exp_id
            ORDER BY peak_fresh DESC
        """).fetchall()
        db.close()
        return [dict(r) for r in rows]

    def get_cycle_history(self, exp_id):
        """All cycle data for one experiment."""
        db = self._db()
        rows = db.execute("""
            SELECT * FROM cycles WHERE exp_id = ? ORDER BY cycle
        """, (exp_id,)).fetchall()
        db.close()
        return [dict(r) for r in rows]

    def get_task_history(self, exp_id, task_type):
        """Accuracy over time for one experiment + task."""
        db = self._db()
        rows = db.execute("""
            SELECT cycle, accuracy, difficulty FROM tasks
            WHERE exp_id = ? AND task_type = ?
            ORDER BY cycle
        """, (exp_id, task_type)).fetchall()
        db.close()
        return [dict(r) for r in rows]

    def get_all_task_latest(self):
        """Latest accuracy per (experiment, task) — for leaderboard."""
        db = self._db()
        rows = db.execute("""
            SELECT t.exp_id, t.task_type, t.accuracy, t.difficulty, t.status
            FROM tasks t
            INNER JOIN (
                SELECT exp_id, task_type, MAX(cycle) as max_cycle
                FROM tasks GROUP BY exp_id, task_type
            ) latest ON t.exp_id = latest.exp_id
                AND t.task_type = latest.task_type
                AND t.cycle = latest.max_cycle
        """).fetchall()
        db.close()
        return [dict(r) for r in rows]

    def get_gpu_history(self, limit=500):
        """GPU usage over time."""
        db = self._db()
        rows = db.execute("""
            SELECT * FROM gpu ORDER BY timestamp DESC LIMIT ?
        """, (limit,)).fetchall()
        db.close()
        return [dict(r) for r in reversed(rows)]

    def get_events(self, event_type=None, limit=100):
        """Recent events."""
        db = self._db()
        if event_type:
            rows = db.execute("""
                SELECT * FROM events WHERE event_type = ?
                ORDER BY timestamp DESC LIMIT ?
            """, (event_type, limit)).fetchall()
        else:
            rows = db.execute("""
                SELECT * FROM events ORDER BY timestamp DESC LIMIT ?
            """, (limit,)).fetchall()
        db.close()
        return [dict(r) for r in reversed(rows)]

    def get_latest_teacher(self):
        """Most recent teacher state."""
        db = self._db()
        row = db.execute("""
            SELECT * FROM teacher ORDER BY timestamp DESC LIMIT 1
        """).fetchone()
        db.close()
        return dict(row) if row else None

    def get_active_tasks(self):
        """All task types that have ever had non-zero accuracy."""
        db = self._db()
        rows = db.execute("""
            SELECT DISTINCT task_type FROM tasks
            WHERE accuracy > 0 ORDER BY task_type
        """).fetchall()
        db.close()
        return [r["task_type"] for r in rows]


# ── Quick test ──────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    test_path = "/tmp/test_metrics.db"
    if os.path.exists(test_path):
        os.remove(test_path)

    w = MetricsWriter(test_path)

    # Write
    w.register_experiment("exp_001", {"d_model": 64, "lr": 1e-3, "weight_decay": 0.1}, 45000)
    w.log_cycle("exp_001", 1, 6.5, 0.03, 0.03, elapsed_s=5.0)
    w.log_cycle("exp_001", 2, 5.8, 0.07, 0.07, elapsed_s=4.8)
    w.log_tasks("exp_001", 1, {"parity": 0.4, "same_different": 0.0})
    w.log_tasks("exp_001", 2, {"parity": 0.6, "same_different": 0.1})
    w.log_gpu(88.0, 15.0, 5900, 80000, 6, 20)
    w.log_event("spawn", "exp_001", "initial seed")
    w.log_event("mastery", "exp_001", "parity mastered at cycle 50")

    # Read
    r = MetricsReader(test_path)
    exps = r.get_experiments()
    print(f"Experiments: {len(exps)}")
    print(f"  {exps[0]['exp_id']}: peak_fresh={exps[0]['peak_fresh']}")

    history = r.get_cycle_history("exp_001")
    print(f"Cycles: {len(history)}")

    tasks = r.get_active_tasks()
    print(f"Active tasks: {tasks}")

    gpu = r.get_gpu_history()
    print(f"GPU readings: {len(gpu)}, latest: {gpu[-1]['gpu_pct']}%")

    events = r.get_events()
    print(f"Events: {len(events)}")

    os.remove(test_path)
    print("\nAll tests passed.")
