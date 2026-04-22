"""
State database: SQLite-backed immutable registry for teachers, lineage, and config.

Teachers and lineage are append-only. Nothing is ever deleted.
Runtime config is mutable and re-read each round for hot-reload.

Usage:
    db = StateDB("state/training.db")
    db.register_teacher("parity", 0.96, 120, {...config...})
    db.log_lineage("parity", 5, 0.63, {...config...}, "severity=1.0 changes={...}")
    teachers = db.get_teachers()
    config = db.get_runtime_config()
"""
import sqlite3
import json
import time
from pathlib import Path


class StateDB:
    """SQLite-backed immutable state for training."""

    def __init__(self, db_path="state/training.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self._create_tables()

    def _create_tables(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS teachers (
                task TEXT PRIMARY KEY,
                accuracy REAL NOT NULL,
                cycles INTEGER NOT NULL,
                config TEXT NOT NULL,
                checkpoint_path TEXT,
                graduated_at REAL NOT NULL,
                exp_id TEXT
            );

            CREATE TABLE IF NOT EXISTS lineage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task TEXT NOT NULL,
                round INTEGER NOT NULL,
                accuracy REAL NOT NULL,
                best_accuracy REAL NOT NULL,
                config TEXT NOT NULL,
                mutation TEXT,
                timestamp REAL NOT NULL,
                checkpoint_path TEXT
            );

            CREATE TABLE IF NOT EXISTS runtime_config (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at REAL NOT NULL
            );
        """)
        self.conn.commit()

    # ── Teachers (append-only) ─────────────────────────────────────

    def register_teacher(self, task, accuracy, cycles, config, exp_id=None,
                         checkpoint_path=None):
        """Register a graduated teacher. Never overwrites — first graduation wins."""
        try:
            self.conn.execute(
                "INSERT OR IGNORE INTO teachers (task, accuracy, cycles, config, "
                "checkpoint_path, graduated_at, exp_id) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (task, accuracy, cycles, json.dumps(config),
                 checkpoint_path, time.time(), exp_id)
            )
            self.conn.commit()
            return True
        except Exception:
            return False

    def get_teachers(self):
        """Get all graduated teachers. Returns dict: task → {accuracy, cycles, config, ...}"""
        cur = self.conn.execute("SELECT * FROM teachers")
        teachers = {}
        for row in cur.fetchall():
            teachers[row[0]] = {
                "task": row[0],
                "accuracy": row[1],
                "cycles": row[2],
                "config": json.loads(row[3]),
                "checkpoint_path": row[4],
                "graduated_at": row[5],
                "exp_id": row[6],
            }
        return teachers

    def is_teacher(self, task):
        """Check if a task has graduated."""
        cur = self.conn.execute("SELECT 1 FROM teachers WHERE task = ?", (task,))
        return cur.fetchone() is not None

    # ── Lineage (append-only) ──────────────────────────────────────

    def log_lineage(self, task, round_num, accuracy, best_accuracy, config,
                    mutation=None, checkpoint_path=None):
        """Log one training round for a task. Append-only, never modified."""
        self.conn.execute(
            "INSERT INTO lineage (task, round, accuracy, best_accuracy, config, "
            "mutation, timestamp, checkpoint_path) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (task, round_num, accuracy, best_accuracy,
             json.dumps(config), mutation, time.time(), checkpoint_path)
        )
        self.conn.commit()

    def get_lineage(self, task):
        """Get full lineage for a task, ordered by round."""
        cur = self.conn.execute(
            "SELECT round, accuracy, best_accuracy, config, mutation, timestamp "
            "FROM lineage WHERE task = ? ORDER BY round",
            (task,)
        )
        return [{
            "round": r[0], "accuracy": r[1], "best_accuracy": r[2],
            "config": json.loads(r[3]), "mutation": r[4], "timestamp": r[5],
        } for r in cur.fetchall()]

    def get_all_lineage(self):
        """Get lineage for all tasks."""
        cur = self.conn.execute(
            "SELECT DISTINCT task FROM lineage"
        )
        result = {}
        for (task,) in cur.fetchall():
            result[task] = self.get_lineage(task)
        return result

    def get_best_config(self, task):
        """Get the config that achieved the highest accuracy for a task."""
        cur = self.conn.execute(
            "SELECT config, accuracy FROM lineage WHERE task = ? "
            "ORDER BY best_accuracy DESC LIMIT 1",
            (task,)
        )
        row = cur.fetchone()
        if row:
            return json.loads(row[0]), row[1]
        return None, 0.0

    # ── Runtime config (mutable, hot-reload) ───────────────────────

    def set_config(self, key, value):
        """Set a runtime config value. Overwrites existing."""
        self.conn.execute(
            "INSERT OR REPLACE INTO runtime_config (key, value, updated_at) "
            "VALUES (?, ?, ?)",
            (key, json.dumps(value), time.time())
        )
        self.conn.commit()

    def get_config(self, key, default=None):
        """Get a runtime config value."""
        cur = self.conn.execute(
            "SELECT value FROM runtime_config WHERE key = ?", (key,)
        )
        row = cur.fetchone()
        if row:
            return json.loads(row[0])
        return default

    def get_all_config(self):
        """Get all runtime config as a dict."""
        cur = self.conn.execute("SELECT key, value FROM runtime_config")
        return {k: json.loads(v) for k, v in cur.fetchall()}

    # ── Export ─────────────────────────────────────────────────────

    def export_lineage_markdown(self, task, path):
        """Export lineage for a task as markdown."""
        lineage = self.get_lineage(task)
        if not lineage:
            return

        with open(path, "w") as f:
            f.write(f"# {task} — Training Lineage\n\n")
            f.write("| Round | Acc | Best | d_model | Layers | LR | WD | "
                   "Optimizer | Loss | Mutation |\n")
            f.write("|-------|-----|------|---------|--------|----|----|"
                   "-----------|------|-----------|\n")
            for e in lineage:
                c = e["config"]
                mut = e.get("mutation") or "—"
                f.write(f"| {e['round']} | {e['accuracy']:.0%} | "
                       f"{e['best_accuracy']:.0%} | {c.get('d_model', 64)} | "
                       f"{c.get('n_kernel_layers', 3)} | "
                       f"{c.get('lr', 1e-3):.0e} | "
                       f"{c.get('weight_decay', 0.1)} | "
                       f"{c.get('optimizer', 'adamw')} | "
                       f"{c.get('loss_fn', 'ce')} | {mut} |\n")

    def close(self):
        self.conn.close()
