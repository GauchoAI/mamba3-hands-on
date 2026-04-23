"""
State database: SQLite-backed immutable registry.

Three sacred tables (append-only, never deleted):
  - teachers: graduated tasks with winning config
  - lineage: every training round, every config tried, every result
  - experiments: every experiment (champion or challenger) with full metadata

One mutable table:
  - runtime_config: hot-reload settings (re-read each round, no restart)

Firebase sync: push teachers + lineage to Firebase for redundancy + UI.

Usage:
    db = StateDB("state/training.db")
    db.register_teacher("parity", 0.96, 120, {...}, checkpoint_path="...")
    db.log_lineage("parity", 5, 0.63, 0.63, {...}, mutation="...")
    db.sync_to_firebase()  # push everything to Firebase
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
        self._migrate()

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
                role TEXT DEFAULT 'champion',
                timestamp REAL NOT NULL,
                checkpoint_path TEXT,
                teachers TEXT DEFAULT '[]'
            );

            CREATE TABLE IF NOT EXISTS experiments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task TEXT NOT NULL,
                exp_id TEXT,
                round INTEGER NOT NULL,
                accuracy REAL NOT NULL,
                best_accuracy REAL NOT NULL,
                config TEXT NOT NULL,
                n_params INTEGER,
                cycles INTEGER,
                role TEXT DEFAULT 'champion',
                mutation TEXT,
                parent_exp TEXT,
                timestamp REAL NOT NULL,
                checkpoint_path TEXT
            );

            CREATE TABLE IF NOT EXISTS runtime_config (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS active_runs (
                task TEXT PRIMARY KEY,
                exp_id TEXT,
                cycle INTEGER NOT NULL,
                accuracy REAL,
                best_accuracy REAL,
                loss REAL,
                config TEXT,
                started_at REAL,
                updated_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS cycle_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task TEXT NOT NULL,
                cycle INTEGER NOT NULL,
                accuracy REAL,
                loss REAL,
                distill_loss REAL,
                grad_norm REAL,
                lr REAL,
                forward_ms REAL,
                backward_ms REAL,
                eval_ms REAL,
                gpu_mem_mb REAL,
                param_norm REAL,
                timestamp REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS teacher_eval_cache (
                teacher TEXT NOT NULL,
                task TEXT NOT NULL,
                accuracy REAL NOT NULL,
                n_examples INTEGER NOT NULL,
                timestamp REAL NOT NULL,
                PRIMARY KEY (teacher, task)
            );
        """)
        self.conn.commit()

    def _migrate(self):
        """Add columns/tables to existing DB if missing."""
        try:
            self.conn.execute("SELECT teachers FROM lineage LIMIT 1")
        except Exception:
            self.conn.execute("ALTER TABLE lineage ADD COLUMN teachers TEXT DEFAULT '[]'")
            self.conn.commit()
        # Ensure new tables exist (idempotent — CREATE IF NOT EXISTS)
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS active_runs (
                task TEXT PRIMARY KEY,
                exp_id TEXT,
                cycle INTEGER NOT NULL,
                accuracy REAL,
                best_accuracy REAL,
                loss REAL,
                config TEXT,
                started_at REAL,
                updated_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS cycle_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task TEXT NOT NULL,
                cycle INTEGER NOT NULL,
                accuracy REAL,
                loss REAL,
                distill_loss REAL,
                grad_norm REAL,
                lr REAL,
                forward_ms REAL,
                backward_ms REAL,
                eval_ms REAL,
                gpu_mem_mb REAL,
                param_norm REAL,
                timestamp REAL NOT NULL
            );
        """)
        self.conn.commit()

    # ── Active runs (real-time, overwritten each cycle) ────────────

    def update_active_run(self, task, cycle, accuracy=None, best_accuracy=None,
                          loss=None, config=None, exp_id=None):
        """Update real-time status for a running task. Overwrites."""
        self.conn.execute(
            "INSERT OR REPLACE INTO active_runs "
            "(task, exp_id, cycle, accuracy, best_accuracy, loss, config, "
            "started_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, "
            "COALESCE((SELECT started_at FROM active_runs WHERE task=?), ?), ?)",
            (task, exp_id, cycle, accuracy, best_accuracy, loss,
             json.dumps(config) if config else None,
             task, time.time(), time.time())
        )
        self.conn.commit()

    def clear_active_run(self, task):
        self.conn.execute("DELETE FROM active_runs WHERE task = ?", (task,))
        self.conn.commit()

    def get_active_runs(self):
        cur = self.conn.execute("SELECT * FROM active_runs ORDER BY updated_at DESC")
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    # ── Cycle history (append-only, per-cycle granularity) ─────────

    def log_cycle(self, task, cycle, accuracy=None, loss=None,
                  distill_loss=None, grad_norm=None, lr=None,
                  forward_ms=None, backward_ms=None, eval_ms=None,
                  gpu_mem_mb=None, param_norm=None):
        """Log detailed per-cycle metrics. Append-only."""
        self.conn.execute(
            "INSERT INTO cycle_history "
            "(task, cycle, accuracy, loss, distill_loss, grad_norm, lr, "
            "forward_ms, backward_ms, eval_ms, gpu_mem_mb, param_norm, timestamp) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (task, cycle, accuracy, loss, distill_loss, grad_norm, lr,
             forward_ms, backward_ms, eval_ms, gpu_mem_mb, param_norm, time.time())
        )
        self.conn.commit()

    def get_cycle_history(self, task, last_n=None):
        if last_n:
            cur = self.conn.execute(
                "SELECT * FROM cycle_history WHERE task = ? ORDER BY cycle DESC LIMIT ?",
                (task, last_n))
        else:
            cur = self.conn.execute(
                "SELECT * FROM cycle_history WHERE task = ? ORDER BY cycle", (task,))
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    # ── Teachers (append-only) ─────────────────────────────────────

    def register_teacher(self, task, accuracy, cycles, config, exp_id=None,
                         checkpoint_path=None):
        """Register a graduated teacher. First graduation wins — never overwrites."""
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
        """Get all graduated teachers."""
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
        cur = self.conn.execute("SELECT 1 FROM teachers WHERE task = ?", (task,))
        return cur.fetchone() is not None

    # ── Lineage (append-only) ──────────────────────────────────────

    def log_lineage(self, task, round_num, accuracy, best_accuracy, config,
                    mutation=None, role="champion", checkpoint_path=None,
                    teachers=None):
        """Log one training round. Append-only."""
        self.conn.execute(
            "INSERT INTO lineage (task, round, accuracy, best_accuracy, config, "
            "mutation, role, timestamp, checkpoint_path, teachers) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (task, round_num, accuracy, best_accuracy,
             json.dumps(config), mutation, role, time.time(), checkpoint_path,
             json.dumps(teachers or []))
        )
        self.conn.commit()

    def get_lineage(self, task):
        cur = self.conn.execute(
            "SELECT round, accuracy, best_accuracy, config, mutation, role, "
            "timestamp, teachers "
            "FROM lineage WHERE task = ? ORDER BY round, id",
            (task,)
        )
        results = []
        for r in cur.fetchall():
            teachers_raw = r[7] if r[7] else "[]"
            try:
                teachers = json.loads(teachers_raw)
            except (json.JSONDecodeError, TypeError):
                teachers = []
            results.append({
                "round": r[0], "accuracy": r[1], "best_accuracy": r[2],
                "config": json.loads(r[3]), "mutation": r[4], "role": r[5],
                "timestamp": r[6], "teachers": teachers,
            })
        return results

    def build_model_card(self, task, decay=0.8):
        """Walk lineage, collect inherited teachers, compute weights.

        Returns dict with task, config, and teachers list sorted by weight.
        Teachers accumulate across generations — each ancestor's teacher
        is inherited with weight decaying by generation distance.
        """
        lineage = self.get_lineage(task)
        if not lineage:
            return {"task": task, "config": {}, "teachers": []}

        teachers = []
        seen = set()

        # Walk backwards — most recent first
        for i, entry in enumerate(reversed(lineage)):
            # Check config for teacher_model
            cfg = entry.get("config", {})
            teacher = cfg.get("teacher_model")
            if teacher and teacher not in seen:
                weight = decay ** i
                teachers.append({
                    "model": teacher,
                    "weight": round(weight, 3),
                    "from_round": entry["round"],
                })
                seen.add(teacher)

            # Also check teachers field (accumulated from breeding)
            for t in entry.get("teachers", []):
                if t.get("model") and t["model"] not in seen:
                    # Extra decay for teachers inherited from ancestors
                    weight = t.get("weight", 1.0) * (decay ** i)
                    teachers.append({
                        "model": t["model"],
                        "weight": round(weight, 3),
                        "from_round": t.get("from_round", entry["round"]),
                    })
                    seen.add(t["model"])

        # Sort by weight descending
        teachers.sort(key=lambda t: -t["weight"])

        best_cfg, best_acc = self.get_best_config(task)
        return {
            "task": task,
            "config": best_cfg or {},
            "best_accuracy": best_acc,
            "teachers": teachers,
        }

    def get_all_lineage(self):
        cur = self.conn.execute("SELECT DISTINCT task FROM lineage")
        return {task: self.get_lineage(task) for (task,) in cur.fetchall()}

    def get_best_config(self, task):
        """Get the config that achieved the highest accuracy."""
        cur = self.conn.execute(
            "SELECT config, best_accuracy FROM lineage WHERE task = ? "
            "ORDER BY best_accuracy DESC LIMIT 1",
            (task,)
        )
        row = cur.fetchone()
        if row:
            return json.loads(row[0]), row[1]
        return None, 0.0

    # ── Experiments (append-only) ──────────────────────────────────

    def log_experiment(self, task, round_num, accuracy, best_accuracy, config,
                       exp_id=None, n_params=None, cycles=None,
                       role="champion", mutation=None, parent_exp=None,
                       checkpoint_path=None):
        """Log an experiment run. Append-only."""
        self.conn.execute(
            "INSERT INTO experiments (task, exp_id, round, accuracy, best_accuracy, "
            "config, n_params, cycles, role, mutation, parent_exp, timestamp, "
            "checkpoint_path) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (task, exp_id, round_num, accuracy, best_accuracy,
             json.dumps(config), n_params, cycles, role, mutation,
             parent_exp, time.time(), checkpoint_path)
        )
        self.conn.commit()

    def get_experiments(self, task=None):
        """Get experiments, optionally filtered by task."""
        if task:
            cur = self.conn.execute(
                "SELECT * FROM experiments WHERE task = ? ORDER BY round, id",
                (task,)
            )
        else:
            cur = self.conn.execute("SELECT * FROM experiments ORDER BY round, id")
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    # ── Runtime config (mutable, hot-reload) ───────────────────────

    def set_config(self, key, value):
        self.conn.execute(
            "INSERT OR REPLACE INTO runtime_config (key, value, updated_at) "
            "VALUES (?, ?, ?)",
            (key, json.dumps(value), time.time())
        )
        self.conn.commit()

    def get_config(self, key, default=None):
        cur = self.conn.execute(
            "SELECT value FROM runtime_config WHERE key = ?", (key,)
        )
        row = cur.fetchone()
        return json.loads(row[0]) if row else default

    def get_all_config(self):
        cur = self.conn.execute("SELECT key, value FROM runtime_config")
        return {k: json.loads(v) for k, v in cur.fetchall()}

    # ── Firebase sync ──────────────────────────────────────────────

    def sync_to_firebase(self):
        """Push full state to Firebase: teachers, lineage, model cards, knowledge flow."""
        try:
            import firebase_push as fb

            # Push teachers
            teachers = self.get_teachers()
            fb_teachers = {}
            for task, info in teachers.items():
                fb_teachers[task] = {
                    "accuracy": info["accuracy"],
                    "cycles": info["cycles"],
                    "config": info["config"],
                    "exp_id": info["exp_id"],
                    "graduated_at": info["graduated_at"],
                }
            fb._put("mamba3/state/teachers", fb_teachers)

            # Push lineage summary with model cards
            all_lineage = self.get_all_lineage()
            fb_lineage = {}
            for task, entries in all_lineage.items():
                card = self.build_model_card(task)
                champions = [e for e in entries if e.get("role") == "champion"]
                challengers = [e for e in entries if e.get("role") == "challenger"]
                wins = sum(1 for i, e in enumerate(entries)
                          if i > 0 and e["accuracy"] > entries[i-1].get("best_accuracy", 0))
                fb_lineage[task] = {
                    "rounds": len(entries),
                    "best": max((e["best_accuracy"] for e in entries), default=0),
                    "latest_config": entries[-1]["config"] if entries else {},
                    "latest_round": entries[-1]["round"] if entries else 0,
                    "n_champions": len(champions),
                    "n_challengers": len(challengers),
                    "n_improvements": wins,
                    "teachers": card.get("teachers", []),
                }
            fb._put("mamba3/state/lineage", fb_lineage)

            # Push teacher eval matrix (who can teach what)
            all_evals = self.get_all_teacher_scores()
            eval_matrix = {}
            for e in all_evals:
                teacher = e["teacher"]
                task = e["task"]
                if teacher not in eval_matrix:
                    eval_matrix[teacher] = {}
                eval_matrix[teacher][task] = round(e["accuracy"], 3)
            fb._put("mamba3/state/teacher_matrix", eval_matrix)

            # Push knowledge flow: which teachers are actively helping which tasks
            knowledge_flow = []
            for task, entries in all_lineage.items():
                for e in entries:
                    for t in e.get("teachers", []):
                        if t.get("model") and t.get("weight", 0) > 0.1:
                            knowledge_flow.append({
                                "from": t["model"],
                                "to": task,
                                "weight": t["weight"],
                                "round": e["round"],
                            })
            if knowledge_flow:
                fb._put("mamba3/state/knowledge_flow", knowledge_flow)

            return True
        except Exception:
            return False

    # ── Teacher evaluation cache (idempotent) ────────────────────

    def get_teacher_score(self, teacher, task):
        """Get cached teacher accuracy for a task. Returns None if not evaluated."""
        cur = self.conn.execute(
            "SELECT accuracy FROM teacher_eval_cache WHERE teacher = ? AND task = ?",
            (teacher, task)
        )
        row = cur.fetchone()
        return row[0] if row else None

    def set_teacher_score(self, teacher, task, accuracy, n_examples=30):
        """Cache teacher evaluation. Idempotent — only writes if not present."""
        existing = self.get_teacher_score(teacher, task)
        if existing is not None:
            return existing  # already cached
        self.conn.execute(
            "INSERT OR IGNORE INTO teacher_eval_cache "
            "(teacher, task, accuracy, n_examples, timestamp) VALUES (?, ?, ?, ?, ?)",
            (teacher, task, accuracy, n_examples, time.time())
        )
        self.conn.commit()
        return accuracy

    def get_best_teachers_for_task(self, task, min_accuracy=0.0):
        """Get all teachers that score above min_accuracy on a task, sorted best first."""
        cur = self.conn.execute(
            "SELECT teacher, accuracy FROM teacher_eval_cache "
            "WHERE task = ? AND accuracy > ? ORDER BY accuracy DESC",
            (task, min_accuracy)
        )
        return [(r[0], r[1]) for r in cur.fetchall()]

    def get_all_teacher_scores(self):
        """Get all cached evaluations."""
        cur = self.conn.execute(
            "SELECT teacher, task, accuracy FROM teacher_eval_cache ORDER BY task, accuracy DESC"
        )
        return [{"teacher": r[0], "task": r[1], "accuracy": r[2]} for r in cur.fetchall()]

    # ── Export ─────────────────────────────────────────────────────

    def export_lineage_markdown(self, task, path):
        """Export lineage as markdown alongside checkpoint."""
        lineage = self.get_lineage(task)
        if not lineage:
            return

        with open(path, "w") as f:
            f.write(f"# {task} — Training Lineage\n\n")
            f.write("| Round | Role | Acc | Best | d_model | Layers | LR | WD | "
                   "Optimizer | Loss | Mutation |\n")
            f.write("|-------|------|-----|------|---------|--------|----|----|"
                   "-----------|------|-----------|\n")
            for e in lineage:
                c = e["config"]
                mut = e.get("mutation") or "—"
                role = e.get("role", "champion")[0].upper()
                f.write(f"| {e['round']} | {role} | {e['accuracy']:.0%} | "
                       f"{e['best_accuracy']:.0%} | {c.get('d_model', 64)} | "
                       f"{c.get('n_kernel_layers', 3)} | "
                       f"{c.get('lr', 1e-3):.0e} | "
                       f"{c.get('weight_decay', 0.1)} | "
                       f"{c.get('optimizer', 'adamw')} | "
                       f"{c.get('loss_fn', 'ce')} | {mut} |\n")

    def export_checkpoint_metadata(self, task, checkpoint_dir):
        """Write lineage + config alongside a checkpoint file."""
        meta_path = Path(checkpoint_dir) / f"{task}_meta.json"
        lineage = self.get_lineage(task)
        best_cfg, best_acc = self.get_best_config(task)
        is_graduated = self.is_teacher(task)

        meta = {
            "task": task,
            "best_accuracy": best_acc,
            "best_config": best_cfg,
            "is_teacher": is_graduated,
            "total_rounds": len(lineage),
            "lineage": lineage,
            "exported_at": time.time(),
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

    def close(self):
        self.conn.close()
