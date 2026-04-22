"""
Coordinator: manages a population of worker processes.

MAP:    N worker processes, each training independently
REDUCE: This process reads metrics, ranks, evolves population

No killing — losers get paused when resources are scarce.
Winners reproduce: their configs get mutated into new experiments.
Losers get another chance with configs inherited from winners.

Genetic competition:
  - Rank all workers by fresh accuracy
  - Top workers: continue + reproduce (mutate config → spawn child)
  - Bottom workers: pause, replace with child of a winner
  - Middle workers: continue (chance to improve)

Usage:
    python coordinator.py                    # auto-detect resources
    python coordinator.py --max-workers 12   # cap parallel workers
    python coordinator.py --generation-every 30  # evolve every 30s
"""
import os
os.environ["PYTHONUNBUFFERED"] = "1"
import sys
sys.path.insert(0, os.path.dirname(__file__))

import argparse
import json
import time
import math
import random
import signal
import subprocess
from pathlib import Path
from datetime import datetime
from collections import defaultdict


# ── Evolution state ─────────────────────────────────────────────────

class EvolutionState:
    """Tracks momentum, lineage, plateau detection, and generation."""
    def __init__(self):
        self.history = {}      # exp_id → [(cycle, best_fresh), ...]
        self.lineage = {}      # exp_id → {"parent": exp_id, "grandparent": exp_id}
        self.generation = 0
        self.best_ever = 0.0   # population best for plateau detection
        self.best_ever_gen = 0 # generation when best was set
        self.plateau_mode = False

    def record(self, exp_id, cycle, best_fresh):
        if exp_id not in self.history:
            self.history[exp_id] = []
        self.history[exp_id].append((cycle, best_fresh))

    def get_momentum(self, exp_id, window=5):
        h = self.history.get(exp_id, [])
        if len(h) < window:
            return 0.0
        recent = h[-1][1]
        old = h[-window][1]
        return (recent - old) / window

    def register_lineage(self, child_id, parent_id):
        parent_info = self.lineage.get(parent_id, {"parent": None})
        self.lineage[child_id] = {
            "parent": parent_id,
            "grandparent": parent_info.get("parent"),
        }

    def get_lineage_root(self, exp_id, depth=3):
        """Walk up the lineage tree to find ancestor."""
        current = exp_id
        for _ in range(depth):
            info = self.lineage.get(current, {})
            parent = info.get("parent")
            if parent is None:
                return current
            current = parent
        return current

    def check_plateau(self, current_best, patience=10):
        """Detect if the population is stuck. Uses trajectory, not just patience counter."""
        if current_best > self.best_ever + 0.005:  # 0.5% improvement threshold
            self.best_ever = current_best
            self.best_ever_gen = self.generation
            self.plateau_mode = False
            return False

        stuck_gens = self.generation - self.best_ever_gen
        was_plateau = self.plateau_mode
        self.plateau_mode = stuck_gens >= patience

        if self.plateau_mode and not was_plateau:
            print(f"\n  🔥 PLATEAU DETECTED — no improvement for {stuck_gens} generations!"
                  f"\n  🔥 Activating aggressive exploration mode.\n", flush=True)

        return self.plateau_mode

    def bootstrap_from_db(self, db_path="metrics.db"):
        """Load historical best from SQLite so plateau detection kicks in immediately."""
        try:
            import sqlite3
            db = sqlite3.connect(db_path)
            row = db.execute("SELECT MAX(best_fresh) FROM cycles").fetchone()
            if row and row[0]:
                self.best_ever = row[0]
                self.best_ever_gen = -20  # far in the past → immediately triggers plateau
                self.plateau_mode = True  # activate right now
                print(f"  📊 Bootstrapped from DB: best_ever={self.best_ever:.1%}, "
                      f"PLATEAU MODE ACTIVE immediately", flush=True)
            db.close()
        except Exception as e:
            print(f"  ⚠ DB bootstrap failed: {e}", flush=True)

    def get_plateau_severity(self):
        """How stuck are we? 0 = not stuck, 1+ = very stuck."""
        if not self.plateau_mode:
            return 0.0
        stuck = self.generation - self.best_ever_gen
        return min(stuck / 10.0, 3.0)  # caps at 3.0


# ── Scoring: momentum + task specialists ────────────────────────────

def score_experiments(results, evo_state, momentum_weight=0.5):
    """Score experiments with momentum bonus and tag task specialists."""
    # Record history + compute momentum
    for r in results:
        eid = r["exp_id"]
        cycle = r.get("cycle", 0)
        fresh = r.get("best_fresh", 0)
        evo_state.record(eid, cycle, fresh)
        momentum = evo_state.get_momentum(eid)
        r["momentum"] = momentum
        r["effective_score"] = fresh + momentum_weight * max(momentum, 0)

    # Tag task specialists
    task_best = {}  # task → (exp_id, accuracy)
    for r in results:
        for task, acc in r.get("type_accs", {}).items():
            if task not in task_best or acc > task_best[task][1]:
                task_best[task] = (r["exp_id"], acc)

    specialist_map = {}  # exp_id → [tasks]
    for task, (eid, acc) in task_best.items():
        if acc > 0:
            specialist_map.setdefault(eid, []).append(task)

    for r in results:
        r["specialist_for"] = specialist_map.get(r["exp_id"], [])

    # Sort by effective score
    results.sort(key=lambda x: -x["effective_score"])
    return results


# ── Parent selection: temperature + focal + lineage dropout ─────────

def select_parent(running_results, evo_state, generation,
                  specialist_prob=0.3, max_temp=5.0, decay_constant=50.0):
    """Select a parent using temperature-annealed softmax + focal attention."""
    if not running_results:
        return running_results[0] if running_results else None, "default"

    # Focal task attention: 30% chance to breed from a task specialist
    if random.random() < specialist_prob:
        specialists = [r for r in running_results if r.get("specialist_for")]
        if specialists:
            chosen = random.choice(specialists)
            task = random.choice(chosen["specialist_for"])
            return chosen, f"specialist:{task}"

    # Temperature-annealed softmax selection
    temperature = max(0.1, max_temp * math.exp(-generation / decay_constant))
    scores = [r.get("effective_score", r.get("best_fresh", 0)) for r in running_results]

    # Softmax with temperature
    max_score = max(scores) if scores else 0
    weights = [math.exp((s - max_score) / temperature) for s in scores]
    total = sum(weights)
    if total == 0:
        return running_results[0], "fallback"
    weights = [w / total for w in weights]

    chosen = random.choices(running_results, weights=weights, k=1)[0]

    # Lineage dropout: if >50% share same ancestor, force diversity
    if len(running_results) >= 6:
        chosen_root = evo_state.get_lineage_root(chosen["exp_id"])
        same_lineage = sum(
            1 for r in running_results
            if evo_state.get_lineage_root(r["exp_id"]) == chosen_root
        )
        if same_lineage > len(running_results) * 0.5:
            # Pick from a different lineage
            others = [r for r in running_results
                     if evo_state.get_lineage_root(r["exp_id"]) != chosen_root]
            if others:
                chosen = random.choice(others)
                return chosen, f"lineage_dropout(was:{chosen_root})"

    return chosen, f"temp={temperature:.1f}"


# ── Genetic config mutation ─────────────────────────────────────────

def mutate_config(parent_config, mutation_strength=0.3, plateau_severity=0.0):
    """Create a child config by mutating a parent.

    When plateau_severity > 0, mutations become more aggressive:
    - Higher probability of changing every parameter
    - Wider ranges for numerical mutations
    - Re-introduces discarded strategies (Lion, focal, SAM, etc.)
    """
    child = parent_config.copy()
    # Ensure all keys exist with defaults
    child.setdefault("loss_fn", "stable_ce")
    child.setdefault("backend", "pytorch")
    child.setdefault("steps_per_cycle", 200)

    # Plateau amplifier: multiply mutation probabilities
    amp = 1.0 + plateau_severity  # 1.0 normal, 2.0-4.0 when stuck

    # Mutate learning rate (log-scale)
    if random.random() < min(0.5 * amp, 0.95):
        spread = mutation_strength * amp
        factor = 2 ** (random.gauss(0, spread))
        child["lr"] = max(1e-5, min(1e-2, child["lr"] * factor))

    # Mutate batch size
    if random.random() < min(0.3 * amp, 0.9):
        child["batch_size"] = random.choice([64, 128, 256, 512, 1024])

    # Mutate d_model
    if random.random() < min(0.2 * amp, 0.8):
        # Dynamic: explore around parent's width
        parent_d = child.get("d_model", 64)
        nearby_d = [max(16, parent_d - 16), parent_d, parent_d + 16, parent_d + 32]
        wild_d = [32, 48, 64, 96, 128, 192, 256]
        choices_d = nearby_d if random.random() < 0.7 else wild_d
        child["d_model"] = random.choice(choices_d)
        child["headdim"] = min(child["headdim"], child["d_model"])

    # Mutate layers
    if random.random() < min(0.2 * amp, 0.8):
        # Dynamic: explore around the parent's depth ± 1-2, plus random
        parent_layers = child.get("n_kernel_layers", 1)
        nearby = [max(1, parent_layers - 1), parent_layers, parent_layers + 1, parent_layers + 2]
        wild = [1, 2, 3, 4, 6, 8]  # occasional wild jump
        choices = nearby if random.random() < 0.7 else wild
        child["n_kernel_layers"] = random.choice(choices)

    # Mutate weight decay
    if random.random() < min(0.2 * amp, 0.8):
        child["weight_decay"] = random.choice([0.0, 0.01, 0.05, 0.1, 0.2, 0.5])

    # Mutate d_state
    if random.random() < min(0.15 * amp, 0.7):
        child["d_state"] = random.choice([8, 16, 32])

    # Flip backend
    if random.random() < min(0.1 * amp, 0.5):
        child["backend"] = "tinygrad" if child.get("backend") != "tinygrad" else "pytorch"

    # Mutate loss function — when stuck, try everything
    if random.random() < min(0.15 * amp, 0.8):
        child["loss_fn"] = random.choice(["stable_ce", "ce", "focal", "label_smooth"])

    # Mutate optimizer — when stuck, try Lion
    if random.random() < min(0.1 * amp, 0.6):
        child["optimizer"] = random.choice(["adamw", "lion"])

    # Mutate warm restarts — when stuck, shake things up
    if random.random() < min(0.1 * amp, 0.5):
        child["warm_restarts"] = not child.get("warm_restarts", False)

    # Mutate noise injection — when stuck, add noise
    if random.random() < min(0.1 * amp, 0.5):
        child["noise_scale"] = random.choice([0.0, 0.0005, 0.001, 0.002, 0.005])

    # Mutate PerpGrad
    if random.random() < min(0.1 * amp, 0.5):
        child["use_perp"] = not child.get("use_perp", child.get("weight_decay", 0) == 0)

    # When severely stuck: radical mutation — completely random config
    if plateau_severity >= 2.0 and random.random() < 0.3:
        child = random.choice(SEED_CONFIGS).copy()
        # Dynamic: explore around the parent's depth ± 1-2, plus random
        parent_layers = child.get("n_kernel_layers", 1)
        nearby = [max(1, parent_layers - 1), parent_layers, parent_layers + 1, parent_layers + 2]
        wild = [1, 2, 3, 4, 6, 8]  # occasional wild jump
        choices = nearby if random.random() < 0.7 else wild
        child["n_kernel_layers"] = random.choice(choices)
        child["loss_fn"] = random.choice(["stable_ce", "ce", "focal", "label_smooth"])
        child["optimizer"] = random.choice(["adamw", "lion"])
        child["warm_restarts"] = random.choice([True, False])
        child["noise_scale"] = random.choice([0.0, 0.001, 0.005])

    return child


# ── Default seed config: proven to graduate 5 tasks in 3 rounds ────

BASE_CONFIG = {
    "d_model": 64, "d_state": 16, "headdim": 16, "n_kernel_layers": 3,
    "batch_size": 256, "lr": 1e-3, "weight_decay": 0.1,
    "steps_per_cycle": 200, "loss_fn": "ce", "optimizer": "adamw",
    "noise_scale": 0.0, "use_perp": False,
}

SEED_CONFIGS = [BASE_CONFIG]  # All tasks start from the proven config


# ── Mutation history: learn what works ─────────────────────────────

class MutationHistory:
    """Tracks which config changes helped vs hurt, to inform future mutations."""

    def __init__(self):
        self.outcomes = defaultdict(list)  # param_name → [{parent_val, child_val, improvement}]

    def record(self, parent_config, child_config, parent_acc, child_acc):
        improvement = child_acc - parent_acc
        for key in ["d_model", "n_kernel_layers", "lr", "weight_decay",
                    "batch_size", "loss_fn", "optimizer"]:
            pv = parent_config.get(key)
            cv = child_config.get(key)
            if pv != cv:
                self.outcomes[key].append({
                    "parent_val": pv, "child_val": cv,
                    "improvement": improvement,
                })

    def get_bias(self, param_name):
        """Return (direction, confidence) based on historical outcomes.

        direction: +1 increase helped, -1 decrease helped, 0 unclear
        confidence: 0.0-1.0 based on sample count
        """
        history = self.outcomes.get(param_name, [])
        if len(history) < 3:
            return 0, 0.0

        # For numeric params: check if increases or decreases helped more
        numeric = [h for h in history
                   if isinstance(h["parent_val"], (int, float))
                   and isinstance(h["child_val"], (int, float))]
        if not numeric:
            return 0, min(len(history) / 10, 1.0)

        increases = [h for h in numeric if h["child_val"] > h["parent_val"]]
        decreases = [h for h in numeric if h["child_val"] < h["parent_val"]]

        inc_avg = sum(h["improvement"] for h in increases) / max(len(increases), 1)
        dec_avg = sum(h["improvement"] for h in decreases) / max(len(decreases), 1)

        confidence = min(len(history) / 10, 1.0)
        if inc_avg > dec_avg + 0.01:
            return +1, confidence
        elif dec_avg > inc_avg + 0.01:
            return -1, confidence
        return 0, confidence


# ── Smart mutation: informed, conservative, cost-aware ─────────────

def _mutate_one_param(child, param, direction, confidence, plateau_severity):
    """Mutate one parameter, biased by history."""
    noise = random.gauss(0, 0.3)

    if param == "lr":
        # Log-scale: bias by direction, always explore a little
        bias = direction * 0.3 * confidence
        factor = 2 ** (bias + noise * 0.5)
        child["lr"] = max(1e-5, min(1e-2, child["lr"] * factor))

    elif param == "weight_decay":
        current = child.get("weight_decay", 0.1)
        step = 0.05 if direction == 0 else (0.05 * direction)
        child["weight_decay"] = max(0.0, min(0.5, current + step + noise * 0.02))

    elif param == "noise_scale":
        options = [0.0, 0.0005, 0.001, 0.002, 0.005]
        current = child.get("noise_scale", 0.0)
        idx = min(range(len(options)), key=lambda i: abs(options[i] - current))
        if direction > 0 and idx < len(options) - 1:
            idx += 1
        elif direction < 0 and idx > 0:
            idx -= 1
        else:
            idx = max(0, min(len(options) - 1, idx + random.choice([-1, 0, 1])))
        child["noise_scale"] = options[idx]

    elif param == "n_kernel_layers":
        current = child.get("n_kernel_layers", 3)
        if confidence > 0.5 and direction != 0:
            child["n_kernel_layers"] = max(1, current + direction)
        else:
            child["n_kernel_layers"] = max(1, current + random.choice([-1, 0, 1]))

    elif param == "d_model":
        current = child.get("d_model", 64)
        if confidence > 0.5 and direction != 0:
            child["d_model"] = max(32, min(256, current + direction * 16))
        else:
            child["d_model"] = max(32, min(256, current + random.choice([-16, 0, 16])))
        child["headdim"] = min(child.get("headdim", 16), child["d_model"])

    elif param == "d_state":
        child["d_state"] = random.choice([8, 16, 32])

    elif param == "batch_size":
        current = child.get("batch_size", 256)
        child["batch_size"] = random.choice([
            max(64, current // 2), current, min(1024, current * 2)])

    elif param == "loss_fn":
        child["loss_fn"] = random.choice(["ce", "stable_ce", "focal", "label_smooth"])

    elif param == "optimizer":
        child["optimizer"] = random.choice(["adamw", "lion"])


def smart_mutate_config(parent_config, mutation_history=None, plateau_severity=0.0):
    """Create a child config with informed, conservative mutations.

    Key principles:
    - Mutate 1 param normally, 2-3 when stuck
    - Cheap params (lr, wd) first, architecture only when stuck
    - Biased by historical outcomes when confident
    """
    child = parent_config.copy()
    child.pop("task", None)  # remove task key if present

    # How many params to mutate
    if plateau_severity < 1.0:
        n_mutations = 1
    elif plateau_severity < 2.0:
        n_mutations = 2
    else:
        n_mutations = 3

    # Categorize by cost of change
    cheap = ["lr", "weight_decay", "noise_scale"]
    medium = ["batch_size", "loss_fn", "optimizer"]
    expensive = ["d_model", "n_kernel_layers", "d_state"]

    # Unlock categories based on plateau severity
    if plateau_severity < 1.0:
        candidates = cheap
    elif plateau_severity < 2.0:
        candidates = cheap + medium
    else:
        candidates = cheap + medium + expensive

    params = random.sample(candidates, min(n_mutations, len(candidates)))

    for param in params:
        direction, confidence = (0, 0.0)
        if mutation_history:
            direction, confidence = mutation_history.get_bias(param)
        _mutate_one_param(child, param, direction, confidence, plateau_severity)

    return child


# ── Worker management ───────────────────────────────────────────────

class WorkerManager:
    def __init__(self, runs_dir="runs"):
        self.runs_dir = Path(runs_dir)
        self.runs_dir.mkdir(exist_ok=True)
        self.processes = {}  # exp_id → Popen
        self.next_id = 0

    def _find_next_id(self):
        existing = [int(p.name.split("_")[1]) for p in self.runs_dir.iterdir()
                    if p.is_dir() and p.name.startswith("exp_")]
        self.next_id = max(existing, default=-1) + 1

    def spawn(self, config, parent_id=None):
        """Launch a new worker process. Optionally inherit weights from parent."""
        self._find_next_id()
        exp_id = f"exp_{self.next_id:03d}"
        run_dir = self.runs_dir / exp_id
        run_dir.mkdir(exist_ok=True)

        config_path = run_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        # Safe weight inheritance: copy parent checkpoint if architecturally compatible
        if parent_id:
            parent_dir = self.runs_dir / parent_id
            parent_ckpt = parent_dir / "checkpoint.pt"
            parent_cfg_path = parent_dir / "config.json"
            if parent_ckpt.exists() and parent_cfg_path.exists():
                try:
                    parent_cfg = json.load(open(parent_cfg_path))
                    # Only inherit if architecture matches
                    arch_keys = ["d_model", "d_state", "headdim", "n_kernel_layers"]
                    compatible = all(
                        config.get(k) == parent_cfg.get(k) for k in arch_keys
                    )
                    if compatible and config.get("backend") == parent_cfg.get("backend", "pytorch"):
                        import shutil
                        shutil.copy2(parent_ckpt, run_dir / "checkpoint.pt")
                        print(f"    📋 Inherited weights from {parent_id}", flush=True)
                    else:
                        print(f"    🆕 Fresh start (arch differs from {parent_id})", flush=True)
                except Exception as e:
                    print(f"    ⚠ Weight inheritance failed: {e}", flush=True)

        # Choose worker script based on backend
        backend = config.get("backend", "pytorch")
        worker_script = "worker_tinygrad.py" if backend == "tinygrad" else "worker.py"

        proc = subprocess.Popen(
            [sys.executable, "-u", worker_script,
             "--run-dir", str(run_dir),
             "--config", str(config_path)],
            stdout=open(run_dir / "stdout.log", "w"),
            stderr=subprocess.STDOUT,
        )
        self.processes[exp_id] = proc
        (run_dir / "status").write_text("running")
        print(f"  + Spawned {exp_id}: d={config['d_model']} L={config['n_kernel_layers']} "
              f"bs={config['batch_size']} lr={config['lr']:.0e} "
              f"wd={config['weight_decay']}", flush=True)
        return exp_id

    def pause(self, exp_id):
        """Gracefully pause a worker (SIGTERM → it saves checkpoint)."""
        if exp_id in self.processes:
            proc = self.processes[exp_id]
            if proc.poll() is None:  # still running
                proc.terminate()
                proc.wait(timeout=30)
            del self.processes[exp_id]
            status_file = self.runs_dir / exp_id / "status"
            if status_file.exists():
                status_file.write_text("paused")
            print(f"  ⏸ Paused {exp_id}", flush=True)

    def resume(self, exp_id):
        """Resume a paused worker from its checkpoint."""
        run_dir = self.runs_dir / exp_id
        config_path = run_dir / "config.json"
        if not config_path.exists():
            return

        proc = subprocess.Popen(
            [sys.executable, "-u", "worker.py",
             "--run-dir", str(run_dir),
             "--config", str(config_path)],
            stdout=open(run_dir / "stdout.log", "a"),
            stderr=subprocess.STDOUT,
        )
        self.processes[exp_id] = proc
        (run_dir / "status").write_text("running")
        print(f"  ▶ Resumed {exp_id}", flush=True)

    def get_running(self):
        """Return list of running experiment IDs."""
        alive = []
        for exp_id, proc in list(self.processes.items()):
            if proc.poll() is None:
                alive.append(exp_id)
            else:
                # Process exited
                del self.processes[exp_id]
        return alive

    def read_metrics(self, exp_id):
        """Read latest metrics from worker."""
        metrics_path = self.runs_dir / exp_id / "metrics.json"
        if metrics_path.exists():
            try:
                with open(metrics_path) as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return None
        return None

    def get_all_metrics(self):
        """Read metrics from all experiments (running + paused)."""
        results = []
        for p in sorted(self.runs_dir.iterdir()):
            if not p.is_dir() or not p.name.startswith("exp_"):
                continue
            exp_id = p.name
            metrics = self.read_metrics(exp_id)
            status = (p / "status").read_text().strip() if (p / "status").exists() else "unknown"
            if metrics:
                results.append({
                    "exp_id": exp_id,
                    "status": status,
                    **metrics,
                })
        # Sort by best_fresh descending
        results.sort(key=lambda x: -x.get("best_fresh", 0))
        return results


# ── Dashboard writer ────────────────────────────────────────────────

def write_dashboard(results, generation, gpu_pct, mem_pct, evo_state=None):
    """Write dashboard.html + dashboard.md from coordinator's view."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Collect chart data from all experiments
    colors = ["#0066CC", "#CC3300", "#339933", "#CC6600", "#6633CC",
              "#CC3399", "#336699", "#996633", "#009999", "#993366",
              "#666600", "#003366"]

    # Discover all tasks that have non-zero accuracy
    all_tasks = set()
    for r in results:
        for t, a in r.get("type_accs", {}).items():
            if a > 0:
                all_tasks.add(t)
    all_tasks = sorted(all_tasks)
    chart_keys = ["fresh"] + all_tasks

    TASK_DESC = {
        "fresh": "Overall Generalization",
        "parity": "Stage 0: Parity",
        "binary_pattern_next": "Stage 0: Binary Patterns",
        "same_different": "Stage 1: Same/Different",
        "odd_one_out": "Stage 1: Odd One Out",
        "sequence_completion": "Stage 2: Sequence Completion",
        "pattern_period": "Stage 2: Pattern Period",
        "run_length_next": "Stage 2: Run Length",
        "mirror_detection": "Stage 3: Mirror Detection",
        "repeat_count": "Stage 3: Repeat Count",
        "arithmetic_next": "Stage 4: Arithmetic",
        "geometric_next": "Stage 4: Geometric",
        "alternating_next": "Stage 4: Alternating",
        "logic_gate": "Stage 5: Logic Gates",
        "logic_chain": "Stage 5: Logic Chains",
        "modus_ponens": "Stage 5: Modus Ponens",
    }

    # Build chart datasets — one line per experiment per chart
    import json
    all_datasets = {}
    for key in chart_keys:
        datasets = []
        for i, r in enumerate(results):
            cfg = r.get("config", {})
            wd = cfg.get("weight_decay", 0)
            method = f"wd={wd}" if wd > 0 else "perp"
            color = colors[i % len(colors)]
            alive = r["status"] == "running"
            # Single data point per experiment (current value)
            val = r.get("type_accs", {}).get(key, 0) * 100 if key != "fresh" else r.get("best_fresh", 0) * 100
            ds = {
                "label": f"{r['exp_id']} d={cfg.get('d_model','?')} [{method}]",
                "data": [{"x": r.get("cycle", 0), "y": round(val, 1)}],
                "borderColor": color,
                "backgroundColor": color,
                "pointRadius": 6,
                "pointStyle": "circle" if alive else "crossRot",
            }
            datasets.append(ds)
        all_datasets[key] = datasets
    datasets_json = json.dumps(all_datasets)

    # Leaderboard HTML
    leaderboard = ""
    for i, r in enumerate(results):
        cfg = r.get("config", {})
        wd = cfg.get("weight_decay", 0)
        method = f"wd={wd}" if wd > 0 else "PerpGrad"
        parity = r.get("type_accs", {}).get("parity", 0)
        status = r["status"]
        sc = "text-green-600" if status == "running" else "text-yellow-600" if status == "paused" else "text-gray-400"
        star = "★" if i == 0 else ""
        leaderboard += f"""<tr class="border-b border-gray-100">
            <td class="py-2 pr-3 font-mono text-sm">{star} {r['exp_id']}</td>
            <td class="py-2 pr-3 text-right">{r.get('params', 0):,}</td>
            <td class="py-2 pr-3">d={cfg.get('d_model','')} L={cfg.get('n_kernel_layers','')}</td>
            <td class="py-2 pr-3">{method}</td>
            <td class="py-2 pr-3 text-right font-bold">{r.get('best_fresh',0):.1%}</td>
            <td class="py-2 pr-3 text-right">{parity:.0%}</td>
            <td class="py-2 pr-3 text-right">{r.get('cycle',0)}</td>
            <td class="py-2 pr-3 {sc}">{status}</td>
        </tr>"""

    # Charts HTML
    charts_html = ""
    for key in chart_keys:
        title = TASK_DESC.get(key, key)
        charts_html += f"""<div>
            <h3 class="text-sm font-bold mb-1 border-b border-gray-200 pb-1">{title}</h3>
            <div class="chart-container"><canvas id="chart_{key}"></canvas></div>
        </div>\n"""

    # Teacher + learning report
    teacher_html = ""
    learning_html = ""
    if results:
        teacher_html = results[0].get("teacher_status", "").replace("\n", "<br>")
        ml = results[0].get("mastery_log", [])
        if ml:
            learning_html = "<br>".join(
                f"Task {e['task_index']+1}: {e['task']} — {e['steps_to_master']} steps"
                for e in ml
            )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<meta http-equiv="refresh" content="30">
<title>Mamba-3 Evolution Dashboard</title>
<script src="https://cdn.tailwindcss.com"></script>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>body {{ font-family: 'Georgia', serif; }} .chart-container {{ height: 200px; }}</style>
</head>
<body class="bg-white text-gray-900 max-w-6xl mx-auto px-8 py-6">

<header class="border-b-2 border-black pb-2 mb-6">
  <h1 class="text-3xl font-bold tracking-tight">Mamba-3 Evolution</h1>
  <p class="text-gray-500 text-sm mt-1">Generation {generation} · {now} · GPU {gpu_pct:.0f}% · VRAM {mem_pct:.0f}% · {len([r for r in results if r['status']=='running'])} workers · Auto-refreshes 30s</p>
</header>

<h2 class="text-lg font-bold mb-2 border-b border-gray-200 pb-1">Leaderboard</h2>
<table class="w-full mb-6 text-sm">
<thead><tr class="border-b-2 border-gray-300 text-left text-gray-500">
  <th class="py-1 pr-3">ID</th><th class="py-1 pr-3 text-right">Params</th>
  <th class="py-1 pr-3">Arch</th><th class="py-1 pr-3">Method</th>
  <th class="py-1 pr-3 text-right">Best Fresh</th><th class="py-1 pr-3 text-right">Parity</th>
  <th class="py-1 pr-3 text-right">Cycles</th><th class="py-1 pr-3">Status</th>
</tr></thead>
<tbody>{leaderboard}</tbody>
</table>

<div class="grid grid-cols-2 gap-6 mb-6">
{charts_html}
</div>

<div class="grid grid-cols-2 gap-6 mb-6">
<div>
  <h2 class="text-lg font-bold mb-2 border-b border-gray-200 pb-1">Curriculum Teacher</h2>
  <div class="bg-gray-50 p-3 rounded font-mono text-xs leading-relaxed">{teacher_html}</div>
</div>
<div>
  <h2 class="text-lg font-bold mb-2 border-b border-gray-200 pb-1">Learning to Learn</h2>
  <div class="bg-blue-50 p-3 rounded font-mono text-xs">{learning_html if learning_html else 'No tasks mastered yet'}</div>
</div>
</div>

<script>
const allData = {datasets_json};
const chartOpts = {{
  responsive: true, maintainAspectRatio: false,
  plugins: {{ legend: {{ position: 'bottom', labels: {{ font: {{ size: 9 }} }} }} }},
  scales: {{ x: {{ type: 'linear', title: {{ display: true, text: 'Cycle' }} }}, y: {{ min: 0, max: 100 }} }}
}};
for (const [key, datasets] of Object.entries(allData)) {{
  const el = document.getElementById('chart_' + key);
  if (el) new Chart(el, {{ type: 'scatter', data: {{ datasets }}, options: chartOpts }});
}}
</script>

<footer class="text-gray-400 text-xs mt-6 border-t pt-2">
  Mamba-3 · GauchoAI · Evolutionary Model Shopping · {len(results)} experiments total
</footer>
</body></html>"""

    Path("dashboard.html").write_text(html)

    # Markdown version
    plateau_md = ""
    if evo_state:
        if evo_state.plateau_mode:
            sev = evo_state.get_plateau_severity()
            stuck = evo_state.generation - evo_state.best_ever_gen
            plateau_md = (f"\n## 🔥 PLATEAU MODE ACTIVE\n"
                         f"- Severity: **{sev:.1f}** (mutation amplifier: {1+sev:.1f}x)\n"
                         f"- Stuck for **{stuck} generations** at {evo_state.best_ever:.1%}\n"
                         f"- Aggressive exploration: re-trying Lion, focal loss, SAM, warm restarts, noise\n"
                         f"- Radical mutation: 30% chance of completely random configs\n")
        else:
            plateau_md = f"\n## ✅ Improving — best ever: {evo_state.best_ever:.1%}\n"

    lines = [f"# Evolution Dashboard — Gen {generation}",
             f"GPU {gpu_pct:.0f}% · VRAM {mem_pct:.0f}%",
             plateau_md, ""]
    lines.append("| Rank | ID | Params | Arch | Method | Fresh | Parity | Cycles | Status |")
    lines.append("|------|-----|--------|------|--------|-------|--------|--------|--------|")
    for i, r in enumerate(results):
        cfg = r.get("config", {})
        wd = cfg.get("weight_decay", 0)
        method = f"wd={wd}" if wd > 0 else "PerpGrad"
        parity = r.get("type_accs", {}).get("parity", 0)
        star = "★" if i == 0 else ""
        lines.append(f"| {i+1} | {star}{r['exp_id']} | {r.get('params',0):,} | "
                    f"d={cfg.get('d_model','')}L={cfg.get('n_kernel_layers','')} | {method} | "
                    f"{r.get('best_fresh',0):.1%} | {parity:.0%} | {r.get('cycle',0)} | {r['status']} |")
    if results and results[0].get("teacher_status"):
        lines.extend(["", "## Teacher", f"```\n{results[0]['teacher_status']}\n```"])
    Path("dashboard.md").write_text("\n".join(lines))


# ── Main coordinator loop ───────────────────────────────────────────

def get_gpu_usage():
    """Get GPU utilization % and memory usage %."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            text=True
        ).strip()
        gpu_pct, mem_used, mem_total = [float(x) for x in out.split(",")]
        mem_pct = mem_used / mem_total * 100
        return gpu_pct, mem_pct
    except Exception:
        return 0.0, 0.0


def get_system_resources():
    """Get CPU%, RAM%, GPU%, VRAM% — no psutil needed. Linux /proc only."""
    gpu_pct, vram_pct = get_gpu_usage()

    # CPU utilization from /proc/stat (instantaneous delta)
    cpu_pct = 0.0
    try:
        with open('/proc/stat') as f:
            fields = f.readline().split()[1:]  # cpu user nice system idle ...
        idle = int(fields[3]) + int(fields[4])  # idle + iowait
        total = sum(int(x) for x in fields)
        # Store previous reading for delta
        prev = getattr(get_system_resources, '_prev_cpu', None)
        get_system_resources._prev_cpu = (idle, total)
        if prev:
            d_idle = idle - prev[0]
            d_total = total - prev[1]
            cpu_pct = (1 - d_idle / max(d_total, 1)) * 100
    except FileNotFoundError:
        # macOS / non-Linux fallback
        try:
            load1 = os.getloadavg()[0]
            cpu_pct = min(load1 / os.cpu_count() * 100, 100)
        except Exception:
            pass

    # RAM from /proc/meminfo
    ram_pct = 0.0
    try:
        info = {}
        with open('/proc/meminfo') as f:
            for line in f:
                parts = line.split()
                info[parts[0].rstrip(':')] = int(parts[1])
        ram_pct = (1 - info.get('MemAvailable', 0) / max(info.get('MemTotal', 1), 1)) * 100
    except FileNotFoundError:
        pass  # macOS — graceful degradation

    return cpu_pct, ram_pct, gpu_pct, vram_pct


def get_actual_worker_count():
    """Count actual worker processes via pgrep, not coordinator state."""
    try:
        out = subprocess.check_output(["pgrep", "-fc", "worker.py"], text=True).strip()
        return int(out)
    except Exception:
        return 0


def run_coordinator(args):
    print(f"{'='*60}", flush=True)
    print(f"COORDINATOR — Evolutionary Model Shopping", flush=True)
    print(f"  max_workers={'auto (fill to 95%)' if args.max_workers == 0 else args.max_workers}", flush=True)
    print(f"  generation_every={args.generation_every}s", flush=True)
    print(f"{'='*60}\n", flush=True)

    from metrics_db import MetricsWriter
    metrics = MetricsWriter()

    evo_state = EvolutionState()
    evo_state.bootstrap_from_db()  # know history immediately, don't wait 10 mins

    mgr = WorkerManager(runs_dir=args.runs_dir)

    # Auto-detect max workers if not set
    max_workers = args.max_workers
    if max_workers == 0:
        # Start with seed configs, scale up based on GPU
        max_workers = len(SEED_CONFIGS)
        print(f"  Auto: starting with {max_workers} workers, will scale based on GPU usage",
              flush=True)

    # Spawn initial population from seed configs
    for cfg in SEED_CONFIGS[:max_workers]:
        mgr.spawn(cfg)

    generation = 0

    # Graceful shutdown
    should_stop = False
    def handle_signal(sig, frame):
        nonlocal should_stop
        should_stop = True
        print("\nCoordinator shutting down...", flush=True)
    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    while not should_stop:
        time.sleep(args.generation_every)
        generation += 1

        try:
            _run_generation(mgr, metrics, args, generation,
                           max_workers, evo_state)
        except Exception as e:
            # LOUD error — never eat exceptions silently
            import traceback
            err_msg = traceback.format_exc()
            print(f"\n{'!'*60}", flush=True)
            print(f"  ERROR in generation {generation}: {e}", flush=True)
            print(err_msg, flush=True)
            print(f"{'!'*60}\n", flush=True)
            # Log to events DB if available
            try:
                metrics.log_event("error", None, f"Gen {generation}: {e}")
            except Exception:
                pass
            # Continue — don't crash the coordinator

    # Shutdown all workers
    print("\nStopping all workers...", flush=True)
    for exp_id in list(mgr.processes.keys()):
        mgr.pause(exp_id)

    results = mgr.get_all_metrics()
    print(f"\n{'='*60}", flush=True)
    print(f"FINAL LEADERBOARD", flush=True)
    print(f"{'='*60}", flush=True)
    for i, r in enumerate(results[:10]):
        cfg = r.get("config", {})
        print(f"  {i+1}. {r['exp_id']}: fresh={r.get('best_fresh', 0):.1%}  "
              f"d={cfg.get('d_model')} L={cfg.get('n_kernel_layers')} "
              f"wd={cfg.get('weight_decay')}  cycles={r.get('cycle', 0)}",
              flush=True)


def _enforce_disk_budget(runs_dir, results, max_gb=50):
    """Evict paused experiments' checkpoints if disk usage exceeds budget."""
    import shutil
    total_bytes = sum(
        f.stat().st_size for f in runs_dir.rglob("*") if f.is_file()
    )
    total_gb = total_bytes / 1e9

    if total_gb <= max_gb:
        return

    # Find paused experiments, sorted by worst performance (evict worst first)
    paused = [r for r in results if r.get("status") == "paused"]
    paused.sort(key=lambda x: x.get("best_fresh", 0))

    for r in paused:
        if total_gb <= max_gb * 0.8:  # evict down to 80% of budget
            break
        exp_dir = runs_dir / r["exp_id"]
        ckpt = exp_dir / "checkpoint.pt"
        if ckpt.exists():
            size = ckpt.stat().st_size / 1e9
            ckpt.unlink()
            total_gb -= size
            print(f"  🗑 Evicted checkpoint {r['exp_id']} "
                  f"(fresh={r.get('best_fresh',0):.1%}, freed {size*1000:.0f}MB)",
                  flush=True)


def _run_generation(mgr, metrics, args, generation, max_workers, evo_state):
    """One generation of the evolutionary loop. Called from main loop with try/except."""
    evo_state.generation = generation
    results = mgr.get_all_metrics()
    results = score_experiments(results, evo_state)
    running = mgr.get_running()

    print(f"\n[Gen {generation}] {len(running)} running, "
          f"{len(results)} total experiments", flush=True)

    # Print leaderboard
    for i, r in enumerate(results[:10]):
        marker = "★" if i == 0 else " "
        parity = r.get("type_accs", {}).get("parity", 0)
        cfg = r.get("config", {})
        wd = cfg.get("weight_decay", 0)
        method = f"wd={wd}" if wd > 0 else "perp"
        backend = cfg.get("backend", "pytorch")
        tag = f"[{method}]" if backend == "pytorch" else f"[{method}/tg]"
        mom = r.get("momentum", 0)
        mom_str = f"↑{mom:.3f}" if mom > 0.001 else (f"↓{mom:.3f}" if mom < -0.001 else "→")
        spec = ",".join(r.get("specialist_for", [])[:2])
        spec_str = f" 🎯{spec}" if spec else ""
        print(f"  {marker} {r['exp_id']}: fresh={r.get('best_fresh', 0):.1%}  "
              f"eff={r.get('effective_score', 0):.1%}  {mom_str}  "
              f"parity={parity:.0%}  cycle={r.get('cycle', 0)}  "
              f"d={cfg.get('d_model', '?')}  {tag}{spec_str}  "
              f"[{r['status']}]", flush=True)

    # Check GPU usage + log + update dashboard
    gpu_pct, mem_pct = get_gpu_usage()
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total",
             "--format=csv,noheader,nounits"], text=True).strip()
        mem_used, mem_total = [float(x) for x in out.split(",")]
    except Exception:
        mem_used, mem_total = 0, 80000
    actual_workers = get_actual_worker_count()
    metrics.log_gpu(gpu_pct, mem_pct, mem_used, mem_total,
                   actual_workers, len(results))
    write_dashboard(results, generation, gpu_pct, mem_pct, evo_state)

    # Push to Firebase (real-time)
    try:
        import firebase_push as fb
        actual_workers = get_actual_worker_count()
        fb.push_snapshot(results, generation, gpu_pct, mem_pct, evo_state)
        fb.push_gpu_tick(gpu_pct, mem_pct, actual_workers, generation)

        # VRAM warning
        if mem_pct > 80:
            fb.evt_vram_warning(mem_pct, actual_workers)
    except Exception as e:
        print(f"  ⚠ Firebase push failed: {e}", flush=True)

    # ── Genetic evolution ──
    if len(results) < 2:
        return

    running_results = [r for r in results if r["status"] == "running"]
    if len(running_results) < 2:
        return

    # Plateau detection + Firebase events
    current_best = results[0].get("best_fresh", 0) if results else 0
    was_plateau = evo_state.plateau_mode
    old_best = evo_state.best_ever
    is_plateau = evo_state.check_plateau(current_best)
    severity = evo_state.get_plateau_severity()

    try:
        import firebase_push as fb
        # New best event
        if current_best > old_best + 0.005:
            fb.evt_new_best(results[0]["exp_id"], current_best, old_best,
                           results[0].get("config", {}))
        # Plateau start
        if is_plateau and not was_plateau:
            fb.evt_plateau_start(evo_state.best_ever,
                                evo_state.generation - evo_state.best_ever_gen, severity)
        # Plateau end
        if was_plateau and not is_plateau:
            fb.evt_plateau_end(old_best, current_best,
                              evo_state.generation - evo_state.best_ever_gen,
                              results[0]["exp_id"])
    except Exception:
        pass

    at_capacity = mem_pct > 75 or len(running) >= max_workers
    has_headroom = mem_pct < 65 and gpu_pct < 90

    plateau_str = f"  🔥 PLATEAU severity={severity:.1f} — aggressive mutation ON" if is_plateau else "  ✅ improving"
    print(f"  GPU: {gpu_pct:.0f}%  VRAM: {mem_pct:.0f}%  "
          f"{'at capacity' if at_capacity else f'headroom ({100-mem_pct:.0f}% VRAM free)'}"
          f"{plateau_str}  best_ever={evo_state.best_ever:.1%}", flush=True)

    # Auto-scale: spawn more workers if GPU has headroom
    if has_headroom and not at_capacity and running_results:
        parent, reason = select_parent(running_results, evo_state, generation)
        child_cfg = mutate_config(parent.get("config", SEED_CONFIGS[0]),
                                  plateau_severity=severity)
        child_id = mgr.spawn(child_cfg, parent_id=parent["exp_id"])
        evo_state.register_lineage(child_id, parent["exp_id"])
        print(f"  📈 Auto-scaled from {parent['exp_id']} ({reason})", flush=True)

    if at_capacity and generation % args.evolve_every == 0:
        # EVOLVE: pause worst running, spawn child of best
        # Grace period: don't pause experiments younger than 200 cycles
        GRACE_CYCLES = 200
        worst = None
        for r in reversed(running_results):
            if r["exp_id"] in running and r.get("cycle", 0) >= GRACE_CYCLES:
                worst = r
                break

        best = running_results[0]

        best, selection_reason = select_parent(running_results, evo_state, generation)

        if worst and best and worst["exp_id"] != best["exp_id"]:
            mgr.pause(worst["exp_id"])

            parent_cfg = best.get("config", SEED_CONFIGS[0])
            child_cfg = mutate_config(parent_cfg, plateau_severity=severity)
            child_id = mgr.spawn(child_cfg, parent_id=best["exp_id"])
            evo_state.register_lineage(child_id, best["exp_id"])

            metrics.log_event("evolve", child_id,
                f"child of {best['exp_id']}, replaced {worst['exp_id']}")
            metrics.log_event("pause", worst["exp_id"],
                f"paused for evolution (fresh={worst.get('best_fresh',0):.1%})")
            metrics.update_status(worst["exp_id"], "paused")
            try:
                import firebase_push as fb
                fb.evt_evolve(child_id, best["exp_id"], worst["exp_id"],
                             selection_reason, best.get("best_fresh", 0), child_cfg)
                fb.evt_pause(worst["exp_id"], worst.get("best_fresh", 0),
                            worst.get("cycle", 0))
            except Exception:
                pass
            print(f"  🧬 Evolution: paused {worst['exp_id']} "
                  f"(fresh={worst.get('best_fresh', 0):.1%}), "
                  f"spawned {child_id} from {best['exp_id']} "
                  f"[{selection_reason}]", flush=True)

    elif not at_capacity:
        parent, reason = select_parent(running_results, evo_state, generation)
        parent_cfg = parent.get("config", SEED_CONFIGS[0])
        child_cfg = mutate_config(parent_cfg)
        child_id = mgr.spawn(child_cfg, parent_id=parent["exp_id"])
        evo_state.register_lineage(child_id, parent["exp_id"])
        print(f"  🎯 Parent: {parent['exp_id']} ({reason})", flush=True)

    # Disk management: keep checkpoints under budget
    _enforce_disk_budget(mgr.runs_dir, results, max_gb=50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-workers", type=int, default=0,
                        help="Max parallel workers (0 = auto, fill GPU to 95%%)")
    parser.add_argument("--generation-every", type=int, default=60,
                        help="Seconds between evolution checks")
    parser.add_argument("--evolve-every", type=int, default=3,
                        help="Evolve every N generations (when at capacity)")
    parser.add_argument("--runs-dir", default="runs",
                        help="Directory for experiment runs")
    args = parser.parse_args()
    run_coordinator(args)
