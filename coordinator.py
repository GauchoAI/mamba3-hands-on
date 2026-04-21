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
import random
import signal
import subprocess
from pathlib import Path
from datetime import datetime


# ── Genetic config mutation ─────────────────────────────────────────

def mutate_config(parent_config, mutation_strength=0.3):
    """Create a child config by mutating a parent."""
    child = parent_config.copy()

    # Mutate learning rate (log-scale)
    if random.random() < 0.5:
        factor = 2 ** (random.gauss(0, mutation_strength))
        child["lr"] = max(1e-5, min(1e-2, child["lr"] * factor))

    # Mutate batch size (powers of 2)
    if random.random() < 0.3:
        child["batch_size"] = random.choice([64, 128, 256, 512, 1024])

    # Mutate d_model
    if random.random() < 0.2:
        child["d_model"] = random.choice([32, 48, 64, 96, 128])
        child["headdim"] = min(child["headdim"], child["d_model"])

    # Mutate layers
    if random.random() < 0.2:
        child["n_kernel_layers"] = random.choice([1, 2, 3])

    # Mutate weight decay (toggle between grokking and PerpGrad)
    if random.random() < 0.2:
        child["weight_decay"] = random.choice([0.0, 0.05, 0.1, 0.2])

    # Mutate d_state
    if random.random() < 0.15:
        child["d_state"] = random.choice([8, 16, 32])

    return child


# ── Default seed configs ────────────────────────────────────────────

SEED_CONFIGS = [
    {"d_model": 64,  "d_state": 16, "headdim": 16, "n_kernel_layers": 1,
     "batch_size": 256, "lr": 1e-3, "weight_decay": 0.1, "steps_per_cycle": 200},
    {"d_model": 32,  "d_state": 16, "headdim": 16, "n_kernel_layers": 1,
     "batch_size": 512, "lr": 1e-3, "weight_decay": 0.0, "steps_per_cycle": 200},
    {"d_model": 64,  "d_state": 16, "headdim": 16, "n_kernel_layers": 2,
     "batch_size": 256, "lr": 1e-3, "weight_decay": 0.1, "steps_per_cycle": 200},
    {"d_model": 128, "d_state": 16, "headdim": 16, "n_kernel_layers": 1,
     "batch_size": 128, "lr": 5e-4, "weight_decay": 0.1, "steps_per_cycle": 200},
    {"d_model": 64,  "d_state": 8,  "headdim": 8,  "n_kernel_layers": 1,
     "batch_size": 512, "lr": 1e-3, "weight_decay": 0.0, "steps_per_cycle": 200},
    {"d_model": 48,  "d_state": 16, "headdim": 16, "n_kernel_layers": 1,
     "batch_size": 256, "lr": 2e-3, "weight_decay": 0.05, "steps_per_cycle": 200},
]


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

    def spawn(self, config):
        """Launch a new worker process."""
        self._find_next_id()
        exp_id = f"exp_{self.next_id:03d}"
        run_dir = self.runs_dir / exp_id
        run_dir.mkdir(exist_ok=True)

        config_path = run_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        proc = subprocess.Popen(
            [sys.executable, "-u", "worker.py",
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

def write_dashboard(results, generation, runs_dir):
    """Write dashboard.md + dashboard.html from coordinator's view."""
    from dashboard import generate_dashboard

    # Also write a simple markdown leaderboard
    lines = [f"# Coordinator Dashboard — Generation {generation}",
             f"Updated: {datetime.now().strftime('%H:%M:%S')}", ""]
    lines.append("## Leaderboard")
    lines.append("| Rank | ID | Status | Fresh | Parity | Params | Config | Method |")
    lines.append("|------|-----|--------|-------|--------|--------|--------|--------|")
    for i, r in enumerate(results):
        cfg = r.get("config", {})
        method = f"wd={cfg.get('weight_decay', '?')}" if cfg.get("weight_decay", 0) > 0 else "PerpGrad"
        parity = r.get("type_accs", {}).get("parity", 0)
        lines.append(
            f"| {i+1} | {r['exp_id']} | {r['status']} | "
            f"{r.get('best_fresh', 0):.1%} | {parity:.0%} | "
            f"{r.get('params', '?'):,} | d={cfg.get('d_model', '?')} L={cfg.get('n_kernel_layers', '?')} | "
            f"{method} |"
        )

    # Teacher status from best experiment
    if results:
        lines.append("")
        lines.append("## Best Experiment Teacher")
        lines.append(f"```\n{results[0].get('teacher_status', 'N/A')}\n```")

        if results[0].get("mastery_log"):
            lines.append("")
            lines.append("## Learning-to-Learn")
            for entry in results[0]["mastery_log"]:
                lines.append(f"- Task {entry['task_index']+1}: {entry['task']} — "
                           f"{entry['steps_to_master']} steps, {entry['examples_to_master']:,} examples")

    Path("dashboard.md").write_text("\n".join(lines))


# ── Main coordinator loop ───────────────────────────────────────────

def run_coordinator(args):
    print(f"{'='*60}", flush=True)
    print(f"COORDINATOR — Evolutionary Model Shopping", flush=True)
    print(f"  max_workers={args.max_workers}", flush=True)
    print(f"  generation_every={args.generation_every}s", flush=True)
    print(f"{'='*60}\n", flush=True)

    mgr = WorkerManager(runs_dir=args.runs_dir)

    # Spawn initial population from seed configs
    for cfg in SEED_CONFIGS[:args.max_workers]:
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

        # Read all metrics
        results = mgr.get_all_metrics()
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
            print(f"  {marker} {r['exp_id']}: fresh={r.get('best_fresh', 0):.1%}  "
                  f"parity={parity:.0%}  cycle={r.get('cycle', 0)}  "
                  f"d={cfg.get('d_model', '?')}  [{method}]  "
                  f"[{r['status']}]", flush=True)

        # Update dashboard
        write_dashboard(results, generation, args.runs_dir)

        # ── Genetic evolution ──
        if len(results) < 2:
            continue

        running_results = [r for r in results if r["status"] == "running"]
        if len(running_results) < 2:
            continue

        # Check if we're at capacity
        at_capacity = len(running) >= args.max_workers

        if at_capacity and generation % args.evolve_every == 0:
            # EVOLVE: pause worst running, spawn child of best
            worst = None
            for r in reversed(running_results):
                if r["exp_id"] in running:
                    worst = r
                    break

            best = running_results[0]

            if worst and best and worst["exp_id"] != best["exp_id"]:
                # Pause the worst
                mgr.pause(worst["exp_id"])

                # Spawn child of the best
                parent_cfg = best.get("config", SEED_CONFIGS[0])
                child_cfg = mutate_config(parent_cfg)
                child_id = mgr.spawn(child_cfg)

                print(f"  🧬 Evolution: paused {worst['exp_id']} "
                      f"(fresh={worst.get('best_fresh', 0):.1%}), "
                      f"spawned {child_id} from {best['exp_id']} "
                      f"(fresh={best.get('best_fresh', 0):.1%})", flush=True)

        elif not at_capacity:
            # Room for more — spawn a mutant of a random top-half worker
            top_half = running_results[:max(1, len(running_results) // 2)]
            parent = random.choice(top_half)
            parent_cfg = parent.get("config", SEED_CONFIGS[0])
            child_cfg = mutate_config(parent_cfg)
            mgr.spawn(child_cfg)

    # Shutdown all workers
    print("\nStopping all workers...", flush=True)
    for exp_id in list(mgr.processes.keys()):
        mgr.pause(exp_id)

    # Final leaderboard
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-workers", type=int, default=6,
                        help="Max parallel worker processes")
    parser.add_argument("--generation-every", type=int, default=60,
                        help="Seconds between evolution checks")
    parser.add_argument("--evolve-every", type=int, default=3,
                        help="Evolve every N generations (when at capacity)")
    parser.add_argument("--runs-dir", default="runs",
                        help="Directory for experiment runs")
    args = parser.parse_args()
    run_coordinator(args)
