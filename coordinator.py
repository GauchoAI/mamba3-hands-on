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

    # Small chance to flip backend (pytorch ↔ tinygrad)
    if random.random() < 0.1:
        child["backend"] = "tinygrad" if child.get("backend") != "tinygrad" else "pytorch"

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
    # tinygrad experiment — same config as best grokking, different backend
    {"d_model": 64,  "d_state": 16, "headdim": 16, "n_kernel_layers": 1,
     "batch_size": 256, "lr": 1e-3, "weight_decay": 0.1, "steps_per_cycle": 200,
     "backend": "tinygrad"},
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

def write_dashboard(results, generation, gpu_pct, mem_pct):
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
    lines = [f"# Evolution Dashboard — Gen {generation}", f"GPU {gpu_pct:.0f}% · VRAM {mem_pct:.0f}%", ""]
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
        import subprocess
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


def run_coordinator(args):
    print(f"{'='*60}", flush=True)
    print(f"COORDINATOR — Evolutionary Model Shopping", flush=True)
    print(f"  max_workers={'auto (fill to 95%)' if args.max_workers == 0 else args.max_workers}", flush=True)
    print(f"  generation_every={args.generation_every}s", flush=True)
    print(f"{'='*60}\n", flush=True)

    from metrics_db import MetricsWriter
    metrics = MetricsWriter()

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
            _run_generation(mgr, teacher, metrics, args, generation,
                           max_workers, should_stop)
        except Exception as e:
            print(f"  ⚠ Generation {generation} error: {e}", flush=True)
            import traceback
            traceback.print_exc()
            # Don't crash — just skip this generation and continue

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


def _run_generation(mgr, teacher, metrics, args, generation, max_workers, should_stop):
    """One generation of the evolutionary loop. Called from main loop with try/except."""
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
        backend = cfg.get("backend", "pytorch")
        tag = f"[{method}]" if backend == "pytorch" else f"[{method}/tg]"
        print(f"  {marker} {r['exp_id']}: fresh={r.get('best_fresh', 0):.1%}  "
              f"parity={parity:.0%}  cycle={r.get('cycle', 0)}  "
              f"d={cfg.get('d_model', '?')}  {tag}  "
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
    metrics.log_gpu(gpu_pct, mem_pct, mem_used, mem_total,
                   len(running), len(results))
    write_dashboard(results, generation, gpu_pct, mem_pct)

    # ── Genetic evolution ──
    if len(results) < 2:
        return

    running_results = [r for r in results if r["status"] == "running"]
    if len(running_results) < 2:
        return

    at_capacity = mem_pct > 90 or len(running) >= max_workers
    has_headroom = mem_pct < 85 and gpu_pct < 90

    print(f"  GPU: {gpu_pct:.0f}%  VRAM: {mem_pct:.0f}%  "
          f"{'at capacity' if at_capacity else f'headroom ({100-mem_pct:.0f}% VRAM free)'}",
          flush=True)

    # Auto-scale: spawn more workers if GPU has headroom
    if has_headroom and not at_capacity and running_results:
        parent = random.choice(running_results[:max(1, len(running_results) // 2)])
        child_cfg = mutate_config(parent.get("config", SEED_CONFIGS[0]))
        mgr.spawn(child_cfg)
        print(f"  📈 Auto-scaled (GPU has headroom)", flush=True)

    if at_capacity and generation % args.evolve_every == 0:
        # EVOLVE: pause worst running, spawn child of best
        worst = None
        for r in reversed(running_results):
            if r["exp_id"] in running:
                worst = r
                break

        best = running_results[0]

        if worst and best and worst["exp_id"] != best["exp_id"]:
            mgr.pause(worst["exp_id"])

            parent_cfg = best.get("config", SEED_CONFIGS[0])
            child_cfg = mutate_config(parent_cfg)
            child_id = mgr.spawn(child_cfg)

            metrics.log_event("evolve", child_id,
                f"child of {best['exp_id']}, replaced {worst['exp_id']}")
            metrics.log_event("pause", worst["exp_id"],
                f"paused for evolution (fresh={worst.get('best_fresh',0):.1%})")
            metrics.update_status(worst["exp_id"], "paused")
            print(f"  🧬 Evolution: paused {worst['exp_id']} "
                  f"(fresh={worst.get('best_fresh', 0):.1%}), "
                  f"spawned {child_id} from {best['exp_id']} "
                  f"(fresh={best.get('best_fresh', 0):.1%})", flush=True)

    elif not at_capacity:
        top_half = running_results[:max(1, len(running_results) // 2)]
        parent = random.choice(top_half)
        parent_cfg = parent.get("config", SEED_CONFIGS[0])
        child_cfg = mutate_config(parent_cfg)
        mgr.spawn(child_cfg)


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
