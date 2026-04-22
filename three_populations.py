"""
Three Populations: Workers → Teachers → Students.

Workers: per-task GA. Each trains on ONE task. Evolve architecture/hyperparams.
         When 100% → graduate to Teacher.
Teachers: frozen masters. One per task. Teach students via distillation.
Students: distill from all available teachers. Also evolve (population of students).

Self-organizing: no manual intervention. Workers graduate automatically.
Students start distilling as soon as the first teacher arrives.

Usage:
    python three_populations.py
"""
import os
os.environ["PYTHONUNBUFFERED"] = "1"
import sys
sys.path.insert(0, os.path.dirname(__file__))

import json
import time
import random
import signal
import subprocess
import torch
from pathlib import Path
from datetime import datetime
from collections import defaultdict

from progressive_model import ProgressiveModel, ByteTokenizer, VOCAB_SIZE, PAD
from coordinator import mutate_config, SEED_CONFIGS, get_gpu_usage, get_actual_worker_count


# ── Tasks ───────────────────────────────────────────────────────────

ALL_TASKS = [
    "parity", "binary_pattern_next", "same_different", "odd_one_out",
    "sequence_completion", "pattern_period", "run_length_next",
    "mirror_detection", "repeat_count", "arithmetic_next",
    "geometric_next", "alternating_next", "logic_gate", "logic_chain",
    "modus_ponens",
]

BASE_CONFIG = {
    "d_model": 64, "d_state": 16, "headdim": 16, "n_kernel_layers": 3,
    "batch_size": 256, "lr": 1e-3, "weight_decay": 0.1,
    "steps_per_cycle": 200, "loss_fn": "ce", "optimizer": "adamw",
    "use_perp": False,
}


# ── Worker: trains on ONE task ──────────────────────────────────────

def train_worker_inline(task, config, runs_dir, worker_id, device):
    """Train a specialist INLINE (same process, shared CUDA context). No subprocess."""
    from specialist_trainer import train_specialist

    exp_id = f"w_{task}_{worker_id:03d}"
    run_dir = runs_dir / exp_id
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg = config.copy()
    cfg["task"] = task
    with open(run_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    # Redirect stdout to log
    log = open(run_dir / "stdout.log", "w")
    old_stdout = sys.stdout
    sys.stdout = log

    try:
        acc = train_specialist(task, cfg, device, max_cycles=200, target_acc=0.95)
    except Exception as e:
        acc = 0
        print(f"ERROR: {e}", flush=True)
    finally:
        sys.stdout = old_stdout
        log.close()

    return exp_id, acc


# ── Student worker: distills from cached teacher outputs ────────────

def spawn_student(student_id, config, runs_dir, teacher_dir):
    """Spawn a student that distills from cached teacher outputs."""
    exp_id = f"s_{student_id:03d}"
    run_dir = runs_dir / exp_id
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg = config.copy()
    config_path = run_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(cfg, f, indent=2)

    proc = subprocess.Popen(
        [sys.executable, "-u", "distill.py",
         "--runs-dir", str(runs_dir.parent / "runs"),
         "--student-d-model", str(cfg.get("d_model", 64)),
         "--student-d-state", str(cfg.get("d_state", 16)),
         "--student-headdim", str(cfg.get("headdim", 16)),
         "--student-layers", str(cfg.get("n_kernel_layers", 3)),
         "--lr", str(cfg.get("lr", 1e-3)),
         "--cycles", "500",
         "--steps-per-cycle", "50",
         "--pcgrad"],
        stdout=open(run_dir / "stdout.log", "w"),
        stderr=subprocess.STDOUT,
        cwd=str(Path(__file__).parent),
    )

    return exp_id, proc


# ── Lineage logger ─────────────────────────────────────────────────

def _write_lineage(path, task, lineage):
    """Write per-task lineage to markdown."""
    with open(path, "w") as f:
        f.write(f"# {task} — Training Lineage\n\n")
        f.write(f"| Round | Acc | Best | d_model | Layers | LR | WD | Optimizer | Loss | Mutation |\n")
        f.write(f"|-------|-----|------|---------|--------|----|----|-----------|------|-----------|\n")
        for e in lineage:
            c = e["config"]
            mut = e.get("mutation") or "—"
            f.write(f"| {e['round']} | {e['acc']:.0%} | {e['best']:.0%} "
                   f"| {c.get('d_model', 64)} | {c.get('n_kernel_layers', 3)} "
                   f"| {c.get('lr', 1e-3):.0e} | {c.get('weight_decay', 0.1)} "
                   f"| {c.get('optimizer', 'adamw')} | {c.get('loss_fn', 'ce')} "
                   f"| {mut} |\n")


# ── Main orchestrator ───────────────────────────────────────────────

def _acquire_lock():
    """Ensure only one instance runs. Kill any existing instance."""
    lock_path = Path("three_pop.pid")
    if lock_path.exists():
        old_pid = int(lock_path.read_text().strip())
        try:
            os.kill(old_pid, 0)  # check if alive
            print(f"Killing existing instance (PID {old_pid})...", flush=True)
            os.kill(old_pid, signal.SIGTERM)
            time.sleep(2)
            try:
                os.kill(old_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        except ProcessLookupError:
            pass  # already dead
    lock_path.write_text(str(os.getpid()))
    return lock_path


def run(args):
    lock_path = _acquire_lock()

    base_dir = Path(args.dir)
    workers_dir = base_dir / "workers"
    teachers_dir = base_dir / "teachers"
    students_dir = base_dir / "students"
    for d in [workers_dir, teachers_dir, students_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print(f"{'='*60}", flush=True)
    print(f"THREE POPULATIONS — Workers → Teachers → Students", flush=True)
    print(f"  Workers dir:  {workers_dir}", flush=True)
    print(f"  Teachers dir: {teachers_dir}", flush=True)
    print(f"  Students dir: {students_dir}", flush=True)
    print(f"{'='*60}\n", flush=True)

    # State
    teacher_tasks = {}   # task → exp_id (graduated workers)
    worker_counter = 0
    student_counter = 0

    # Device
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Device: {device}\n", flush=True)

    # Graceful shutdown
    should_stop = False
    def handle_signal(sig, frame):
        nonlocal should_stop
        should_stop = True
        print("\nShutting down...", flush=True)
    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    # ── Train specialists one at a time (shared CUDA context) ──
    # Round-robin: train each task for a few cycles, rotate, repeat.
    # This way ALL tasks show progress on the UI simultaneously.

    from specialist_trainer import train_specialist, load_generators
    load_generators()

    tasks_remaining = list(ALL_TASKS)
    cycles_per_round = 10  # train each task for 10 cycles before rotating
    round_num = 0

    # Per-task tracking: config, plateau detection, lineage
    task_config = {}      # task → current config dict
    task_best = {}        # task → best_acc seen
    task_best_round = {}  # task → round when best was set
    task_lineage = {}     # task → list of {round, config, acc, mutation}

    # Initialize all tasks with BASE_CONFIG
    task_best_config = {}  # task → config that achieved the best accuracy
    for t in ALL_TASKS:
        task_config[t] = BASE_CONFIG.copy()
        task_best[t] = 0.0
        task_best_round[t] = 0
        task_best_config[t] = BASE_CONFIG.copy()
        task_lineage[t] = []

    # Lineage log directory
    lineage_dir = base_dir / "lineage"
    lineage_dir.mkdir(parents=True, exist_ok=True)

    PLATEAU_THRESHOLD = 3  # rounds without improvement before mutating

    print(f"Training {len(tasks_remaining)} tasks, {cycles_per_round} cycles per round", flush=True)
    print(f"Round-robin: all tasks advance simultaneously\n", flush=True)

    while tasks_remaining and not should_stop:
        round_num += 1
        print(f"\n{'='*60}", flush=True)
        print(f"Round {round_num} — {len(tasks_remaining)} tasks, "
              f"{len(teacher_tasks)} teachers", flush=True)
        print(f"{'='*60}", flush=True)

        # Track per-task accuracy for this round
        task_accs = {}

        def on_cycle(task_name, cycle, acc, best, loss):
            """Called after each training cycle — push to Firebase."""
            task_accs[task_name] = {"acc": round(acc, 3), "best": round(best, 3),
                                    "cycle": cycle, "loss": round(loss, 3)}
            try:
                import firebase_push as fb
                # Update snapshot with current state
                leaderboard = []
                for t in ALL_TASKS:
                    if t in teacher_tasks:
                        leaderboard.append({"task": t, "acc": 1.0, "status": "teacher",
                                           "exp_id": teacher_tasks[t]})
                    elif t in task_accs:
                        leaderboard.append({"task": t, "acc": task_accs[t]["acc"],
                                           "best": task_accs[t]["best"],
                                           "cycle": task_accs[t]["cycle"],
                                           "exp_id": t,
                                           "fresh": task_accs[t]["acc"],
                                           "status": "training"})
                    else:
                        leaderboard.append({"task": t, "acc": 0, "status": "waiting"})
                leaderboard.sort(key=lambda x: -x["acc"])

                tasks_data = {}
                for entry in leaderboard:
                    tasks_data[entry["task"]] = {"acc": entry["acc"], "exp": entry.get("exp_id", "worker")}

                gpu_pct, mem_pct = get_gpu_usage()

                # Build lineage from task_lineage for family tree
                lineage_data = {}
                for t, entries in task_lineage.items():
                    for i, e in enumerate(entries):
                        node_id = f"{t}_r{e['round']}"
                        parent_id = f"{t}_r{entries[i-1]['round']}" if i > 0 else None
                        lineage_data[node_id] = {"parent": parent_id}

                fb._put("mamba3/snapshot", {
                    "timestamp": time.time(),
                    "mode": "three_populations",
                    "gpu_pct": round(gpu_pct, 1),
                    "mem_pct": round(mem_pct, 1),
                    "n_running": len(tasks_remaining),
                    "n_total": len(ALL_TASKS),
                    "best_fresh": 0,
                    "leaderboard": leaderboard,
                    "tasks": tasks_data,
                    "lineage": lineage_data,
                    "three_pop": {
                        "n_workers": len(tasks_remaining),
                        "n_teachers": len(teacher_tasks),
                        "n_students": 0,
                        "teachers": {t: eid for t, eid in teacher_tasks.items()},
                        "tasks_remaining": tasks_remaining,
                        "generation": round_num,
                    },
                })
                # Per-task timeseries
                fb._put(f"mamba3/task_series/{task_name}/{cycle}", {
                    "acc": round(acc, 3), "diff": 0,
                })
                # Per-experiment data (for fresh accuracy chart)
                fb._put(f"mamba3/experiments/{task_name}/best_fresh", round(best, 4))
                fb._put(f"mamba3/experiments/{task_name}/status",
                       "teacher" if task_name in teacher_tasks else "training")
                fb._put(f"mamba3/experiments/{task_name}/cycle", cycle)
                fb._put(f"mamba3/experiments/{task_name}/cycles/{cycle}", {
                    "fresh": round(acc, 4), "loss": round(loss, 4), "t": time.time(),
                })
            except Exception:
                pass

        # ── Determine pool size from VRAM ──
        gpu_pct, mem_pct = get_gpu_usage()
        vram_free_pct = 100 - mem_pct
        # Each worker uses ~2-5% VRAM. Allow up to N workers while keeping 30% free.
        max_concurrent = max(1, min(4, int(vram_free_pct / 20)))

        active_tasks = [t for t in tasks_remaining
                       if t not in teacher_tasks and not should_stop]

        # ── Train champions in batches of max_concurrent ──
        for batch_start in range(0, len(active_tasks), max_concurrent):
            if should_stop:
                break
            batch = active_tasks[batch_start:batch_start + max_concurrent]

            if len(batch) > 1:
                print(f"\n  Training {len(batch)} tasks concurrently "
                      f"(VRAM: {mem_pct:.0f}%, pool: {max_concurrent})", flush=True)

            # Launch batch as subprocesses
            procs = {}
            for task in batch:
                cfg = task_config[task]
                print(f"  + {task} (d={cfg.get('d_model')} L={cfg.get('n_kernel_layers')} "
                      f"lr={cfg.get('lr', 1e-3):.0e} {cfg.get('optimizer', 'adamw')} "
                      f"{cfg.get('loss_fn', 'ce')})", flush=True)

                proc = subprocess.Popen(
                    [sys.executable, "-u", "specialist_trainer.py",
                     "--task", task,
                     "--d-model", str(cfg.get("d_model", 64)),
                     "--d-state", str(cfg.get("d_state", 16)),
                     "--headdim", str(cfg.get("headdim", 16)),
                     "--layers", str(cfg.get("n_kernel_layers", 3)),
                     "--lr", str(cfg.get("lr", 1e-3)),
                     "--weight-decay", str(cfg.get("weight_decay", 0.1)),
                     "--optimizer", str(cfg.get("optimizer", "adamw")),
                     "--loss-fn", str(cfg.get("loss_fn", "ce")),
                     "--batch-size", str(cfg.get("batch_size", 256)),
                     "--steps-per-cycle", str(cfg.get("steps_per_cycle", 200)),
                     "--max-cycles", str(cycles_per_round),
                     "--target-acc", "0.95"],
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    cwd=str(Path(__file__).parent),
                )
                procs[task] = proc

            # Wait for all to finish, collect results
            for task, proc in procs.items():
                output = proc.communicate(timeout=600)[0].decode("utf-8", errors="ignore")
                # Print last few lines
                lines = output.strip().split("\n")
                for line in lines[-3:]:
                    print(f"    {line}", flush=True)

                # Read accuracy from checkpoint
                cfg = task_config[task]
                ckpt_path = Path("checkpoints/specialists") / f"{task}.pt"
                acc = 0.0
                cycle = 0
                if ckpt_path.exists():
                    try:
                        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                        cycle = ckpt.get("cycles", 0)
                        acc = ckpt.get("accuracy", 0.0)
                    except Exception:
                        pass

                # Track best
                mutation_desc = None
                if acc > task_best.get(task, 0):
                    task_best[task] = acc
                    task_best_round[task] = round_num
                    task_best_config[task] = cfg.copy()

                # Check graduation
                if acc >= 0.95:
                    exp_id = f"w_{task}_{worker_counter:03d}"
                    teacher_tasks[task] = exp_id
                    if task in tasks_remaining:
                        tasks_remaining.remove(task)
                    worker_counter += 1
                    print(f"  🎓 GRADUATED: {task} → Teacher ({acc:.0%})", flush=True)

                    import shutil
                    if ckpt_path.exists():
                        shutil.copy2(ckpt_path, teachers_dir / f"{task}.pt")
                    cache = Path("checkpoints/specialists") / f"{task}_cache.pt"
                    if cache.exists():
                        shutil.copy2(cache, teachers_dir / f"{task}_cache.pt")

                    try:
                        import firebase_push as fb
                        fb.evt_mastery(exp_id, task, 0, 0, 0)
                    except Exception:
                        pass

                # Push to Firebase after each task
                try:
                    on_cycle(task, cycle, acc, task_best.get(task, 0), 0)
                except Exception:
                    pass

                # Plateau? Run challenger (sequential — one at a time for challengers)
                rounds_stuck = round_num - task_best_round.get(task, 0)
                best = task_best.get(task, 0)

                if rounds_stuck >= PLATEAU_THRESHOLD and best > 0 and acc < 0.95:
                    severity = min(3.0, rounds_stuck / 3)
                    base = task_best_config[task].copy()
                    challenger_cfg = mutate_config(base, plateau_severity=severity)

                    changes = {k: challenger_cfg[k] for k in challenger_cfg
                              if challenger_cfg.get(k) != base.get(k) and k != "steps_per_cycle"}
                    mutation_desc = f"severity={severity:.1f} changes={changes}"

                    arch_changed = any(challenger_cfg.get(k) != base.get(k)
                                      for k in ["d_model", "d_state", "headdim", "n_kernel_layers"])

                    # Back up champion
                    champion_ckpt = Path("checkpoints/specialists") / f"{task}_champion.pt"
                    if ckpt_path.exists():
                        import shutil
                        shutil.copy2(ckpt_path, champion_ckpt)
                        if arch_changed:
                            ckpt_path.unlink()

                    print(f"  🧬 CHALLENGER for {task}: {changes}"
                          f"{' (fresh)' if arch_changed else ''}", flush=True)

                    # Run challenger inline (sequential)
                    challenger_acc = train_specialist(
                        task, challenger_cfg, device,
                        max_cycles=cycles_per_round,
                        target_acc=0.95,
                        on_cycle=on_cycle,
                    )

                    if challenger_acc and challenger_acc > best:
                        print(f"  ✓ Challenger wins: {challenger_acc:.0%} > {best:.0%}", flush=True)
                        task_config[task] = challenger_cfg
                        task_best[task] = challenger_acc
                        task_best_round[task] = round_num
                        task_best_config[task] = challenger_cfg.copy()
                        acc = challenger_acc
                    else:
                        print(f"  ✗ Champion holds: {best:.0%} >= {challenger_acc:.0%}", flush=True)
                        if champion_ckpt.exists():
                            import shutil
                            shutil.copy2(champion_ckpt, ckpt_path)
                        task_config[task] = task_best_config[task].copy()

            # Log lineage
            entry = {
                "round": round_num,
                "config": {k: v for k, v in cfg.items() if k != "steps_per_cycle"},
                "acc": round(acc, 3) if acc else 0,
                "best": round(task_best.get(task, 0), 3),
                "mutation": mutation_desc,
            }
            task_lineage[task].append(entry)

            # Write lineage markdown
            _write_lineage(lineage_dir / f"{task}.md", task, task_lineage[task])

            # Check graduation
            if acc and acc >= 0.95:
                exp_id = f"w_{task}_{worker_counter:03d}"
                teacher_tasks[task] = exp_id
                tasks_remaining.remove(task)
                worker_counter += 1
                print(f"  🎓 GRADUATED: {task} → Teacher ({acc:.0%})", flush=True)

                import shutil
                ckpt = Path("checkpoints/specialists") / f"{task}.pt"
                if ckpt.exists():
                    shutil.copy2(ckpt, teachers_dir / f"{task}.pt")
                cache = Path("checkpoints/specialists") / f"{task}_cache.pt"
                if cache.exists():
                    shutil.copy2(cache, teachers_dir / f"{task}_cache.pt")

                try:
                    import firebase_push as fb
                    fb.evt_mastery(exp_id, task, 0, 0, 0)
                except Exception:
                    pass

        # Push status to Firebase
        gpu_pct, mem_pct = get_gpu_usage()
        try:
            import firebase_push as fb
            fb._put("mamba3/three_pop", {
                "timestamp": time.time(),
                "generation": round_num,
                "n_workers": len(tasks_remaining),
                "n_teachers": len(teacher_tasks),
                "n_students": 0,
                "teachers": {t: eid for t, eid in teacher_tasks.items()},
                "tasks_remaining": tasks_remaining,
                "gpu_pct": round(gpu_pct, 1),
                "mem_pct": round(mem_pct, 1),
            })
        except Exception:
            pass

        print(f"\n[Round {round_num}] Teachers: {len(teacher_tasks)}/{len(ALL_TASKS)} | "
              f"Remaining: {len(tasks_remaining)} | GPU: {gpu_pct:.0f}%", flush=True)
        for task in ALL_TASKS:
            if task in teacher_tasks:
                print(f"    ✅ {task}", flush=True)
            elif task in tasks_remaining:
                print(f"    🔄 {task}", flush=True)

    # All tasks mastered — start distillation
    if not tasks_remaining:
        print(f"\n{'='*60}", flush=True)
        print(f"ALL {len(ALL_TASKS)} TASKS MASTERED — Starting distillation!", flush=True)
        print(f"{'='*60}", flush=True)
        # TODO: distillation phase

    lock_path.unlink(missing_ok=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="three_pop")
    parser.add_argument("--check-interval", type=int, default=30)
    args = parser.parse_args()
    run(args)
