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


# ── Main orchestrator ───────────────────────────────────────────────

def run(args):
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

    # Per-task state: model + optimizer persist across rounds
    task_state = {}  # task → {"model": model, "opt": opt, "cycle": N, "best_acc": N}

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
                                           "status": "training"})
                    else:
                        leaderboard.append({"task": t, "acc": 0, "status": "waiting"})
                leaderboard.sort(key=lambda x: -x["acc"])

                tasks_data = {}
                for entry in leaderboard:
                    tasks_data[entry["task"]] = {"acc": entry["acc"], "exp": entry.get("exp_id", "worker")}

                gpu_pct, mem_pct = get_gpu_usage()
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
                # Per-experiment cycle data (for Fresh Accuracy Over Time chart)
                fb._put(f"mamba3/experiments/{task_name}", {
                    "best_fresh": round(best, 4),
                    "status": "training",
                    "config": {"task": task_name, "d_model": 64, "n_kernel_layers": 3},
                    "n_params": 103539,
                    "cycle": cycle,
                    "cycles": None,  # will be set below
                })
                fb._put(f"mamba3/experiments/{task_name}/cycles/{cycle}", {
                    "fresh": round(acc, 4),
                    "loss": round(loss, 4),
                    "t": time.time(),
                })
                # Append-only history for replay
                fb._put(f"mamba3/history/{int(time.time())}", {
                    "best_fresh": round(max(ta.get("best", 0) for ta in task_accs.values()) if task_accs else 0, 4),
                    "gpu_pct": round(gpu_pct, 1),
                    "mem_pct": round(mem_pct, 1),
                    "n_workers": len(tasks_remaining),
                    "n_teachers": len(teacher_tasks),
                    "n_students": 0,
                    "tasks": {t: {"acc": round(ta["acc"], 3), "exp": "worker"} for t, ta in task_accs.items()},
                    "teachers": list(teacher_tasks.keys()),
                    "worker_best": {t: round(ta["best"], 3) for t, ta in task_accs.items()},
                })
            except Exception:
                pass

        for task in list(tasks_remaining):
            if should_stop:
                break
            if task in teacher_tasks:
                continue

            print(f"\n  Training {task}...", flush=True)
            cfg = BASE_CONFIG.copy()
            acc = train_specialist(
                task, cfg, device,
                max_cycles=cycles_per_round,
                target_acc=0.95,
                on_cycle=on_cycle,
            )

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

    print("Done.", flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="three_pop")
    parser.add_argument("--check-interval", type=int, default=30)
    args = parser.parse_args()
    run(args)
