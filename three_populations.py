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

def spawn_worker(task, config, runs_dir, worker_id):
    """Spawn a worker that trains on one specific task."""
    exp_id = f"w_{task}_{worker_id:03d}"
    run_dir = runs_dir / exp_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Add task to config
    cfg = config.copy()
    cfg["task"] = task

    config_path = run_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(cfg, f, indent=2)

    proc = subprocess.Popen(
        [sys.executable, "-u", "specialist_trainer.py",
         "--task", task,
         "--d-model", str(cfg.get("d_model", 64)),
         "--d-state", str(cfg.get("d_state", 16)),
         "--headdim", str(cfg.get("headdim", 16)),
         "--layers", str(cfg.get("n_kernel_layers", 3)),
         "--lr", str(cfg.get("lr", 1e-3)),
         "--weight-decay", str(cfg.get("weight_decay", 0.0)),
         "--optimizer", str(cfg.get("optimizer", "adamw")),
         "--loss-fn", str(cfg.get("loss_fn", "stable_ce")),
         "--batch-size", str(cfg.get("batch_size", 256)),
         "--steps-per-cycle", str(cfg.get("steps_per_cycle", 200)),
         "--max-cycles", "500",
         "--target-acc", "0.95"],
        stdout=open(run_dir / "stdout.log", "w"),
        stderr=subprocess.STDOUT,
        cwd=str(Path(__file__).parent),
    )

    return exp_id, proc


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
    worker_procs = {}    # exp_id → (proc, task, config)
    teacher_tasks = {}   # task → exp_id (graduated workers)
    student_procs = {}   # exp_id → (proc, config)
    worker_counter = 0
    student_counter = 0

    # Graceful shutdown
    should_stop = False
    def handle_signal(sig, frame):
        nonlocal should_stop
        should_stop = True
        print("\nShutting down...", flush=True)
    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    # ── Phase 1: Spawn workers (VRAM-aware, fill to 75%) ──
    print("Phase 1: Spawning workers (adapts to available hardware)...", flush=True)
    tasks_queue = list(ALL_TASKS)  # tasks needing their first worker
    for task in tasks_queue:
        gpu_pct, mem_pct = get_gpu_usage()
        if mem_pct > 75:
            print(f"  VRAM at {mem_pct:.0f}% — waiting for space before spawning more", flush=True)
            break
        cfg = BASE_CONFIG.copy()
        exp_id, proc = spawn_worker(task, cfg, workers_dir, worker_counter)
        worker_procs[exp_id] = (proc, task, cfg)
        worker_counter += 1
        print(f"  + Worker {exp_id} (VRAM: {mem_pct:.0f}%)", flush=True)
        time.sleep(10)  # let CUDA fully initialize before spawning next

    print(f"\n{len(worker_procs)} workers spawned.\n", flush=True)

    # ── Main loop ──
    generation = 0
    while not should_stop:
        time.sleep(args.check_interval)
        generation += 1

        # ── Check workers: who finished? who graduated? ──
        graduated = []
        finished = []
        for exp_id, (proc, task, cfg) in list(worker_procs.items()):
            if proc.poll() is not None:
                # Worker finished — check if it mastered
                specialist_ckpt = Path("checkpoints/specialists") / f"{task}.pt"
                if specialist_ckpt.exists():
                    ckpt = torch.load(specialist_ckpt, map_location="cpu", weights_only=False)
                    acc = ckpt.get("accuracy", 0)
                    if acc >= 0.95 and task not in teacher_tasks:
                        # GRADUATED! Move to teachers
                        teacher_tasks[task] = exp_id
                        # Copy checkpoint to teachers dir
                        import shutil
                        shutil.copy2(specialist_ckpt, teachers_dir / f"{task}.pt")
                        cache = Path("checkpoints/specialists") / f"{task}_cache.pt"
                        if cache.exists():
                            shutil.copy2(cache, teachers_dir / f"{task}_cache.pt")
                        graduated.append((task, exp_id, acc))
                    else:
                        finished.append((task, exp_id, acc))
                else:
                    finished.append((task, exp_id, 0))
                del worker_procs[exp_id]

        for task, exp_id, acc in graduated:
            print(f"  🎓 GRADUATED: {exp_id} → Teacher for {task} ({acc:.0%})", flush=True)
            try:
                import firebase_push as fb
                fb.evt_mastery(exp_id, task, 0, 0, 0)
                fb._put(f"mamba3/teachers/{task}", {
                    "exp_id": exp_id, "accuracy": round(acc, 3),
                    "graduated_at": time.time(),
                })
            except Exception:
                pass

        # ── Spawn workers for tasks that need them (VRAM-aware) ──
        gpu_pct, mem_pct = get_gpu_usage()
        tasks_needing_workers = [t for t in ALL_TASKS if t not in teacher_tasks]
        active_worker_tasks = set(task for _, (_, task, _) in worker_procs.items())

        for task in tasks_needing_workers:
            if task not in active_worker_tasks:
                gpu_pct, mem_pct = get_gpu_usage()
                if mem_pct > 75:
                    break  # wait for space
                cfg = mutate_config(BASE_CONFIG.copy(), plateau_severity=0)
                exp_id, proc = spawn_worker(task, cfg, workers_dir, worker_counter)
                worker_procs[exp_id] = (proc, task, cfg)
                worker_counter += 1
                print(f"  + Spawned {exp_id} (VRAM: {mem_pct:.0f}%)", flush=True)

        # ── Spawn/check students if we have teachers ──
        if teacher_tasks and not student_procs and mem_pct < 75:
            # First teacher arrived! Spawn first student
            print(f"\n  🎓 {len(teacher_tasks)} teachers available — spawning student!", flush=True)
            cfg = BASE_CONFIG.copy()
            exp_id, proc = spawn_student(student_counter, cfg, students_dir, teachers_dir)
            student_procs[exp_id] = (proc, cfg)
            student_counter += 1

        # Check student processes
        for exp_id, (proc, cfg) in list(student_procs.items()):
            if proc.poll() is not None:
                # Student finished — spawn evolved variant if space
                del student_procs[exp_id]
                if mem_pct < 75:
                    new_cfg = mutate_config(cfg.copy(), plateau_severity=0.5)
                    new_id, new_proc = spawn_student(student_counter, new_cfg, students_dir, teachers_dir)
                    student_procs[new_id] = (new_proc, new_cfg)
                    student_counter += 1
                    print(f"  🧬 Student evolved: {new_id}", flush=True)

        # ── Status ──
        n_workers = len(worker_procs)
        n_teachers = len(teacher_tasks)
        n_students = len(student_procs)
        actual = get_actual_worker_count()

        print(f"[Gen {generation}] Workers: {n_workers} | Teachers: {n_teachers}/{len(ALL_TASKS)} | "
              f"Students: {n_students} | GPU: {gpu_pct:.0f}% | VRAM: {mem_pct:.0f}%", flush=True)

        # Worker leaderboard
        if generation % 5 == 0:
            print(f"\n  --- WORKERS ---", flush=True)
            for task in ALL_TASKS:
                if task in teacher_tasks:
                    print(f"    ✅ {task}: GRADUATED", flush=True)
                elif task in active_worker_tasks:
                    print(f"    🔄 {task}: training...", flush=True)
                else:
                    print(f"    ⏳ {task}: waiting", flush=True)
            print(f"  --- TEACHERS ({n_teachers}/{len(ALL_TASKS)}) ---", flush=True)
            for task, exp_id in teacher_tasks.items():
                print(f"    🎓 {task}: {exp_id}", flush=True)
            print(f"  --- STUDENTS ({n_students}) ---", flush=True)
            for exp_id in student_procs:
                print(f"    📚 {exp_id}: distilling...", flush=True)
            print(flush=True)

        # Push to Firebase
        try:
            import firebase_push as fb
            fb._put("mamba3/three_pop", {
                "timestamp": time.time(),
                "generation": generation,
                "n_workers": n_workers,
                "n_teachers": n_teachers,
                "n_students": n_students,
                "teachers": {t: eid for t, eid in teacher_tasks.items()},
                "tasks_remaining": tasks_needing_workers,
                "gpu_pct": round(gpu_pct, 1),
                "mem_pct": round(mem_pct, 1),
            })
        except Exception:
            pass

    # Cleanup
    print("\nStopping all processes...", flush=True)
    for exp_id, (proc, _, _) in worker_procs.items():
        proc.terminate()
    for exp_id, (proc, _) in student_procs.items():
        proc.terminate()
    print("Done.", flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="three_pop")
    parser.add_argument("--check-interval", type=int, default=30)
    args = parser.parse_args()
    run(args)
