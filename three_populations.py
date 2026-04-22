"""
Three Populations: Workers → Teachers → Students.

Workers: per-task GA. Multiple workers per task with different configs.
         Genetic algorithm evolves architecture/hyperparams.
         When any worker hits 95% → task graduates to Teacher.
Teachers: frozen masters. One per task. Teach students via distillation.
Students: distill from all available teachers. Start on FIRST teacher.

Resource-aware: workers admitted 1-at-a-time based on CPU/RAM/GPU headroom.
Self-organizing: no manual intervention. Workers graduate automatically.

Usage:
    python three_populations.py
"""
import os
os.environ["PYTHONUNBUFFERED"] = "1"
import sys
sys.path.insert(0, os.path.dirname(__file__))

import json
import time
import signal
import shutil
import torch
from pathlib import Path
from collections import defaultdict

from coordinator import SEED_CONFIGS, get_system_resources
from resource_pool import ResourceAwarePool


# ── Tasks ───────────────────────────────────────────────────────────

ALL_TASKS = [
    "parity", "binary_pattern_next", "same_different", "odd_one_out",
    "sequence_completion", "pattern_period", "run_length_next",
    "mirror_detection", "repeat_count", "arithmetic_next",
    "geometric_next", "alternating_next", "logic_gate", "logic_chain",
    "modus_ponens",
]


# ── Main orchestrator ───────────────────────────────────────────────

def run(args):
    base_dir = Path(args.dir)
    teachers_dir = base_dir / "teachers"
    teachers_dir.mkdir(parents=True, exist_ok=True)

    # Device
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print(f"Device: {device}\n", flush=True)

    # Track per-task latest accuracy (for Firebase snapshot)
    task_accs = {}  # task → {"acc": N, "best": N, "cycle": N, "loss": N}

    # ── Firebase on_cycle callback ──────────────────────────────────

    def on_cycle(*, task, cycle, acc, best_acc, loss, exp_id, config,
                 parent_id, n_params, cpu_pct, ram_pct, gpu_pct, vram_pct,
                 n_workers, n_teachers, n_students, teacher_tasks,
                 tasks_remaining, lineage, generation, student_type_accs=None):
        """Called by every worker/student after each cycle. Pushes to Firebase."""
        # Skip student for per-task tracking
        if task != "_student":
            task_accs[task] = {
                "acc": round(acc, 3), "best": round(best_acc, 3),
                "cycle": cycle, "loss": round(loss, 3),
                "exp_id": exp_id,
            }

        try:
            import firebase_push as fb

            # Build leaderboard from current state
            leaderboard = []
            for t in ALL_TASKS:
                if t in teacher_tasks:
                    leaderboard.append({
                        "task": t, "acc": 1.0, "status": "teacher",
                        "exp_id": teacher_tasks[t],
                    })
                elif t in task_accs:
                    ta = task_accs[t]
                    leaderboard.append({
                        "task": t, "acc": ta["acc"], "best": ta["best"],
                        "cycle": ta["cycle"], "status": "training",
                        "exp_id": ta.get("exp_id", ""),
                    })
                else:
                    leaderboard.append({"task": t, "acc": 0, "status": "waiting"})
            leaderboard.sort(key=lambda x: -x["acc"])

            tasks_data = {}
            for entry in leaderboard:
                tasks_data[entry["task"]] = {
                    "acc": entry["acc"],
                    "exp": entry.get("exp_id", "worker"),
                }

            # Clean config for Firebase (remove internal keys)
            clean_cfg = {k: v for k, v in (config or {}).items()
                        if not k.startswith('_')}

            # Full snapshot
            fb._put("mamba3/snapshot", {
                "timestamp": time.time(),
                "mode": "three_populations",
                "gpu_pct": round(gpu_pct, 1),
                "mem_pct": round(vram_pct, 1),
                "cpu_pct": round(cpu_pct, 1),
                "ram_pct": round(ram_pct, 1),
                "n_running": n_workers,
                "n_total": len(ALL_TASKS),
                "best_fresh": round(max((ta["best"] for ta in task_accs.values()), default=0), 4),
                "generation": generation,
                "leaderboard": leaderboard,
                "tasks": tasks_data,
                "lineage": lineage or {},
                "three_pop": {
                    "n_workers": n_workers,
                    "n_teachers": n_teachers,
                    "n_students": n_students,
                    "teachers": teacher_tasks,
                    "tasks_remaining": tasks_remaining,
                    "generation": generation,
                },
            })

            if task != "_student":
                # Per-task timeseries
                fb._put(f"mamba3/task_series/{task}/{cycle}", {
                    "acc": round(acc, 3), "diff": 0,
                })

                # Per-experiment data (individual fields to avoid overwriting cycles)
                fb._put(f"mamba3/experiments/{exp_id}/best_fresh", round(best_acc, 4))
                fb._put(f"mamba3/experiments/{exp_id}/status", "training")
                fb._put(f"mamba3/experiments/{exp_id}/config", clean_cfg)
                fb._put(f"mamba3/experiments/{exp_id}/parent_id", parent_id)
                fb._put(f"mamba3/experiments/{exp_id}/n_params", n_params)
                fb._put(f"mamba3/experiments/{exp_id}/cycle", cycle)
                fb._put(f"mamba3/experiments/{exp_id}/task", task)
                fb._put(f"mamba3/experiments/{exp_id}/cycles/{cycle}", {
                    "fresh": round(acc, 4),
                    "loss": round(loss, 4),
                    "t": time.time(),
                })
            else:
                # Student experiment data
                fb._put(f"mamba3/experiments/student/best_fresh", round(best_acc, 4))
                fb._put(f"mamba3/experiments/student/status", "distilling")
                fb._put(f"mamba3/experiments/student/cycle", cycle)
                fb._put(f"mamba3/experiments/student/n_teachers", n_teachers)
                if student_type_accs:
                    fb._put(f"mamba3/experiments/student/type_accs", {
                        t: round(a, 3) for t, a in student_type_accs.items()
                    })
                fb._put(f"mamba3/experiments/student/cycles/{cycle}", {
                    "fresh": round(acc, 4),
                    "loss": round(loss, 4),
                    "t": time.time(),
                })

            # Append-only history for replay
            fb._put(f"mamba3/history/{int(time.time()*1000)}", {
                "best_fresh": round(max((ta["best"] for ta in task_accs.values()), default=0), 4),
                "gpu_pct": round(gpu_pct, 1),
                "mem_pct": round(vram_pct, 1),
                "cpu_pct": round(cpu_pct, 1),
                "ram_pct": round(ram_pct, 1),
                "n_workers": n_workers,
                "n_teachers": n_teachers,
                "n_students": n_students,
                "tasks": {t: {"acc": round(ta["acc"], 3), "exp": ta.get("exp_id", "worker")}
                         for t, ta in task_accs.items()},
                "teachers": list(teacher_tasks.keys()),
                "worker_best": {t: round(ta["best"], 3) for t, ta in task_accs.items()},
            })

            # Three populations status
            fb._put("mamba3/three_pop", {
                "timestamp": time.time(),
                "generation": generation,
                "n_workers": n_workers,
                "n_teachers": n_teachers,
                "n_students": n_students,
                "teachers": teacher_tasks,
                "tasks_remaining": tasks_remaining,
                "gpu_pct": round(gpu_pct, 1),
                "mem_pct": round(vram_pct, 1),
                "cpu_pct": round(cpu_pct, 1),
                "ram_pct": round(ram_pct, 1),
            })

        except Exception as e:
            print(f"  Firebase error: {e}", flush=True)

    # ── Graduation callback ─────────────────────────────────────────

    def on_graduate(task, exp_id, config):
        """Copy teacher checkpoint to teachers_dir, push mastery event."""
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

    # ── Graceful shutdown ───────────────────────────────────────��───

    pool = None

    def handle_signal(sig, frame):
        print("\nShutting down...", flush=True)
        if pool:
            pool.shutdown()
    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    # ── Launch pool ─────────────────────────────────────────────────

    pool = ResourceAwarePool(
        tasks=ALL_TASKS,
        seed_configs=SEED_CONFIGS,
        device=device,
        on_cycle=on_cycle,
        on_graduate=on_graduate,
        target_acc=0.95,
        teachers_dir=str(teachers_dir),
    )
    pool.start()

    print("Done.", flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="three_pop")
    args = parser.parse_args()
    run(args)
