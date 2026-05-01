"""
Three Populations: Workers → Teachers → Students.

STATELESS ORCHESTRATOR — all state in SQLite + checkpoints.
Kill and replace at any time. Workers are self-sufficient.

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
import subprocess
import shutil
import torch
from pathlib import Path

from coordinator import mutate_config, get_gpu_usage
from state_db import StateDB
from registry.problem_registry import ProblemRegistry

# Discover tasks from YAML manifests — no hardcoded list
_problems_dir = "problems"  # default, overridden by --problems-dir CLI arg
_db_path = "three_pop/training.db"  # default, derived from --dir CLI arg

def _set_db_path(path):
    global _db_path
    _db_path = path
_problem_registry = ProblemRegistry()
_problem_registry.discover([_problems_dir])
ALL_TASKS = _problem_registry.list_problems()


def reload_problems(problems_dir="problems"):
    """Re-discover problems from a different directory."""
    global _problems_dir, ALL_TASKS
    _problems_dir = problems_dir
    _problem_registry.problems.clear()
    _problem_registry._generators.clear()
    _problem_registry.discover([problems_dir])
    ALL_TASKS[:] = _problem_registry.list_problems()

import platform as _platform

# Lighter base config for CPU-only nodes (Apple Silicon, etc.)
if _platform.system() == "Darwin":
    BASE_CONFIG = {
        "d_model": 32, "d_state": 16, "headdim": 16, "n_kernel_layers": 1,
        "batch_size": 128, "lr": 1e-3, "weight_decay": 0.1,
        "steps_per_cycle": 100, "loss_fn": "ce", "optimizer": "adamw",
        "use_perp": False, "device": "cpu", "scan_backend": "jit",
    }
else:
    BASE_CONFIG = {
        "d_model": 64, "d_state": 16, "headdim": 16, "n_kernel_layers": 3,
        "batch_size": 256, "lr": 1e-3, "weight_decay": 0.1,
        "steps_per_cycle": 200, "loss_fn": "ce", "optimizer": "adamw",
        "use_perp": False,
    }


def _acquire_lock():
    lock_path = Path("three_pop.pid")
    if lock_path.exists():
        try:
            old_pid = int(lock_path.read_text().strip())
            os.kill(old_pid, 0)  # check if alive
            # Don't kill — just warn. The old instance will exit on its own
            # or the user can kill it manually. Prevents accidental kills on deploy.
            print(f"Warning: existing instance (PID {old_pid}) still running.", flush=True)
            print(f"  Kill it first: kill {old_pid}", flush=True)
            sys.exit(1)
        except (ProcessLookupError, ValueError):
            pass  # old process is dead, safe to take over
    lock_path.write_text(str(os.getpid()))
    return lock_path


def spawn_worker(task, config, mode="champion", cycles=10, target_acc=0.95):
    """Spawn a specialist_trainer subprocess. Returns Popen.

    Routes to specialist_trainer.py — the PyTorch+MPS engine. The
    NVIDIA PTX path (ptxd_specialist) lives on the `pod-archive`
    branch; this branch (Mac-focused) uses one engine end to end.
    """
    cfg = config if isinstance(config, dict) else {}
    worker_script = "specialist_trainer.py"
    cmd = [sys.executable, "-u", worker_script,
         "--task", task,
         "--mode", mode,
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
         "--max-cycles", str(cycles),
         "--target-acc", str(target_acc)]
    if cfg.get("scan_backend"):
        cmd.extend(["--scan-backend", cfg["scan_backend"]])
    # Device: use config value, or default to cpu on Apple Silicon (MPS has precision issues)
    device = cfg.get("device")
    if not device:
        import platform
        if platform.system() == "Darwin":
            device = "cpu"
    if device:
        cmd.extend(["--device", device])
    if _problems_dir != "problems":
        cmd.extend(["--problems-dir", _problems_dir])
    if _db_path != "three_pop/training.db":
        cmd.extend(["--db-path", _db_path])
    return subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        cwd=str(Path(__file__).parent),
    )


def run(args):
    lock_path = _acquire_lock()
    db = StateDB(str(Path(args.dir) / "training.db"))
    teachers_dir = Path(args.dir) / "teachers"
    teachers_dir.mkdir(parents=True, exist_ok=True)
    lineage_dir = Path(args.dir) / "lineage"
    lineage_dir.mkdir(parents=True, exist_ok=True)

    # Initialize task_status for any missing tasks
    for t in ALL_TASKS:
        if not db.get_task_status(t):
            if db.is_teacher(t):
                db.update_task_status(t, "mastered")
            else:
                best_cfg, best_acc = db.get_best_config(t)
                db.update_task_status(t, "training" if best_acc > 0 else "waiting",
                                      best_cfg or BASE_CONFIG, best_acc)

    # Runtime config defaults
    if db.get_config("cycles_per_round") is None:
        db.set_config("cycles_per_round", 10)
    if db.get_config("plateau_threshold") is None:
        db.set_config("plateau_threshold", 3)
    if db.get_config("max_concurrent") is None:
        db.set_config("max_concurrent", 4)
    if db.get_config("target_acc") is None:
        db.set_config("target_acc", 0.95)

    teachers = db.get_teachers()
    print(f"{'='*60}", flush=True)
    print(f"STATELESS ORCHESTRATOR", flush=True)
    print(f"  DB: {db.db_path}", flush=True)
    print(f"  Teachers: {len(teachers)}", flush=True)
    print(f"  Tasks: {len(ALL_TASKS)}", flush=True)
    print(f"{'='*60}\n", flush=True)

    should_stop = False
    def handle_signal(sig, frame):
        nonlocal should_stop
        should_stop = True
        print("\nShutting down...", flush=True)
    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    round_num = 0

    while not should_stop:
        # ── Read everything from DB (stateless) ──
        cfg = db.get_all_config()
        cycles_per_round = cfg.get("cycles_per_round", 10)
        plateau_threshold = cfg.get("plateau_threshold", 3)
        max_concurrent = cfg.get("max_concurrent", 4)
        target_acc = cfg.get("target_acc", 0.95)

        tasks_needing_training = db.get_tasks_needing_training()
        if not tasks_needing_training:
            print("All tasks mastered!", flush=True)
            break

        round_num += 1
        gpu_pct, mem_pct = get_gpu_usage()
        vram_free = 100 - mem_pct
        pool_size = max(1, min(max_concurrent, int(vram_free / 20)))

        print(f"\n{'='*60}", flush=True)
        print(f"Round {round_num} — {len(tasks_needing_training)} tasks, "
              f"{len(db.get_teachers())} teachers (pool={pool_size})", flush=True)
        print(f"{'='*60}", flush=True)

        # ── Train in batches ──
        for batch_start in range(0, len(tasks_needing_training), pool_size):
            if should_stop:
                break
            batch = tasks_needing_training[batch_start:batch_start + pool_size]

            # Spawn champions
            procs = {}
            for task in batch:
                status = db.get_task_status(task)
                task_cfg = (status["current_config"] if status and status["current_config"]
                           else BASE_CONFIG.copy())
                print(f"  + {task} (d={task_cfg.get('d_model')} "
                      f"L={task_cfg.get('n_kernel_layers')} "
                      f"lr={task_cfg.get('lr', 1e-3):.0e})", flush=True)
                procs[task] = spawn_worker(task, task_cfg, "champion",
                                          cycles_per_round, target_acc)

            # Wait for champions
            for task, proc in procs.items():
                try:
                    output = proc.communicate(timeout=None)[0].decode("utf-8", errors="ignore")
                    for line in output.strip().split("\n")[-3:]:
                        print(f"    {line}", flush=True)
                except Exception as e:
                    print(f"    {task} error: {e}", flush=True)

            # Workers push their own Firebase data during training

            # ── Challengers for plateaued tasks ──
            for task in batch:
                if should_stop:
                    break
                status = db.get_task_status(task)
                if not status or status["status"] == "mastered":
                    continue

                best = status["best_accuracy"]
                # Simple plateau check: has this task improved recently?
                lineage = db.get_lineage(task)
                if len(lineage) < plateau_threshold:
                    continue
                recent_bests = [e["best_accuracy"] for e in lineage[-plateau_threshold:]]
                if max(recent_bests) > recent_bests[0]:
                    continue  # still improving

                # Read diagnostic signals (written by workers)
                signals = status.get("diagnostic_signals", [])
                diagnostic_bias = None
                if signals:
                    try:
                        from diagnostician import Diagnostician
                        diag = Diagnostician(db)
                        rx = diag.prescribe(signals[0], task, status["current_config"])
                        if rx:
                            if rx["type"] in ("protect", "wait"):
                                print(f"  🛡 {task}: {signals[0]['signal']} → {rx['type']}", flush=True)
                                continue
                            diagnostic_bias = rx
                            print(f"  🔬 {task}: {signals[0]['signal']} → {rx['type']}", flush=True)
                    except Exception:
                        pass

                # Mutate from best config
                severity = min(3.0, len(lineage) / 10)
                base_cfg = status["current_config"] or BASE_CONFIG.copy()
                base_cfg["task"] = task
                challenger_cfg, provenance = mutate_config(
                    base_cfg, plateau_severity=severity,
                    diagnostic_bias=diagnostic_bias)
                challenger_cfg.pop("task", None)

                changes = {k: challenger_cfg[k] for k in challenger_cfg
                          if challenger_cfg.get(k) != base_cfg.get(k)
                          and k not in ("steps_per_cycle", "task")}
                if not changes:
                    continue

                # Back up champion checkpoint
                ckpt = Path("checkpoints/specialists") / f"{task}.pt"
                champ_ckpt = Path("checkpoints/specialists") / f"{task}_champion.pt"
                if ckpt.exists():
                    shutil.copy2(ckpt, champ_ckpt)
                    arch_changed = any(challenger_cfg.get(k) != base_cfg.get(k)
                                      for k in ["d_model", "d_state", "headdim", "n_kernel_layers"])
                    if arch_changed:
                        ckpt.unlink()

                print(f"  🧬 Challenger {task}: {changes}", flush=True)
                cproc = spawn_worker(task, challenger_cfg, "challenger",
                                    cycles_per_round, target_acc)
                try:
                    output = cproc.communicate(timeout=None)[0].decode("utf-8", errors="ignore")
                    for line in output.strip().split("\n")[-2:]:
                        print(f"    {line}", flush=True)
                except Exception:
                    pass

                # Log to lineage with provenance
                status_after = db.get_task_status(task)
                acc_after = status_after["best_accuracy"] if status_after else 0
                db.log_lineage(task, round_num, acc_after, acc_after,
                              challenger_cfg, role="challenger",
                              mutation=f"severity={severity:.1f} changes={changes}",
                              provenance=provenance)
                db.export_lineage_markdown(task, lineage_dir / f"{task}.md")

        # ── Sync to Firebase ──
        db.sync_to_firebase()
        try:
            from server.push_state import push_state
            push_state(_db_path)
        except Exception:
            pass

        # ── Round summary ──
        all_status = db.get_all_task_status()
        teachers = db.get_teachers()
        gpu_pct, mem_pct = get_gpu_usage()
        print(f"\n[Round {round_num}] Teachers: {len(teachers)}/{len(ALL_TASKS)} | "
              f"GPU: {gpu_pct:.0f}%", flush=True)
        for s in all_status:
            icon = "✅" if s["status"] == "mastered" else "🔄"
            print(f"    {icon} {s['task']} ({s['best_accuracy']:.0%})", flush=True)

        # Push Firebase snapshot
        try:
            from lab_platform import firebase_push as fb
            leaderboard = []
            for s in all_status:
                entry = {"task": s["task"], "acc": s["best_accuracy"],
                         "status": s["status"], "exp_id": s["task"],
                         "fresh": s["best_accuracy"]}
                if s["status"] == "mastered":
                    entry["acc"] = 1.0
                    entry["fresh"] = 1.0
                    t_info = teachers.get(s["task"], {})
                    entry["exp_id"] = t_info.get("exp_id", s["task"])
                leaderboard.append(entry)
            leaderboard.sort(key=lambda x: -x["acc"])

            tasks_data = {e["task"]: {"acc": e["acc"], "exp": e["exp_id"]}
                         for e in leaderboard}

            fb._put("mamba3/snapshot", {
                "timestamp": time.time(),
                "mode": "three_populations",
                "gpu_pct": round(gpu_pct, 1),
                "mem_pct": round(mem_pct, 1),
                "n_running": len(tasks_needing_training),
                "n_total": len(ALL_TASKS),
                "best_fresh": 0,
                "leaderboard": leaderboard,
                "tasks": tasks_data,
                "three_pop": {
                    "n_workers": len(tasks_needing_training),
                    "n_teachers": len(teachers),
                    "n_students": 0,
                    "teachers": {t: info.get("exp_id", t) for t, info in teachers.items()},
                    "tasks_remaining": tasks_needing_training,
                    "generation": round_num,
                },
            })
        except Exception:
            pass

    lock_path.unlink(missing_ok=True)
    db.close()
    print("Done.", flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="three_pop")
    parser.add_argument("--problems-dir", default="problems",
                       help="Directory containing problem YAML manifests")
    parser.add_argument("--job-id", default=None,
                       help="Job ID for tracking (set by lab submit)")
    args = parser.parse_args()

    # Re-discover problems if custom dir specified
    if args.problems_dir != "problems":
        reload_problems(args.problems_dir)

    # Set DB path so workers use the same DB as orchestrator
    _set_db_path(str(Path(args.dir) / "training.db"))

    run(args)
