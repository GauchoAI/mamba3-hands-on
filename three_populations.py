"""
Three Populations: Workers → Teachers → Students.

State is sacred:
  - Teachers and lineage stored in SQLite (append-only, never deleted)
  - Runtime config in DB (hot-reload each round, no restart needed)
  - Checkpoints persist across rounds and restarts

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
from datetime import datetime

from coordinator import mutate_config, get_gpu_usage
from state_db import StateDB


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


# ── PID lock ────────────────────────────────────────────────────────

def _acquire_lock():
    lock_path = Path("three_pop.pid")
    if lock_path.exists():
        old_pid = int(lock_path.read_text().strip())
        try:
            os.kill(old_pid, 0)
            print(f"Killing existing instance (PID {old_pid})...", flush=True)
            os.kill(old_pid, signal.SIGTERM)
            time.sleep(2)
            try:
                os.kill(old_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        except ProcessLookupError:
            pass
    lock_path.write_text(str(os.getpid()))
    return lock_path


# ── Main orchestrator ───────────────────────────────────────────────

def run(args):
    lock_path = _acquire_lock()

    base_dir = Path(args.dir)
    teachers_dir = base_dir / "teachers"
    teachers_dir.mkdir(parents=True, exist_ok=True)

    # ── State database (sacred — never deleted) ──
    db = StateDB(str(base_dir / "training.db"))

    # Initialize runtime config defaults (only if not already set)
    if db.get_config("cycles_per_round") is None:
        db.set_config("cycles_per_round", 10)
    if db.get_config("plateau_threshold") is None:
        db.set_config("plateau_threshold", 3)
    if db.get_config("max_concurrent") is None:
        db.set_config("max_concurrent", 4)
    if db.get_config("target_acc") is None:
        db.set_config("target_acc", 0.95)

    print(f"{'='*60}", flush=True)
    print(f"THREE POPULATIONS — Workers → Teachers → Students", flush=True)
    print(f"  State DB: {base_dir / 'training.db'}", flush=True)
    print(f"  Teachers dir: {teachers_dir}", flush=True)
    print(f"{'='*60}\n", flush=True)

    # Load existing teachers from DB (sacred, never re-earned)
    teacher_tasks = db.get_teachers()
    print(f"  Existing teachers: {len(teacher_tasks)}", flush=True)
    for task, info in teacher_tasks.items():
        print(f"    🎓 {task}: {info['accuracy']:.0%} (graduated)", flush=True)

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}\n", flush=True)

    # Graceful shutdown
    should_stop = False
    def handle_signal(sig, frame):
        nonlocal should_stop
        should_stop = True
        print("\nShutting down...", flush=True)
    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    # Per-task state (in memory, backed by DB lineage)
    task_config = {}
    task_best = {}
    task_best_round = {}
    for t in ALL_TASKS:
        if t in teacher_tasks:
            continue
        # Load best config from lineage if exists
        best_cfg, best_acc = db.get_best_config(t)
        task_config[t] = best_cfg if best_cfg else BASE_CONFIG.copy()
        task_best[t] = best_acc
        task_best_round[t] = 0

    from specialist_trainer import train_specialist, load_generators
    load_generators()

    worker_counter = len(teacher_tasks)
    round_num = 0

    # Find the last round number from lineage
    all_lineage = db.get_all_lineage()
    for task, entries in all_lineage.items():
        if entries:
            round_num = max(round_num, entries[-1]["round"])

    tasks_remaining = [t for t in ALL_TASKS if t not in teacher_tasks]

    print(f"  Remaining tasks: {len(tasks_remaining)}", flush=True)
    print(f"  Resuming from round {round_num}", flush=True)
    print(f"  Runtime config: {db.get_all_config()}\n", flush=True)

    # ── Firebase push callback ──────────────────────────────────────
    # Pre-populate with best known values so tasks don't disappear between batches
    task_accs = {}
    for t in ALL_TASKS:
        if t not in teacher_tasks:
            best_cfg, best_acc = db.get_best_config(t)
            if best_acc > 0:
                task_accs[t] = {"acc": round(best_acc, 3), "best": round(best_acc, 3),
                               "cycle": 0, "loss": 0}
                print(f"    Pre-loaded {t}: {best_acc:.0%}", flush=True)

    def on_cycle(task_name, cycle, acc, best, loss):
        task_accs[task_name] = {"acc": round(acc, 3), "best": round(best, 3),
                                "cycle": cycle, "loss": round(loss, 3)}
        try:
            import firebase_push as fb
            leaderboard = []
            for t in ALL_TASKS:
                if t in teacher_tasks:
                    leaderboard.append({"task": t, "acc": 1.0, "status": "teacher",
                                       "exp_id": teacher_tasks[t].get("exp_id", t),
                                       "fresh": 1.0})
                elif t in task_accs:
                    leaderboard.append({"task": t, "acc": task_accs[t]["acc"],
                                       "best": task_accs[t]["best"],
                                       "cycle": task_accs[t]["cycle"],
                                       "exp_id": t, "fresh": task_accs[t]["acc"],
                                       "status": "training"})
                else:
                    leaderboard.append({"task": t, "acc": 0, "status": "waiting"})
            leaderboard.sort(key=lambda x: -x["acc"])

            tasks_data = {e["task"]: {"acc": e["acc"], "exp": e.get("exp_id", "worker")}
                         for e in leaderboard}

            gpu_pct, mem_pct = get_gpu_usage()

            # Build rich lineage from DB
            lineage_data = {}
            try:
                all_lin = db.get_all_lineage()
                for t, entries in all_lin.items():
                    best_so_far = 0
                    for i, e in enumerate(entries):
                        role_char = e.get("role", "champion")[0]
                        node_id = f"{t}_r{e['round']}_{role_char}"
                        parent_id = f"{t}_r{entries[i-1]['round']}_{entries[i-1].get('role','champion')[0]}" if i > 0 else None

                        # Did this entry improve the best?
                        improved = e["accuracy"] > best_so_far
                        if e["accuracy"] > best_so_far:
                            best_so_far = e["accuracy"]

                        # Config summary
                        cfg = e.get("config", {})
                        config_summary = {
                            "d": cfg.get("d_model", 64),
                            "L": cfg.get("n_kernel_layers", 3),
                            "lr": cfg.get("lr", 1e-3),
                            "wd": cfg.get("weight_decay", 0.1),
                            "opt": cfg.get("optimizer", "adamw"),
                            "loss": cfg.get("loss_fn", "ce"),
                        }
                        if cfg.get("use_perp"):
                            config_summary["perp"] = True
                        if cfg.get("warm_restarts"):
                            config_summary["wr"] = True
                        if cfg.get("teacher_model"):
                            config_summary["teacher"] = cfg["teacher_model"]

                        lineage_data[node_id] = {
                            "parent": parent_id,
                            "task": t,
                            "round": e["round"],
                            "acc": round(e["accuracy"], 3),
                            "best": round(best_so_far, 3),
                            "role": e.get("role", "champion"),
                            "won": improved,
                            "config": config_summary,
                            "teachers": e.get("teachers", []),
                            "mutation": e.get("mutation"),
                        }
            except Exception:
                pass

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
                    "teachers": {t: info.get("exp_id", t)
                                for t, info in teacher_tasks.items()},
                    "tasks_remaining": tasks_remaining,
                    "generation": round_num,
                },
            })
            fb._put(f"mamba3/task_series/{task_name}/{cycle}", {
                "acc": round(acc, 3), "diff": 0,
            })
            fb._put(f"mamba3/experiments/{task_name}/best_fresh", round(best, 4))
            fb._put(f"mamba3/experiments/{task_name}/status", "training")
            fb._put(f"mamba3/experiments/{task_name}/cycle", cycle)
            fb._put(f"mamba3/experiments/{task_name}/cycles/{cycle}", {
                "fresh": round(acc, 4), "loss": round(loss, 4), "t": time.time(),
            })
        except Exception:
            pass

    # ── Training loop ───────────────────────────────────────────────

    while tasks_remaining and not should_stop:
        round_num += 1

        # Hot-reload runtime config from DB each round
        cycles_per_round = db.get_config("cycles_per_round", 10)
        plateau_threshold = db.get_config("plateau_threshold", 3)
        max_concurrent = db.get_config("max_concurrent", 4)
        target_acc = db.get_config("target_acc", 0.95)

        print(f"\n{'='*60}", flush=True)
        print(f"Round {round_num} — {len(tasks_remaining)} tasks, "
              f"{len(teacher_tasks)} teachers "
              f"(pool={max_concurrent}, cycles={cycles_per_round})", flush=True)
        print(f"{'='*60}", flush=True)

        # Determine pool size from VRAM
        gpu_pct, mem_pct = get_gpu_usage()
        vram_free_pct = 100 - mem_pct
        pool_size = max(1, min(max_concurrent, int(vram_free_pct / 20)))

        active_tasks = [t for t in tasks_remaining if t not in teacher_tasks]

        for batch_start in range(0, len(active_tasks), pool_size):
            if should_stop:
                break
            batch = active_tasks[batch_start:batch_start + pool_size]

            if len(batch) > 1:
                print(f"\n  Training {len(batch)} tasks concurrently (pool={pool_size})",
                      flush=True)

            # Launch batch as subprocesses
            procs = {}
            for task in batch:
                cfg = task_config.get(task, BASE_CONFIG.copy())
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
                     "--target-acc", str(target_acc)],
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    cwd=str(Path(__file__).parent),
                )
                procs[task] = proc

            # Collect results
            for task, proc in procs.items():
                output = proc.communicate(timeout=600)[0].decode("utf-8", errors="ignore")
                for line in output.strip().split("\n")[-3:]:
                    print(f"    {line}", flush=True)

                cfg = task_config.get(task, BASE_CONFIG.copy())
                ckpt_path = Path("checkpoints/specialists") / f"{task}.pt"
                acc = 0.0
                cycle = 0
                if ckpt_path.exists():
                    try:
                        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                        acc = ckpt.get("accuracy", 0.0)
                        cycle = ckpt.get("cycles", 0)
                    except Exception:
                        pass

                # Track best
                mutation_desc = None
                n_params = 0
                if ckpt_path.exists():
                    try:
                        n_params = torch.load(ckpt_path, map_location="cpu",
                                             weights_only=False).get("n_params", 0)
                    except Exception:
                        pass

                if acc > task_best.get(task, 0):
                    task_best[task] = acc
                    task_best_round[task] = round_num
                    task_config[task] = cfg.copy()

                # Push to Firebase
                try:
                    on_cycle(task, cycle, acc, task_best.get(task, 0), 0)
                except Exception:
                    pass

                # Log to DB (all sacred, append-only)
                db.log_lineage(
                    task=task, round_num=round_num, accuracy=acc,
                    best_accuracy=task_best.get(task, 0), config=cfg,
                    mutation=mutation_desc,
                    checkpoint_path=str(ckpt_path) if ckpt_path.exists() else None,
                )
                db.log_experiment(
                    task=task, round_num=round_num, accuracy=acc,
                    best_accuracy=task_best.get(task, 0), config=cfg,
                    exp_id=f"{task}_r{round_num}", n_params=n_params,
                    cycles=cycle, role="champion",
                    checkpoint_path=str(ckpt_path) if ckpt_path.exists() else None,
                )

                # Export lineage markdown + checkpoint metadata
                lineage_dir = base_dir / "lineage"
                lineage_dir.mkdir(parents=True, exist_ok=True)
                db.export_lineage_markdown(task, lineage_dir / f"{task}.md")
                db.export_checkpoint_metadata(task, "checkpoints/specialists")

                # Check graduation
                if acc >= target_acc:
                    exp_id = f"w_{task}_{worker_counter:03d}"
                    worker_counter += 1

                    # Register in DB (sacred, append-only)
                    db.register_teacher(
                        task=task, accuracy=acc, cycles=cycle,
                        config=cfg, exp_id=exp_id,
                        checkpoint_path=str(ckpt_path),
                    )
                    teacher_tasks[task] = {
                        "exp_id": exp_id, "accuracy": acc,
                    }
                    if task in tasks_remaining:
                        tasks_remaining.remove(task)

                    # Copy checkpoint to teachers dir
                    if ckpt_path.exists():
                        shutil.copy2(ckpt_path, teachers_dir / f"{task}.pt")
                    cache = Path("checkpoints/specialists") / f"{task}_cache.pt"
                    if cache.exists():
                        shutil.copy2(cache, teachers_dir / f"{task}_cache.pt")

                    print(f"  🎓 GRADUATED: {task} → Teacher ({acc:.0%})", flush=True)

                    try:
                        import firebase_push as fb
                        fb.evt_mastery(exp_id, task, 0, 0, 0)
                    except Exception:
                        pass

                # Champion-challenger: if plateaued, try a mutation
                rounds_stuck = round_num - task_best_round.get(task, 0)
                best = task_best.get(task, 0)

                if rounds_stuck >= plateau_threshold and best > 0 and acc < target_acc:
                    severity = min(3.0, rounds_stuck / 3)
                    base_cfg = task_config.get(task, BASE_CONFIG.copy())
                    base_cfg["task"] = task

                    # Run diagnostician — get mutation bias if any signal detected
                    diagnostic_bias = None
                    _diag_signal = None
                    try:
                        from diagnostician import Diagnostician
                        diag = Diagnostician(db)
                        signals = diag.diagnose(task)
                        if signals:
                            _diag_signal = signals[0]
                            rx = diag.prescribe(signals[0], task, base_cfg)
                            if rx:
                                diagnostic_bias = rx
                                # Check if prescription says "protect" or "wait"
                                applied = diag.apply_prescription(base_cfg, rx)
                                if applied is None:
                                    if rx["type"] == "protect":
                                        print(f"  🛡 PROTECTING {task} — {signals[0]['signal']} "
                                              f"(no mutation)", flush=True)
                                    elif rx["type"] == "wait":
                                        print(f"  ⏳ WAITING on {task} — {signals[0]['signal']}",
                                              flush=True)
                                    continue  # skip mutation for this task
                                print(f"  🔬 DIAGNOSTIC for {task}: {signals[0]['signal']} "
                                      f"→ {rx['type']}", flush=True)
                    except Exception as e:
                        print(f"  Diagnostic error: {e}", flush=True)

                    challenger_cfg, challenger_provenance = mutate_config(
                        base_cfg, plateau_severity=severity,
                        diagnostic_bias=diagnostic_bias)
                    challenger_cfg.pop("task", None)

                    changes = {k: challenger_cfg[k] for k in challenger_cfg
                              if challenger_cfg.get(k) != base_cfg.get(k)
                              and k not in ("steps_per_cycle", "task")}
                    mutation_desc = f"severity={severity:.1f} changes={changes}"

                    # Build model card: accumulate teachers from lineage
                    model_card = db.build_model_card(task)
                    inherited_teachers = model_card.get("teachers", [])

                    # If mutation added a new teacher_model, add to teachers list
                    new_teacher = challenger_cfg.pop("teacher_model", None)
                    if new_teacher:
                        # Check cache first
                        cached = db.get_teacher_score(new_teacher, task)
                        if cached is not None and cached <= best:
                            print(f"  Skipping teacher {new_teacher} "
                                  f"(cached: {cached:.0%} <= best: {best:.0%})", flush=True)
                            new_teacher = None
                        else:
                            # Evaluate if not cached
                            if cached is None:
                                try:
                                    from external_teacher import run_experiment
                                    results = run_experiment(new_teacher, tasks=[task],
                                                            n_examples=30)
                                    cached = results.get(task, 0)
                                except Exception:
                                    cached = 0
                            if cached > best:
                                inherited_teachers.append({
                                    "model": new_teacher,
                                    "weight": 1.0,
                                    "from_round": round_num,
                                })
                            else:
                                new_teacher = None

                    # Filter teachers with cached score <= current best
                    active_teachers = []
                    for t in inherited_teachers:
                        t_score = db.get_teacher_score(t["model"], task)
                        if t_score is None or t_score > best * 0.5:
                            active_teachers.append(t)

                    arch_changed = any(challenger_cfg.get(k) != base_cfg.get(k)
                                      for k in ["d_model", "d_state", "headdim",
                                                "n_kernel_layers"])

                    # Back up champion
                    champion_ckpt = Path("checkpoints/specialists") / f"{task}_champion.pt"
                    if ckpt_path.exists():
                        shutil.copy2(ckpt_path, champion_ckpt)
                        if arch_changed:
                            ckpt_path.unlink()

                    if active_teachers:
                        t_names = [t["model"] for t in active_teachers]
                        print(f"  🧬 CHALLENGER for {task} with {len(active_teachers)} "
                              f"teachers {t_names}: {changes}", flush=True)
                    else:
                        print(f"  🧬 CHALLENGER for {task}: {changes}"
                              f"{' (fresh)' if arch_changed else ''}", flush=True)

                    # Train challenger with accumulated teachers
                    challenger_acc = train_specialist(
                        task, challenger_cfg, device,
                        max_cycles=cycles_per_round,
                        target_acc=target_acc,
                        on_cycle=on_cycle,
                        teachers=active_teachers if active_teachers else None,
                    )

                    # Log challenger to DB (sacred, with teachers)
                    db.log_lineage(
                        task=task, round_num=round_num,
                        accuracy=challenger_acc or 0,
                        best_accuracy=max(best, challenger_acc or 0),
                        config=challenger_cfg, mutation=mutation_desc,
                        role="challenger",
                        teachers=active_teachers,
                        provenance=challenger_provenance,
                    )
                    db.log_experiment(
                        task=task, round_num=round_num,
                        accuracy=challenger_acc or 0,
                        best_accuracy=max(best, challenger_acc or 0),
                        config=challenger_cfg,
                        exp_id=f"{task}_r{round_num}_challenger",
                        role="challenger", mutation=mutation_desc,
                        parent_exp=f"{task}_r{round_num}",
                    )

                    won = challenger_acc and challenger_acc > best
                    if won:
                        print(f"  ✓ Challenger wins: {challenger_acc:.0%} > {best:.0%}",
                              flush=True)
                        task_config[task] = challenger_cfg
                        task_best[task] = challenger_acc
                        task_best_round[task] = round_num
                    else:
                        print(f"  ✗ Champion holds: {best:.0%} >= "
                              f"{challenger_acc:.0%}", flush=True)
                        if champion_ckpt.exists():
                            shutil.copy2(champion_ckpt, ckpt_path)
                        task_config[task] = base_cfg

                    # Log diagnostic outcome (if diagnostic was involved)
                    if _diag_signal and diagnostic_bias:
                        db.log_diagnostic(
                            task=task, round_num=round_num,
                            signal=_diag_signal["signal"],
                            prescription_type=diagnostic_bias["type"],
                            prescription_params=diagnostic_bias["params"],
                            challenger_acc=challenger_acc or 0,
                            champion_acc=best,
                            won=won,
                        )

                    db.export_lineage_markdown(task, lineage_dir / f"{task}.md")

        # Sync state to Firebase (redundancy)
        db.sync_to_firebase()

        # Round summary
        gpu_pct, mem_pct = get_gpu_usage()
        print(f"\n[Round {round_num}] Teachers: {len(teacher_tasks)}/{len(ALL_TASKS)} | "
              f"Remaining: {len(tasks_remaining)} | GPU: {gpu_pct:.0f}%", flush=True)
        for task in ALL_TASKS:
            if task in teacher_tasks:
                print(f"    ✅ {task}", flush=True)
            elif task in tasks_remaining:
                best = task_best.get(task, 0)
                print(f"    🔄 {task} (best: {best:.0%})", flush=True)

    # Done
    if not tasks_remaining:
        print(f"\n{'='*60}", flush=True)
        print(f"ALL {len(ALL_TASKS)} TASKS MASTERED!", flush=True)
        print(f"{'='*60}", flush=True)

    lock_path.unlink(missing_ok=True)
    db.close()
    print("Done.", flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="three_pop")
    args = parser.parse_args()
    run(args)
