"""
Firebase sync — reads from SQLite, pushes to Firebase with debouncing.

Like Debezium: captures changes from the local DB and syncs them.
One process, controlled rate, no flooding.

Workers write to SQLite (fast, local).
This script reads SQLite and pushes snapshots to Firebase (debounced).

Usage:
    python firebase_sync.py                    # one-shot sync
    python firebase_sync.py --watch --interval 30  # continuous sync
"""
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import json
import time
import argparse
from datetime import datetime

from metrics_db import MetricsReader
import firebase_push as fb


def sync_once(reader):
    """Read SQLite, push snapshot + timeseries to Firebase."""
    t0 = time.time()

    experiments = reader.get_experiments()
    active_tasks = reader.get_active_tasks()
    gpu_history = reader.get_gpu_history(limit=1)
    events = reader.get_events(limit=20)
    teacher = reader.get_latest_teacher()

    gpu_pct = gpu_history[-1]["gpu_pct"] if gpu_history else 0
    mem_pct = gpu_history[-1]["mem_pct"] if gpu_history else 0
    n_workers = gpu_history[-1]["n_workers"] if gpu_history else 0

    # ── Build per-experiment type_accs from latest task data ──
    all_task_rows = reader.get_all_task_latest()
    exp_type_accs = {}  # exp_id → {task: acc}
    for t in all_task_rows:
        eid = t["exp_id"]
        if eid not in exp_type_accs:
            exp_type_accs[eid] = {}
        exp_type_accs[eid][t["task_type"]] = round(t["accuracy"], 3)

    # ── Snapshot ──
    leaderboard = []
    for exp in experiments[:30]:
        cfg = json.loads(exp["config_json"]) if exp.get("config_json") else {}
        eid = exp["exp_id"]
        leaderboard.append({
            "exp_id": eid,
            "fresh": round(exp.get("peak_fresh", 0) or 0, 4),
            "cycle": exp.get("max_cycle", 0) or 0,
            "d_model": exp.get("d_model"),
            "d_state": exp.get("d_state"),
            "n_layers": exp.get("n_kernel_layers"),
            "n_params": exp.get("n_params", 0),
            "weight_decay": cfg.get("weight_decay", 0),
            "optimizer": cfg.get("optimizer", "adamw"),
            "loss_fn": cfg.get("loss_fn", "stable_ce"),
            "warm_restarts": cfg.get("warm_restarts", False),
            "noise_scale": cfg.get("noise_scale", 0),
            "backend": cfg.get("backend", "pytorch"),
            "method": f"wd={cfg.get('weight_decay')}" if cfg.get("weight_decay", 0) > 0 else "PerpGrad",
            "status": exp["status"],
            "parent_id": cfg.get("_parent_id"),
            "lr": cfg.get("lr"),
            "batch_size": cfg.get("batch_size"),
            "type_accs": exp_type_accs.get(eid, {}),
        })

    # Best per task
    all_task_data = reader.get_all_task_latest()
    task_best = {}
    for t in all_task_data:
        tt = t["task_type"]
        if tt not in task_best or t["accuracy"] > task_best[tt]["acc"]:
            task_best[tt] = {"acc": round(t["accuracy"], 3), "exp": t["exp_id"]}

    snapshot = {
        "timestamp": time.time(),
        "gpu_pct": round(gpu_pct, 1),
        "mem_pct": round(mem_pct, 1),
        "n_running": sum(1 for e in experiments if e["status"] == "running"),
        "n_paused": sum(1 for e in experiments if e["status"] == "paused"),
        "n_total": len(experiments),
        "n_workers": n_workers,
        "best_fresh": round(experiments[0].get("peak_fresh", 0) or 0, 4) if experiments else 0,
        "leaderboard": leaderboard,
        "tasks": task_best,
    }

    # Teacher
    if teacher:
        snapshot["teacher"] = {
            "status": teacher.get("status_text", ""),
            "unlocked": json.loads(teacher.get("unlocked_tasks", "[]")),
            "mastery_log": json.loads(teacher.get("mastery_log", "[]")),
        }

    # Build lineage from experiment configs
    lineage = {}
    for exp in experiments[:30]:
        cfg = json.loads(exp["config_json"]) if exp.get("config_json") else {}
        pid = cfg.get("_parent_id")
        if pid:
            lineage[exp["exp_id"]] = {"parent": pid}
    snapshot["lineage"] = lineage

    fb._put("mamba3/snapshot", snapshot)

    # APPEND to history — never overwrite. This is the replayable event store.
    # Every snapshot is preserved with a timestamp key.
    ts_key = str(int(time.time()))
    fb._put(f"mamba3/history/{ts_key}", {
        "best_fresh": snapshot["best_fresh"],
        "gpu_pct": snapshot["gpu_pct"],
        "mem_pct": snapshot["mem_pct"],
        "n_running": snapshot["n_running"],
        "n_total": snapshot["n_total"],
        "n_workers": snapshot.get("n_workers", 0),
        "tasks": snapshot["tasks"],
        "top3": [{k: v for k, v in e.items() if k in ("exp_id","fresh","method","d_model","n_layers")}
                 for e in snapshot["leaderboard"][:3]],
    })

    # ── Per-experiment timeseries (top 8) ──
    for exp in experiments[:8]:
        eid = exp["exp_id"]
        history = reader.get_cycle_history(eid)
        if not history:
            continue
        # Push last 100 cycles max
        cycles_data = {}
        for h in history[-100:]:
            cycles_data[str(h["cycle"])] = {
                "fresh": round(h["fresh_acc"] or 0, 4),
                "loss": round(h["loss"] or 0, 4),
                "t": h.get("timestamp", 0),
            }
        cfg = json.loads(exp["config_json"]) if exp.get("config_json") else {}
        fb._put(f"mamba3/experiments/{eid}", {
            "config": cfg,
            "status": exp["status"],
            "cycle": exp.get("max_cycle", 0) or 0,
            "best_fresh": round(exp.get("peak_fresh", 0) or 0, 4),
            "n_params": exp.get("n_params", 0),
            "cycles": cycles_data,
        })

    # ── Per-task timeseries from top experiment ──
    if experiments:
        best = experiments[0]
        for task in active_tasks:
            history = reader.get_task_history(best["exp_id"], task)
            if history:
                task_ts = {}
                for h in history[-100:]:
                    task_ts[str(h["cycle"])] = {
                        "acc": round(h["accuracy"], 3),
                        "diff": round(h.get("difficulty", 0), 3),
                    }
                fb._put(f"mamba3/task_series/{task}", task_ts)

    # ── Events (last 20) ──
    if events:
        events_data = {}
        for e in events[-20:]:
            key = str(int(e["timestamp"] * 1000))
            events_data[key] = {
                "type": e["event_type"],
                "exp_id": e.get("exp_id"),
                "details": e.get("details"),
                "timestamp": e["timestamp"],
            }
        fb._put("mamba3/events", events_data)

    elapsed = time.time() - t0
    print(f"  synced at {datetime.now().strftime('%H:%M:%S')} ({elapsed:.1f}s) "
          f"— {len(leaderboard)} exps, {len(task_best)} tasks, {len(events)} events",
          flush=True)


def watch(db_path="metrics.db", interval=30):
    print(f"Firebase sync watching {db_path} every {interval}s...", flush=True)
    reader = MetricsReader(db_path)
    while True:
        try:
            sync_once(reader)
        except Exception as e:
            import traceback
            print(f"  ❌ Sync error: {e}", flush=True)
            traceback.print_exc()
        time.sleep(interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="metrics.db")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=int, default=30)
    args = parser.parse_args()

    if args.watch:
        watch(args.db, args.interval)
    else:
        reader = MetricsReader(args.db)
        sync_once(reader)
        print("Done.")
