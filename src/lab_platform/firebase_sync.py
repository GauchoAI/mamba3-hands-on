"""
Firebase sync — reads three_pop state + SQLite, pushes clean data.

Only pushes three_pop experiments, not old multi-task ones.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import json
import time
import argparse
import urllib.request
from datetime import datetime
from pathlib import Path

from . import firebase_push as fb

FIREBASE_URL = "https://signaling-dcfad-default-rtdb.europe-west1.firebasedatabase.app"


def read_three_pop():
    """Read three_pop state from Firebase."""
    try:
        resp = urllib.request.urlopen(f"{FIREBASE_URL}/mamba3/three_pop.json", timeout=5)
        return json.loads(resp.read())
    except Exception:
        return None


def read_specialist_logs(base_dir="three_pop/workers"):
    """Read latest accuracy from specialist worker logs."""
    workers = {}
    base = Path(base_dir)
    if not base.exists():
        return workers

    for d in sorted(base.iterdir()):
        if not d.is_dir():
            continue
        log = d / "stdout.log"
        if not log.exists():
            continue
        # Parse task from dir name: w_parity_000 → parity
        parts = d.name.split("_")
        if len(parts) >= 3:
            task = "_".join(parts[1:-1])
        else:
            continue

        # Find last acc= line
        try:
            content = log.read_text()
            acc_lines = [l for l in content.split("\n") if "acc=" in l]
            if acc_lines:
                last = acc_lines[-1]
                # Parse: [parity] cycle  10  loss=0.350  acc=53%
                acc_str = last.split("acc=")[1].split("%")[0]
                acc = int(acc_str) / 100
                cycle_str = last.split("cycle")[1].strip().split()[0]
                cycle = int(cycle_str)
                workers.setdefault(task, []).append({
                    "exp_id": d.name, "acc": acc, "cycle": cycle,
                })
        except Exception:
            continue

    return workers


def sync_once():
    t0 = time.time()

    tp = read_three_pop()
    workers_data = read_specialist_logs()

    if not tp:
        print(f"  no three_pop data yet", flush=True)
        return

    teachers = tp.get("teachers", {})
    remaining = tp.get("tasks_remaining", [])

    # Build worker leaderboard (per-task best)
    worker_leaderboard = []
    for task, exps in sorted(workers_data.items()):
        best = max(exps, key=lambda x: x["acc"])
        worker_leaderboard.append({
            "exp_id": best["exp_id"],
            "task": task,
            "acc": best["acc"],
            "cycle": best["cycle"],
            "status": "graduated" if task in teachers else "training",
            "n_variants": len(exps),
        })
    worker_leaderboard.sort(key=lambda x: -x["acc"])

    # Build teacher leaderboard
    teacher_leaderboard = []
    for task, exp_id in teachers.items():
        teacher_leaderboard.append({
            "exp_id": exp_id,
            "task": task,
            "acc": 1.0,
            "status": "teaching",
        })

    # Push clean snapshot
    snapshot = {
        "timestamp": time.time(),
        "mode": "three_populations",
        "gpu_pct": tp.get("gpu_pct", 0),
        "mem_pct": tp.get("mem_pct", 0),
        "n_running": tp.get("n_workers", 0),
        "n_total": tp.get("n_workers", 0) + tp.get("n_teachers", 0) + tp.get("n_students", 0),
        "best_fresh": 0,  # student's fresh, once available
        "three_pop": tp,
        "leaderboard": worker_leaderboard,
        "teacher_leaderboard": teacher_leaderboard,
        "tasks": {
            t: {"acc": 1.0, "exp": teachers[t]} for t in teachers
        },
    }

    # Add task accs from workers
    for w in worker_leaderboard:
        if w["task"] not in snapshot["tasks"]:
            snapshot["tasks"][w["task"]] = {"acc": round(w["acc"], 3), "exp": w["exp_id"]}

    fb._put("mamba3/snapshot", snapshot)

    # History
    ts_key = str(int(time.time()))
    fb._put(f"mamba3/history/{ts_key}", {
        "n_workers": tp.get("n_workers", 0),
        "n_teachers": tp.get("n_teachers", 0),
        "n_students": tp.get("n_students", 0),
        "teachers": list(teachers.keys()),
        "worker_best": {w["task"]: round(w["acc"], 2) for w in worker_leaderboard[:10]},
    })

    elapsed = time.time() - t0
    print(f"  synced at {datetime.now().strftime('%H:%M:%S')} ({elapsed:.1f}s) "
          f"— {len(worker_leaderboard)} workers, {len(teachers)} teachers, "
          f"mode=three_pop", flush=True)


def watch(interval=15):
    print(f"Firebase sync (three_pop mode) every {interval}s...", flush=True)
    while True:
        try:
            sync_once()
        except Exception as e:
            import traceback
            print(f"  ❌ Sync error: {e}", flush=True)
            traceback.print_exc()
        time.sleep(interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="metrics.db")  # kept for compat
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=int, default=15)
    args = parser.parse_args()
    if args.watch:
        watch(args.interval)
    else:
        sync_once()
