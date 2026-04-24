#!/usr/bin/env python3
"""Push full node state to Firebase — models catalog, teachers, task progress.

Run on any node to sync its state to the centralized Firebase registry.
Called automatically during training, but can also be run manually.

Usage:
    python server/push_state.py
"""

import json
import os
import socket
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

FIREBASE_URL = "https://signaling-dcfad-default-rtdb.europe-west1.firebasedatabase.app"


def _put(path, data):
    body = json.dumps(data).encode("utf-8")
    req = urllib.request.Request(
        f"{FIREBASE_URL}/{path}.json", data=body, method="PUT",
        headers={"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(req, timeout=5)
        return True
    except Exception as e:
        print(f"  Firebase PUT failed ({path}): {e}")
        return False


def _patch(path, data):
    """PATCH merges into existing data instead of overwriting."""
    body = json.dumps(data).encode("utf-8")
    req = urllib.request.Request(
        f"{FIREBASE_URL}/{path}.json", data=body, method="PATCH",
        headers={"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(req, timeout=5)
        return True
    except Exception as e:
        print(f"  Firebase PATCH failed ({path}): {e}")
        return False


def push_state(db_path="three_pop/training.db"):
    from state_db import StateDB

    db = StateDB(db_path)
    hostname = socket.gethostname()
    node_id = os.environ.get("MAMBA_NODE_ID", hostname)

    # 1. Push task statuses
    all_status = {}
    for t in db.get_all_task_status():
        entry = {
            "best_accuracy": round(t.get("best_accuracy", 0), 4),
            "status": t.get("status", "waiting"),
            "current_stage": t.get("current_stage", 1),
            "confidence_score": t.get("confidence_score", 0),
            "total_cycles": t.get("total_cycles", 0),
            "node": node_id,
            "updated_at": time.time(),
        }
        cfg = t.get("current_config", {})
        if isinstance(cfg, dict):
            entry["config"] = {
                "d_model": cfg.get("d_model"),
                "n_kernel_layers": cfg.get("n_kernel_layers"),
                "device": cfg.get("device", "cpu"),
                "scan_backend": cfg.get("scan_backend", "jit"),
                "lr": cfg.get("lr"),
                "optimizer": cfg.get("optimizer"),
                "loss_fn": cfg.get("loss_fn"),
            }
            if cfg.get("distilled_from"):
                entry["distilled_from"] = str(cfg["distilled_from"])
        all_status[t["task"]] = entry

    # Use PATCH so nodes merge their data instead of overwriting each other
    _patch("mamba3/three_pop/tasks", all_status)
    print(f"  Pushed {len(all_status)} task statuses")

    # 2. Push teachers
    teachers = db.get_teachers()
    teacher_data = {}
    for t, info in teachers.items():
        cfg = info.get("config", {})
        if isinstance(cfg, str):
            try:
                cfg = json.loads(cfg)
            except Exception:
                cfg = {}
        teacher_data[t] = {
            "accuracy": round(info["accuracy"], 4),
            "cycles": info.get("cycles", 0),
            "graduated_at": info.get("graduated_at", 0),
            "node": node_id,
            "checkpoint": info.get("checkpoint_path", ""),
            "config": cfg if isinstance(cfg, dict) else {},
        }
    _patch("mamba3/three_pop/teachers", teacher_data)
    print(f"  Pushed {len(teacher_data)} teachers")

    # 3. Push model catalog — every checkpoint available for inference
    catalog = {}
    ckpt_dir = Path("checkpoints/specialists")
    if ckpt_dir.exists():
        for f in ckpt_dir.glob("*.pt"):
            if any(x in f.name for x in ["_cache", "_champion", "_inspection", "_meta"]):
                continue
            task = f.stem
            catalog[task] = {
                "path": str(f),
                "size_kb": round(f.stat().st_size / 1024, 1),
                "node": node_id,
                "available": True,
                "updated_at": time.time(),
            }
    _patch("mamba3/models", catalog)
    print(f"  Pushed {len(catalog)} models to catalog")

    # 4. Learning rate — how fast are we mastering new tasks?
    teachers_db = db.get_teachers()
    if teachers_db:
        mastery_order = sorted(teachers_db.items(), key=lambda x: x[1].get("graduated_at", 0))
        learning_data = {
            "total_teachers": len(mastery_order),
            "tasks": {},
        }
        prev_time = None
        first_cycles = None
        for i, (task, info) in enumerate(mastery_order):
            cycles = info.get("cycles", 0)
            grad_time = info.get("graduated_at", 0)
            # Skip tasks that mastered in <3 cycles (resumed from checkpoint)
            if first_cycles is None and cycles >= 3:
                first_cycles = cycles
            entry = {
                "order": i + 1,
                "cycles": cycles,
                "graduated_at": grad_time,
            }
            if prev_time and grad_time:
                entry["time_since_prev_s"] = round(grad_time - prev_time, 1)
            if first_cycles and cycles > 0:
                entry["speedup_vs_first"] = round(first_cycles / max(cycles, 1), 2)
            learning_data["tasks"][task] = entry
            prev_time = grad_time

        # Overall learning rate
        if len(mastery_order) >= 2 and first_cycles:
            last_cycles = mastery_order[-1][1].get("cycles", 0) or 1
            learning_data["learning_ratio"] = round(last_cycles / first_cycles, 4)
            learning_data["first_task"] = mastery_order[0][0]
            learning_data["first_cycles"] = first_cycles
            learning_data["last_task"] = mastery_order[-1][0]
            learning_data["last_cycles"] = last_cycles

        _patch("mamba3/learning_rate", learning_data)
        lr = learning_data.get("learning_ratio")
        if lr:
            print(f"  Learning rate: {lr:.3f} (last/first cycles)")

    db.close()
    return len(all_status), len(teacher_data), len(catalog)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="three_pop/training.db")
    args = parser.parse_args()
    push_state(args.db)
