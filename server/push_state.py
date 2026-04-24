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


def _firebase_get(path):
    try:
        resp = urllib.request.urlopen(f"{FIREBASE_URL}/{path}.json", timeout=5)
        return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None


_bandwidth_bytes = 0  # track bytes sent this session


def _put(path, data):
    global _bandwidth_bytes
    body = json.dumps(data).encode("utf-8")
    _bandwidth_bytes += len(body)
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
    global _bandwidth_bytes
    body = json.dumps(data).encode("utf-8")
    _bandwidth_bytes += len(body)
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

    # Merge with existing Firebase data — only update if our accuracy is higher
    existing_tasks = _firebase_get("mamba3/three_pop/tasks") or {}
    merged = {}
    for task, entry in all_status.items():
        existing = existing_tasks.get(task, {})
        existing_acc = existing.get("best_accuracy", 0)
        our_acc = entry.get("best_accuracy", 0)
        if our_acc >= existing_acc:
            merged[task] = entry
    if merged:
        _patch("mamba3/three_pop/tasks", merged)
    print(f"  Pushed {len(merged)} task statuses (of {len(all_status)}, merged by best accuracy)")

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

    # 5. Schema + bandwidth + usage stats — always up to date in Firebase
    # Count total data
    total_cycles = db.conn.execute("SELECT COUNT(*) FROM cycle_history").fetchone()[0]
    total_lineage = db.conn.execute("SELECT COUNT(*) FROM lineage").fetchone()[0]

    # Build current schema definition
    SCHEMA_VERSION = 4  # bump when schema changes
    current_schema = {
        "nodes": {
            "desc": "Training nodes with heartbeats",
            "path": "/mamba3/nodes/{node_id}",
            "fields": "backends, vram_mb, status, last_heartbeat, ssh, gpu_name",
            "example": {"backends": ["cuda","jit"], "vram_mb": 81079, "status": "online"},
            "added_in": 1,
        },
        "models": {
            "desc": "Checkpoint catalog — every model available for inference",
            "path": "/mamba3/models/{task}",
            "fields": "path, size_kb, node, available, updated_at",
            "example": {"path": "checkpoints/specialists/parity.pt", "size_kb": 208, "node": "h100", "available": True},
            "added_in": 2,
        },
        "teachers": {
            "desc": "Graduated specialists — passed confidence gate",
            "path": "/mamba3/three_pop/teachers/{task}",
            "fields": "accuracy, cycles, node, graduated_at, checkpoint, config",
            "example": {"accuracy": 1.0, "cycles": 25, "node": "h100", "config": {"d_model": 64}},
            "added_in": 1,
        },
        "tasks": {
            "desc": "Per-task training progress (best accuracy wins across nodes)",
            "path": "/mamba3/three_pop/tasks/{task}",
            "fields": "best_accuracy, status, current_stage, confidence_score, node, config, distilled_from",
            "example": {"best_accuracy": 0.96, "status": "training", "current_stage": 2, "distilled_from": "parity.pt"},
            "added_in": 2,
        },
        "learning_rate": {
            "desc": "Meta-learning: cycles-to-mastery per task, speedup ratio",
            "path": "/mamba3/learning_rate",
            "fields": "learning_ratio, first_task, first_cycles, last_task, last_cycles, tasks",
            "example": {"learning_ratio": 0.8, "first_task": "parity", "last_cycles": 20},
            "added_in": 3,
        },
        "meta": {
            "desc": "Schema reference, usage stats, bandwidth tracking (self-documenting)",
            "path": "/mamba3/meta",
            "fields": "schema, usage, bandwidth, version, changelog",
            "example": {"version": SCHEMA_VERSION, "usage": {"total_cycles": 925}},
            "added_in": 4,
        },
    }

    # Check if schema version changed — log to changelog
    existing_meta = _firebase_get("mamba3/meta") or {}
    old_version = existing_meta.get("version", 0)
    changelog = existing_meta.get("changelog", {})
    if old_version != SCHEMA_VERSION:
        # Find what's new
        new_paths = [k for k, v in current_schema.items()
                     if v.get("added_in", 0) > old_version]
        if new_paths:
            changelog[str(SCHEMA_VERSION)] = {
                "timestamp": time.time(),
                "added": new_paths,
                "description": f"Added: {', '.join(new_paths)}",
            }

    _patch("mamba3/meta", {
        "version": SCHEMA_VERSION,
        "schema": current_schema,
        "changelog": changelog,
        "usage": {
            "total_cycles": total_cycles,
            "total_lineage_entries": total_lineage,
            "total_teachers": len(teacher_data),
            "total_models": len(catalog),
            "total_problems": len(all_status),
            "node": node_id,
            "updated_at": time.time(),
        },
        "bandwidth": {
            "last_push_bytes": _bandwidth_bytes,
            "last_push_at": time.time(),
            "node": node_id,
            "estimated_per_push_kb": round(_bandwidth_bytes / 1024, 1),
            "estimated_daily_mb": round(_bandwidth_bytes * 2 * 12 * 24 / (1024 * 1024), 1),
            "estimated_monthly_gb": round(_bandwidth_bytes * 2 * 12 * 24 * 30 / (1024 * 1024 * 1024), 2),
            "limit_gb": 10,
        },
    })

    db.close()
    return len(all_status), len(teacher_data), len(catalog)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="three_pop/training.db")
    args = parser.parse_args()
    push_state(args.db)
