"""
Push training metrics to Firebase Realtime DB.

Zero dependencies beyond urllib (stdlib). Called by coordinator each generation.
"""
import json
import time
import urllib.request
import urllib.error

FIREBASE_URL = "https://signaling-dcfad-default-rtdb.europe-west1.firebasedatabase.app"


def push(path, data):
    """PUT data to Firebase path."""
    url = f"{FIREBASE_URL}/{path}.json"
    body = json.dumps(data).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="PUT",
                                headers={"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(req, timeout=5)
        return True
    except Exception as e:
        print(f"  ⚠ Firebase push failed: {e}", flush=True)
        return False


def append(path, data):
    """POST data to Firebase path (auto-generates key)."""
    url = f"{FIREBASE_URL}/{path}.json"
    body = json.dumps(data).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST",
                                headers={"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(req, timeout=5)
        return True
    except Exception as e:
        print(f"  ⚠ Firebase append failed: {e}", flush=True)
        return False


def push_snapshot(results, generation, gpu_pct, mem_pct, evo_state=None):
    """Push current state snapshot — ALL data the UI could ever need."""
    # Full leaderboard (not just top 10)
    leaderboard = []
    for r in results[:30]:
        cfg = r.get("config", {})
        parent = cfg.get("_parent_id")
        leaderboard.append({
            "exp_id": r["exp_id"],
            "fresh": round(r.get("best_fresh", 0), 4),
            "effective_score": round(r.get("effective_score", 0), 4),
            "type_accs": {k: round(v, 3) for k, v in r.get("type_accs", {}).items() if v > 0},
            "cycle": r.get("cycle", 0),
            "d_model": cfg.get("d_model"),
            "d_state": cfg.get("d_state"),
            "n_layers": cfg.get("n_kernel_layers"),
            "batch_size": cfg.get("batch_size"),
            "lr": cfg.get("lr"),
            "weight_decay": cfg.get("weight_decay", 0),
            "optimizer": cfg.get("optimizer", "adamw"),
            "loss_fn": cfg.get("loss_fn", "stable_ce"),
            "warm_restarts": cfg.get("warm_restarts", False),
            "noise_scale": cfg.get("noise_scale", 0),
            "backend": cfg.get("backend", "pytorch"),
            "method": f"wd={cfg.get('weight_decay')}" if cfg.get("weight_decay", 0) > 0 else "PerpGrad",
            "status": r["status"],
            "momentum": round(r.get("momentum", 0), 4),
            "specialist": r.get("specialist_for", [])[:3],
            "parent_id": parent,
            "n_params": r.get("params", 0),
        })

    # Best per task
    task_best = {}
    for r in results:
        for task, acc in r.get("type_accs", {}).items():
            if task not in task_best or acc > task_best[task]["acc"]:
                task_best[task] = {"acc": round(acc, 3), "exp": r["exp_id"]}

    snapshot = {
        "timestamp": time.time(),
        "generation": generation,
        "gpu_pct": round(gpu_pct, 1),
        "mem_pct": round(mem_pct, 1),
        "n_running": sum(1 for r in results if r["status"] == "running"),
        "n_total": len(results),
        "best_fresh": round(results[0].get("best_fresh", 0), 4) if results else 0,
        "leaderboard": top10,
        "tasks": task_best,
    }

    if evo_state:
        snapshot["plateau"] = {
            "active": evo_state.plateau_mode,
            "severity": round(evo_state.get_plateau_severity(), 2),
            "best_ever": round(evo_state.best_ever, 4),
            "stuck_gens": evo_state.generation - evo_state.best_ever_gen,
        }
        # Family tree — full lineage
        snapshot["lineage"] = {
            eid: info for eid, info in evo_state.lineage.items()
        }

    return push("mamba3/snapshot", snapshot)


def push_gpu_tick(gpu_pct, mem_pct, n_workers, generation):
    """Push GPU timeseries point (for historical charts)."""
    return push(f"mamba3/gpu_history/{generation}", {
        "gpu": round(gpu_pct, 1),
        "mem": round(mem_pct, 1),
        "workers": n_workers,
        "t": time.time(),
    })


def push_event(event_type, exp_id=None, details=None):
    """Push a real-time event."""
    return append("mamba3/events", {
        "type": event_type,
        "exp_id": exp_id,
        "details": details,
        "timestamp": time.time(),
    })


def push_experiment_cycle(exp_id, cycle, fresh, loss, type_accs):
    """Push per-experiment cycle data for timeseries."""
    return push(f"mamba3/experiments/{exp_id}/cycles/{cycle}", {
        "fresh": round(fresh, 4),
        "loss": round(loss, 4),
        "tasks": {k: round(v, 3) for k, v in type_accs.items() if v > 0},
        "t": time.time(),
    })


# ── Test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Testing Firebase push...")
    ok = push("mamba3/test", {"hello": "world", "time": time.time()})
    print(f"Push: {'OK' if ok else 'FAILED'}")

    ok = push_event("test", "exp_000", "testing firebase connection")
    print(f"Event: {'OK' if ok else 'FAILED'}")

    # Read it back
    import urllib.request
    resp = urllib.request.urlopen(f"{FIREBASE_URL}/mamba3/test.json")
    data = json.loads(resp.read())
    print(f"Read back: {data}")

    # Cleanup
    push("mamba3/test", None)
    print("Cleaned up. Done.")
