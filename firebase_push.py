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
    """Push current state snapshot."""
    top10 = []
    for r in results[:10]:
        cfg = r.get("config", {})
        top10.append({
            "exp_id": r["exp_id"],
            "fresh": round(r.get("best_fresh", 0), 4),
            "parity": round(r.get("type_accs", {}).get("parity", 0), 2),
            "cycle": r.get("cycle", 0),
            "d_model": cfg.get("d_model"),
            "n_layers": cfg.get("n_kernel_layers"),
            "method": f"wd={cfg.get('weight_decay')}" if cfg.get("weight_decay", 0) > 0 else "PerpGrad",
            "status": r["status"],
            "momentum": round(r.get("momentum", 0), 4),
            "specialist": r.get("specialist_for", [])[:3],
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

    return push("mamba3/snapshot", snapshot)


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
