"""
Push ALL training telemetry to Firebase Realtime DB.

Zero dependencies beyond urllib (stdlib). Every event, every metric,
every state change — all pushed for the real-time dashboard.

See docs/EVENT_CATALOG.md for the full event specification.
"""
import json
import time
import urllib.request
import urllib.error

FIREBASE_URL = "https://signaling-dcfad-default-rtdb.europe-west1.firebasedatabase.app"


def _put(path, data):
    url = f"{FIREBASE_URL}/{path}.json"
    body = json.dumps(data).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="PUT",
                                headers={"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(req, timeout=5)
        return True
    except Exception:
        return False


def _post(path, data):
    url = f"{FIREBASE_URL}/{path}.json"
    body = json.dumps(data).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST",
                                headers={"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(req, timeout=5)
        return True
    except Exception:
        return False


# ── Snapshot (every generation) ─────────────────────────────────────

def push_snapshot(results, generation, gpu_pct, mem_pct, evo_state=None):
    """Push complete world state."""
    leaderboard = []
    for r in results[:30]:
        cfg = r.get("config", {})
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
            "parent_id": cfg.get("_parent_id"),
            "n_params": r.get("params", 0),
        })

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
        "n_paused": sum(1 for r in results if r["status"] == "paused"),
        "n_total": len(results),
        "best_fresh": round(results[0].get("best_fresh", 0), 4) if results else 0,
        "best_exp_id": results[0]["exp_id"] if results else None,
        "leaderboard": leaderboard,
        "tasks": task_best,
    }

    if evo_state:
        snapshot["plateau"] = {
            "active": evo_state.plateau_mode,
            "severity": round(evo_state.get_plateau_severity(), 2),
            "best_ever": round(evo_state.best_ever, 4),
            "stuck_gens": evo_state.generation - evo_state.best_ever_gen,
        }
        snapshot["lineage"] = dict(evo_state.lineage)

    # Also push task accuracy timeseries
    _put(f"mamba3/task_history/{generation}", {
        k: round(v["acc"], 3) for k, v in task_best.items()
    })

    # Push best_fresh timeseries
    _put(f"mamba3/fresh_history/{generation}", {
        "best": round(snapshot["best_fresh"], 4),
        "t": time.time(),
    })

    return _put("mamba3/snapshot", snapshot)


# ── GPU timeseries ──────────────────────────────────────────────────

def push_gpu_tick(gpu_pct, mem_pct, n_workers, generation):
    return _put(f"mamba3/gpu_history/{generation}", {
        "gpu": round(gpu_pct, 1),
        "mem": round(mem_pct, 1),
        "workers": n_workers,
        "t": time.time(),
    })


# ── Stream events (all types) ──────────────────────────────────────

def _event(data):
    data["timestamp"] = time.time()
    return _post("mamba3/events", data)


def evt_mastery(exp_id, task, steps, examples=0, difficulty=0):
    return _event({
        "type": "mastery", "exp_id": exp_id,
        "details": f"{task} mastered in {steps:,} steps ({examples:,} examples)",
        "task": task, "steps": steps, "examples": examples, "difficulty": difficulty,
    })


def evt_unlock(exp_id, task):
    return _event({
        "type": "unlock", "exp_id": exp_id,
        "details": f"{task} unlocked", "task": task,
    })


def evt_evolve(child_id, parent_id, replaced_id, selection_reason, parent_fresh, child_config):
    return _event({
        "type": "evolve", "exp_id": child_id,
        "details": f"child of {parent_id}, replaced {replaced_id} [{selection_reason}]",
        "parent_id": parent_id, "replaced_id": replaced_id,
        "selection_reason": selection_reason, "parent_fresh": round(parent_fresh, 4),
        "child_config": child_config,
    })


def evt_pause(exp_id, fresh, cycles):
    return _event({
        "type": "pause", "exp_id": exp_id,
        "details": f"paused (fresh={fresh:.1%}, {cycles} cycles)",
        "fresh_at_pause": round(fresh, 4), "cycles_completed": cycles,
    })


def evt_spawn(exp_id, config, parent_id=None, inherited_weights=False):
    return _event({
        "type": "spawn", "exp_id": exp_id,
        "details": f"d={config.get('d_model')} L={config.get('n_kernel_layers')} "
                   f"{config.get('optimizer','adamw')} {config.get('loss_fn','stable_ce')}",
        "config": config, "parent_id": parent_id, "inherited_weights": inherited_weights,
    })


def evt_plateau_start(best_ever, stuck_gens, severity):
    return _event({
        "type": "plateau_start",
        "details": f"no improvement for {stuck_gens} gens at {best_ever:.1%}",
        "best_ever": round(best_ever, 4), "stuck_gens": stuck_gens,
        "severity": round(severity, 2),
    })


def evt_plateau_end(old_best, new_best, stuck_duration, breakthrough_exp):
    return _event({
        "type": "plateau_end",
        "details": f"breakthrough! {old_best:.1%} → {new_best:.1%}",
        "old_best": round(old_best, 4), "new_best": round(new_best, 4),
        "stuck_duration_gens": stuck_duration, "breakthrough_exp": breakthrough_exp,
    })


def evt_new_best(exp_id, fresh, previous_best, config):
    return _event({
        "type": "new_best", "exp_id": exp_id,
        "details": f"new best: {fresh:.1%} (was {previous_best:.1%})",
        "fresh": round(fresh, 4), "previous_best": round(previous_best, 4),
        "config": config,
    })


def evt_regression(exp_id, task, from_acc, to_acc):
    return _event({
        "type": "regression", "exp_id": exp_id,
        "details": f"{task} regressed {from_acc:.0%} → {to_acc:.0%}",
        "task": task, "from_acc": round(from_acc, 3), "to_acc": round(to_acc, 3),
    })


def evt_lineage_dropout(dominant_ancestor, dominance_pct, selected_from):
    return _event({
        "type": "lineage_dropout",
        "details": f"{dominance_pct:.0%} share ancestor {dominant_ancestor} — forced diversity",
        "dominant_ancestor": dominant_ancestor, "dominance_pct": round(dominance_pct, 2),
        "selected_from": selected_from,
    })


def evt_specialist_breed(child_id, parent_id, task, task_acc):
    return _event({
        "type": "specialist_breed", "exp_id": child_id,
        "details": f"bred from {parent_id} (specialist: {task} {task_acc:.0%})",
        "specialist_task": task, "specialist_acc": round(task_acc, 3), "parent_id": parent_id,
    })


def evt_radical_mutation(exp_id, severity, config):
    return _event({
        "type": "radical_mutation", "exp_id": exp_id,
        "details": f"severity {severity:.1f} — random config",
        "severity": round(severity, 2), "config": config,
    })


def evt_error(details, exp_id=None):
    return _event({
        "type": "error", "exp_id": exp_id, "details": details,
    })


def evt_vram_warning(mem_pct, n_workers):
    return _event({
        "type": "vram_warning",
        "details": f"VRAM at {mem_pct:.0f}% — pausing bottom experiments",
        "mem_pct": round(mem_pct, 1), "n_workers": n_workers,
    })


def evt_supervisor_restart(service, new_pid):
    return _event({
        "type": "supervisor_restart",
        "details": f"{service} died — restarted (PID {new_pid})",
        "service": service, "new_pid": new_pid,
    })


# ── Per-experiment data ─────────────────────────────────────────────

def push_experiment(exp_id, config, status, cycle, best_fresh, parent_id=None, n_params=0):
    return _put(f"mamba3/experiments/{exp_id}", {
        "config": config, "status": status, "cycle": cycle,
        "best_fresh": round(best_fresh, 4), "parent_id": parent_id,
        "n_params": n_params, "updated_at": time.time(),
    })


def push_experiment_cycle(exp_id, cycle, fresh, loss, type_accs):
    return _put(f"mamba3/experiments/{exp_id}/cycles/{cycle}", {
        "fresh": round(fresh, 4), "loss": round(loss, 4),
        "tasks": {k: round(v, 3) for k, v in type_accs.items() if v > 0},
        "t": time.time(),
    })


def push_experiment_teacher(exp_id, teacher):
    return _put(f"mamba3/experiments/{exp_id}/teacher", {
        "unlocked": list(teacher.unlocked_tasks),
        "difficulties": {t: round(c.difficulty, 3) for t, c in teacher.task_configs.items()},
        "mastery_log": teacher.mastery_log,
        "updated_at": time.time(),
    })


# ── Test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Testing Firebase push...")
    ok = _put("mamba3/test", {"hello": "world", "time": time.time()})
    print(f"PUT: {'OK' if ok else 'FAILED'}")

    ok = evt_mastery("exp_test", "parity", 5600, 140000, 0.3)
    print(f"Event: {'OK' if ok else 'FAILED'}")

    ok = evt_new_best("exp_test", 0.35, 0.315, {"d_model": 64})
    print(f"New best: {'OK' if ok else 'FAILED'}")

    # Cleanup
    _put("mamba3/test", None)
    print("Done.")
