---
title: Firebase Event Catalog
chapter: Infrastructure
status: reference
open_sections: 0
summary: "Reference catalog for the live Firebase event surface. The book currently renders {{firebase.mamba3Roots}} /mamba3 roots and surfaces {{firebase.signals}} signal namespace(s)."
---

# Firebase Event Catalog — Every Event We Can Send

Every piece of data that moves through the system, captured and pushed to Firebase.

## Snapshot Events (every generation, ~60s)

### `/mamba3/snapshot`
Full state of the world. Pushed every generation.

```json
{
  "timestamp": 1776813000,
  "generation": 42,
  "gpu_pct": 68,
  "mem_pct": 59,
  "n_running": 50,
  "n_paused": 30,
  "n_total": 80,
  "best_fresh": 0.315,
  "best_exp_id": "exp_045",
  "leaderboard": [...],  // top 30 with full config
  "tasks": {...},         // best per task
  "plateau": {...},       // severity, stuck_gens, best_ever
  "lineage": {...},       // full family tree
  "teacher": {...}        // curriculum state
}
```

### `/mamba3/gpu_history/{generation}`
GPU timeseries point.
```json
{ "gpu": 68, "mem": 59, "workers": 50, "t": 1776813000 }
```

## Stream Events (real-time, as they happen)

### `/mamba3/events` (append-only)

#### `mastery` — a task was mastered
```json
{
  "type": "mastery",
  "exp_id": "exp_045",
  "details": "parity mastered in 5600 steps (140000 examples)",
  "task": "parity",
  "steps": 5600,
  "examples": 140000,
  "difficulty": 0.3,
  "timestamp": 1776813000
}
```

#### `unlock` — a new task was unlocked in the curriculum
```json
{
  "type": "unlock",
  "exp_id": "exp_045",
  "details": "binary_pattern_next unlocked",
  "task": "binary_pattern_next",
  "timestamp": 1776813000
}
```

#### `evolve` — genetic evolution: child spawned from parent
```json
{
  "type": "evolve",
  "exp_id": "exp_100",
  "details": "child of exp_045, replaced exp_020",
  "parent_id": "exp_045",
  "replaced_id": "exp_020",
  "selection_reason": "specialist:logic_chain",
  "parent_fresh": 0.315,
  "child_config": {
    "d_model": 64, "n_kernel_layers": 3,
    "optimizer": "adamw", "loss_fn": "focal",
    "weight_decay": 0.1, "warm_restarts": true
  },
  "timestamp": 1776813000
}
```

#### `pause` — experiment paused (resources needed elsewhere)
```json
{
  "type": "pause",
  "exp_id": "exp_020",
  "details": "paused for evolution (fresh=10.1%)",
  "fresh_at_pause": 0.101,
  "cycles_completed": 400,
  "timestamp": 1776813000
}
```

#### `spawn` — new experiment started
```json
{
  "type": "spawn",
  "exp_id": "exp_100",
  "details": "d=64 L=3 PerpGrad lion focal warm_restart",
  "config": {...},
  "parent_id": "exp_045",
  "inherited_weights": true,
  "timestamp": 1776813000
}
```

#### `plateau_start` — population entered plateau mode
```json
{
  "type": "plateau_start",
  "details": "no improvement for 10 generations at 31.5%",
  "best_ever": 0.315,
  "stuck_gens": 10,
  "severity": 1.0,
  "timestamp": 1776813000
}
```

#### `plateau_end` — population broke through plateau
```json
{
  "type": "plateau_end",
  "details": "breakthrough! 31.5% → 35.2%",
  "old_best": 0.315,
  "new_best": 0.352,
  "stuck_duration_gens": 25,
  "breakthrough_exp": "exp_120",
  "timestamp": 1776813000
}
```

#### `new_best` — new population record
```json
{
  "type": "new_best",
  "exp_id": "exp_120",
  "details": "new best fresh: 35.2% (was 31.5%)",
  "fresh": 0.352,
  "previous_best": 0.315,
  "config": {...},
  "timestamp": 1776813000
}
```

#### `regression` — a mastered task dropped below mastery
```json
{
  "type": "regression",
  "exp_id": "exp_045",
  "details": "parity regressed from 92% to 68% — heavy replay activated",
  "task": "parity",
  "from_acc": 0.92,
  "to_acc": 0.68,
  "timestamp": 1776813000
}
```

#### `lineage_dropout` — forced breed from different lineage
```json
{
  "type": "lineage_dropout",
  "details": "60% of population shares ancestor exp_031 — forced diversity",
  "dominant_ancestor": "exp_031",
  "dominance_pct": 0.6,
  "selected_from": "exp_008",
  "timestamp": 1776813000
}
```

#### `specialist_breed` — bred from task specialist instead of overall leader
```json
{
  "type": "specialist_breed",
  "exp_id": "exp_110",
  "details": "bred from exp_049 (specialist: modus_ponens 75%)",
  "specialist_task": "modus_ponens",
  "specialist_acc": 0.75,
  "parent_id": "exp_049",
  "timestamp": 1776813000
}
```

#### `radical_mutation` — plateau triggered completely random config
```json
{
  "type": "radical_mutation",
  "exp_id": "exp_115",
  "details": "severity 3.0 — random config: d=128 L=6 lion focal noise",
  "severity": 3.0,
  "config": {...},
  "timestamp": 1776813000
}
```

#### `error` — something went wrong
```json
{
  "type": "error",
  "exp_id": null,
  "details": "Gen 42: division by zero in score_experiments",
  "traceback": "...",
  "timestamp": 1776813000
}
```

#### `worker_crash` — a worker process died
```json
{
  "type": "worker_crash",
  "exp_id": "exp_080",
  "details": "worker died after 200 cycles — OOM?",
  "last_cycle": 200,
  "last_fresh": 0.08,
  "timestamp": 1776813000
}
```

#### `vram_warning` — VRAM approaching limit
```json
{
  "type": "vram_warning",
  "details": "VRAM at 85% (69GB/81GB) — pausing bottom experiments",
  "mem_pct": 85,
  "n_workers": 50,
  "timestamp": 1776813000
}
```

#### `supervisor_restart` — supervisor restarted a service
```json
{
  "type": "supervisor_restart",
  "details": "renderer died — restarted (PID 12345)",
  "service": "renderer",
  "new_pid": 12345,
  "timestamp": 1776813000
}
```

## Per-Experiment Data

### `/mamba3/experiments/{exp_id}`
Full experiment state, updated every cycle.
```json
{
  "config": {...},
  "status": "running",
  "cycle": 400,
  "best_fresh": 0.315,
  "parent_id": "exp_031",
  "created_at": 1776810000,
  "n_params": 116067
}
```

### `/mamba3/experiments/{exp_id}/cycles/{cycle}`
Per-cycle metrics for timeseries charts.
```json
{
  "fresh": 0.315,
  "loss": 0.234,
  "tasks": { "parity": 0.82, "same_different": 0.91, ... },
  "t": 1776813000
}
```

### `/mamba3/experiments/{exp_id}/teacher`
Teacher state snapshot (periodically).
```json
{
  "unlocked_tasks": ["parity", "binary_pattern_next", "same_different"],
  "difficulties": { "parity": 0.74, "binary_pattern_next": 1.0, "same_different": 0.0 },
  "mastery_log": [...]
}
```
