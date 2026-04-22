# UI Refactor Plan — One Shot

## Problem

The `index.html` has all UI update logic inside `db.ref('mamba3/snapshot').on('value', ...)`.
When replay mode disconnects this listener, nothing updates except a few header fields.
Replay needs to call the SAME update logic with historical data.

## Current State

- `index.html` is ~600 lines
- The snapshot listener (line ~240) does EVERYTHING:
  - Header stats (gpu, vram, workers, best_fresh)
  - Three populations banner (tp-workers, tp-teachers, tp-students, teacher badges)
  - Curriculum bar (colored task badges)
  - Curriculum detail (per-task breakdown)
  - Workers table (task, accuracy bar, best, cycle)
  - Teachers table (graduated tasks)
  - Students content
  - Task bar chart (updTask)
  - GPU/workers mini charts (updMini)
  - Plateau banner
  - Freshness indicator
- Other listeners:
  - `mamba3/events` → events stream
  - `mamba3/experiments` → per-experiment charts + per-task-by-experiment charts
  - `mamba3/task_series` → task accuracy over time chart
- Replay currently:
  - Disconnects all listeners
  - Clears UI to blank
  - Reads `/mamba3/history` + `/mamba3/events`
  - Sorts chronologically
  - Plays them back but only updates header fields + events

## The Fix (one function extraction)

### Step 1: Extract `updateUI(d)`

Take the ENTIRE body of the `db.ref('mamba3/snapshot').on('value', snap => { ... })` 
callback and move it into a standalone function:

```javascript
function updateUI(d) {
  if (!d) return;
  lastTs = d.timestamp || 0;
  // ... ALL the code that's currently inside the listener ...
  // headers, three_pop, curriculum, workers, teachers, students, 
  // task chart, gpu chart, workers chart, plateau
}
```

### Step 2: Live mode calls it

```javascript
db.ref('mamba3/snapshot').on('value', snap => updateUI(snap.val()));
```

### Step 3: Replay calls it

In `playNext()`, when `item.type === 'snapshot'`, build a compatible object from history data and call `updateUI()`:

```javascript
if (item.type === 'snapshot') {
  const h = item.data;
  // Map history format → snapshot format
  const fakeSnapshot = {
    timestamp: item.time,
    gpu_pct: h.gpu_pct || 0,
    mem_pct: h.mem_pct || 0,
    n_running: h.n_workers || 0,
    n_total: (h.n_workers||0) + (h.n_teachers||0) + (h.n_students||0),
    best_fresh: h.best_fresh || 0,
    tasks: h.tasks || {},
    // Build leaderboard from worker_best
    leaderboard: Object.entries(h.worker_best || {}).map(([task, acc]) => ({
      task, acc, best: acc, cycle: 0, status: acc >= 0.95 ? 'teacher' : 'training'
    })).concat(
      // Add teachers
      (h.teachers || []).map(t => ({ task: t, acc: 1.0, status: 'teacher' }))
    ).sort((a,b) => b.acc - a.acc),
    teacher_leaderboard: (h.teachers || []).map(t => ({ task: t, acc: 1.0, status: 'teaching' })),
    three_pop: {
      n_workers: h.n_workers || 0,
      n_teachers: h.n_teachers || 0,
      n_students: h.n_students || 0,
      teachers: Object.fromEntries((h.teachers||[]).map(t => [t, ''])),
      tasks_remaining: [],
      generation: 0,
    },
  };
  updateUI(fakeSnapshot);
}
```

### Step 4: Events replay (already working)

The events replay code already works — it prepends to the events div. No changes needed.

### Step 5: Task series replay

The task_series listener reads from `/mamba3/task_series` which has historical data. 
In replay mode, load it once and progressively reveal data points:

```javascript
// During replay init, load task_series once
const taskSeriesSnap = await db.ref('mamba3/task_series').once('value');
const allTaskSeries = taskSeriesSnap.val() || {};

// In playNext, filter task_series to only show data up to current replay time
// (task_series keys are cycle numbers, history keys are timestamps)
```

This is optional for v1 — the bar chart from updateUI is sufficient.

## Data Format Reference

### `/mamba3/history/{timestamp}` (what replay reads)
```json
{
  "best_fresh": 0.315,
  "gpu_pct": 59.0,
  "mem_pct": 55.8,
  "n_workers": 13,
  "n_teachers": 2,
  "n_students": 0,
  "tasks": { "parity": {"acc": 0.91, "exp": "worker"}, ... },
  "top3": [{"exp_id": "...", "fresh": 0.31, ...}],
  "teachers": ["modus_ponens", "run_length_next"],
  "worker_best": {"parity": 0.62, "logic_gate": 0.45, ...}
}
```

### `/mamba3/snapshot` (what live mode reads — updateUI expects this)
```json
{
  "timestamp": 1776823000,
  "mode": "three_populations",
  "gpu_pct": 60.0,
  "mem_pct": 56.0,
  "n_running": 13,
  "n_total": 15,
  "best_fresh": 0,
  "leaderboard": [{"task": "parity", "acc": 0.62, "best": 0.62, "cycle": 10, "status": "training"}, ...],
  "teacher_leaderboard": [{"task": "geometric_next", "acc": 1.0, "exp_id": "...", "status": "teaching"}],
  "tasks": { "parity": {"acc": 0.62, "exp": "worker"}, ... },
  "three_pop": { "n_workers": 13, "n_teachers": 2, "n_students": 0, "teachers": {...}, "tasks_remaining": [...] }
}
```

## Checklist

- [ ] Extract `updateUI(d)` from the snapshot listener
- [ ] Wire live mode: `on('value', snap => updateUI(snap.val()))`
- [ ] Wire replay: build fakeSnapshot from history, call `updateUI(fakeSnapshot)`
- [ ] Test live mode still works after refactor
- [ ] Test replay shows task chart, workers, teachers, curriculum updating
- [ ] Test replay with `?replay=true&speed=10`
