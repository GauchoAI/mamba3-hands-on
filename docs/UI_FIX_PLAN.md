# UI Fix Plan — Every Broken Item, Why, and How to Fix

## Context

The training system works perfectly. Three populations architecture:
- 5/15 tasks graduated to teachers in 3 rounds
- Per-task specialists converge in minutes
- Firebase receives data via `on_cycle` callback in `three_populations.py`

The ONLY problem is the UI (`docs/index.html`). Multiple charts and
sections are empty or broken because the data paths don't match between
what the training pushes and what the UI reads.

## File under surgery: `docs/index.html` (~750 lines)

---

## Issue 1: "Fresh Accuracy Over Time" chart — EMPTY

### What the user sees
An empty chart with no lines.

### What the UI code does (line ~350)
```javascript
db.ref('mamba3/experiments').on('value', s => {
  // Reads /mamba3/experiments/*/cycles/* 
  // Builds one line per experiment
  // Expects: { cycles: { "1": {fresh: 0.5, loss: 0.3}, "2": {...} } }
});
```

### What Firebase actually has
`/mamba3/experiments` is **null** (empty). The `three_populations.py` 
`on_cycle` callback pushes to `/mamba3/snapshot` and `/mamba3/task_series`
but NOT to `/mamba3/experiments`.

### The fix
In `three_populations.py`, inside the `on_cycle` callback, add:
```python
fb._put(f"mamba3/experiments/{task_name}/cycles/{cycle}", {
    "fresh": round(acc, 4),
    "loss": round(loss, 4),
})
fb._put(f"mamba3/experiments/{task_name}/best_fresh", round(best, 4))
fb._put(f"mamba3/experiments/{task_name}/status", "training")
fb._put(f"mamba3/experiments/{task_name}/config", {"task": task_name})
```

This creates one "experiment" per task (not per worker), with cycle
timeseries. The UI will show one line per task on the fresh chart,
which is actually more useful than one line per experiment.

---

## Issue 2: "Task Accuracy Over Time" chart — EMPTY

### What the user sees
Empty chart, or briefly flashes then disappears.

### What the UI code does (line ~375)
```javascript
db.ref('mamba3/task_series').on('value', s => {
  // Reads /mamba3/task_series/{task}/{cycle}
  // Expects: { "parity": { "1": {acc: 0.5, diff: 0}, "2": {...} } }
});
```

### What Firebase actually has
The `on_cycle` callback in `three_populations.py` pushes:
```python
fb._put(f"mamba3/task_series/{task_name}/{cycle}", {
    "acc": round(acc, 3), "diff": 0,
})
```

This SHOULD work. But the data might have been cleared when we wiped
Firebase. Need to verify data exists:
```
curl https://...firebasedatabase.app/mamba3/task_series.json?shallow=true
```

If null → the on_cycle callback isn't firing or failing silently.
Check the three_populations.py log for Firebase errors.

### The fix
1. Verify data exists in Firebase
2. If not, check on_cycle is being called (add a print)
3. The UI code should work as-is once data flows

---

## Issue 3: GPU & Workers mini charts — EMPTY

### What the user sees
Two blank chart areas.

### What the UI code does (inside `updateUI`)
```javascript
gpuH.push(d.gpu_pct||0); if(gpuH.length>100) gpuH.shift();
wkH.push(d.n_running||0); if(wkH.length>100) wkH.shift();
updMini('gpuChart', gpuH, ...);
updMini('workersChart', wkH, ...);
```

These arrays accumulate one point per snapshot update. They start
empty and grow as the live listener fires.

### Why it's empty
The snapshot updates every ~10 seconds (one per training cycle). 
With only a few updates, the arrays have <5 points. The Chart.js
line chart needs multiple points to render visibly.

Also: `d.gpu_pct` might be 0 because the three_populations.py 
`on_cycle` callback pushes gpu_pct from `get_gpu_usage()` which
might fail (returns 0.0 on error).

### The fix
1. Verify gpu_pct is non-zero in the Firebase snapshot:
   `curl .../mamba3/snapshot/gpu_pct.json`
2. If zero, fix get_gpu_usage() in the on_cycle callback
3. The charts will populate over time as more snapshots arrive
4. Consider seeding the arrays with a few initial points so the
   chart renders immediately

---

## Issue 4: Family Tree — EMPTY

### What the user sees
Blank white area.

### What the UI code does (inside `updateUI`)
```javascript
if(d.lineage) drawTree(d.leaderboard||[], d.lineage);
```

### Why it's empty
`d.lineage` is **undefined** in the snapshot. The three populations
system has no lineage — each specialist trains independently on one
task. There's no parent-child breeding. The family tree concept
doesn't apply to the three populations architecture.

### The fix
**Option A:** Hide the family tree section entirely in three_pop mode.
Replace with a "Task Mastery Timeline" showing when each task graduated.

**Option B:** Show a simple visualization of tasks → teachers → student
flow instead of a family tree. Like a pipeline diagram.

**Option C:** Remove the family tree card and use the space for something
useful (e.g., a larger task accuracy chart, or the distillation progress).

Recommended: Option A — timeline of graduations.

---

## Issue 5: Replay jumps to end instantly

### What the user sees
Clicks replay, sees the loading message, then immediately the final
state with no animation.

### What the code does
```javascript
const realDelay = (timeline[idx].time - item.time) * 1000;
const replayDelay = Math.max(50, realDelay / replaySpeed);
setTimeout(playNext, Math.min(replayDelay, 2000));
```

### Why it jumps
The history snapshots were pushed in rapid succession during fast
training (~10 seconds apart real-time). At speed=10, that's 1 second
between snapshots. But many snapshots have nearly identical timestamps
(pushed in the same second during on_cycle bursts), so realDelay is
0ms → replayDelay is 50ms. 100 events × 50ms = 5 seconds total.
It LOOKS like it jumps because 5 seconds is too fast to see.

### The fix
Set a **minimum visual delay** regardless of real timestamps:
```javascript
const minVisualDelay = 300; // 300ms minimum between steps
const replayDelay = Math.max(minVisualDelay, realDelay / replaySpeed);
```

At 300ms minimum, 100 events takes 30 seconds — watchable.
At speed=2, use 500ms minimum. At speed=10, use 200ms minimum.
Scale: `minVisualDelay = Math.max(100, 500 / replaySpeed)`

---

## Issue 6: Leaderboard in replay shows wrong data

### What the user sees
Workers table is empty or shows stale data during replay.

### Why
The replay `updateUI(fakeSnapshot)` builds a leaderboard from
`h.worker_best` (dict of task→acc). This mapping was fixed in the
last refactor. But old history entries might not have `worker_best`
(they were pushed by the old firebase_sync format).

### The fix
The mapping code handles missing data gracefully (empty arrays).
New history entries (from three_populations) have `worker_best`.
The replay will work correctly for new history — old history may
show empty leaderboards, which is acceptable.

---

## Summary: what to do (in order)

1. **three_populations.py**: Add experiment cycle push to on_cycle
   callback (fixes Fresh Accuracy Over Time chart)
2. **Verify** task_series data in Firebase (fixes Task Accuracy Over Time)
3. **docs/index.html**: Set replay minimum delay to 300ms
   (fixes replay jumping)
4. **docs/index.html**: Replace family tree with task mastery timeline
   (fixes empty section)
5. **Verify** gpu_pct in snapshot is non-zero (fixes GPU chart)
6. **Test** live mode — all charts should populate
7. **Test** replay — should step through visibly with all UI updating

Estimated time: 30 minutes for a focused implementation.
