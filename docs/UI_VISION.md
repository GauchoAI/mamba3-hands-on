---
title: UI Vision
chapter: Infrastructure
status: planning
open_sections: 0
summary: "Design roadmap for turning Firebase into a richer reader experience; current book surface already sees {{lab.tasksTracked}} tasks, {{lab.models}} models, and {{lab.streams}} streams."
---

# UI Vision — Exploiting the Full Data Stack

## Data Architecture

```
SQLite (source of truth)         Firebase (real-time UI feed)
┌────────────────────┐           ┌──────────────────────┐
│ teachers           │──sync──→  │ /state/teachers      │
│ lineage            │──sync──→  │ /state/lineage       │
│ experiments        │           │ /state/model_cards   │
│ task_status        │──sync──→  │ /state/task_status   │
│ active_runs        │──push──→  │ /realtime/active     │
│ cycle_history      │──push──→  │ /realtime/cycles     │
│ error_analysis     │──sync──→  │ /state/errors        │
│ diagnostic_history │──sync──→  │ /state/diagnostics   │
│ teacher_eval_cache │──sync──→  │ /state/teacher_matrix│
│ runtime_config     │           │ /state/config        │
└────────────────────┘           └──────────────────────┘
         ↑                                ↑
    Workers write                   UI subscribes
    every cycle                     (event-based)
```

### Push Strategy — Event-Based, Not Polling

**Real-time (pushed by workers every cycle, ~2s latency):**
- `/realtime/active/{task}` — current cycle, accuracy, loss, grad_norm
- `/realtime/gpu` — GPU%, VRAM%, timestamp
- `/snapshot/timestamp` — freshness indicator

**Snapshot (pushed by orchestrator end of each round, ~30-60s):**
- `/snapshot/leaderboard` — all 15 tasks ranked
- `/snapshot/three_pop` — teachers/workers/students counts
- `/snapshot/tasks` — per-task accuracy for bar chart

**Sync (pushed by sync_to_firebase() end of each round):**
- `/state/teachers` — graduated tasks (append-only)
- `/state/lineage` — per-task summary + model cards
- `/state/task_status` — full status per task
- `/state/diagnostics` — diagnostic history + win rates
- `/state/teacher_matrix` — teacher × task heatmap data
- `/state/errors` — latest error analysis per task
- `/state/knowledge_flow` — teacher→task edges

**The UI subscribes to Firebase paths using `db.ref(path).on('value')`
— event-based, no polling. When data changes, the callback fires.**

---

## UI Features — From Simple to Ambitious

### Tier 1: Already Working (improve quality)

#### 1.1 Leaderboard
**Data:** `/snapshot/leaderboard`
**Current:** Shows task name, accuracy, status. NaN issues fixed.
**Improve:**
- Add sparkline per task (last 10 cycle accuracies from `/realtime/cycles`)
- Color-code by trend: green=improving, yellow=plateau, red=regressing
- Show cycle count and time training
- Click to expand model card

#### 1.2 Task Accuracy Bar Chart
**Data:** `/snapshot/tasks`
**Current:** Horizontal bars per task.
**Improve:**
- Gradient fill: green for mastered, blue for training, gray for waiting
- Show historical best as a faded bar behind current
- Animate transitions when accuracy changes

#### 1.3 GPU/VRAM Stats
**Data:** `/realtime/gpu` (pushed by workers)
**Current:** Header numbers, mini sparklines.
**Improve:**
- Show per-worker GPU usage (which task is using how much)
- Color: green=healthy, yellow=high, red=saturated

#### 1.4 Mutation Timeline
**Data:** `/snapshot/lineage`
**Current:** File-tree with progress bars per round.
**Improve:**
- Show provenance colors: gray=inherited, orange=GA, blue=diagnostic, green=teacher
- Expandable: click to see full config diff from parent
- Filter: "show only diagnostic mutations" or "show only winners"

---

### Tier 2: New Features (use existing Firebase data)

#### 2.1 Live Training Monitor
**Data:** `/realtime/active/{task}`
**What:** Real-time dashboard showing what's training RIGHT NOW.
```
┌─────────────────────────────────────────┐
│ LIVE TRAINING                           │
│                                         │
│ 🔄 parity        cycle 237  acc 51%     │
│    loss ████████░░ 0.347                │
│    grad ░░░░░░░░░░ 0.1 ← DEAD          │
│                                         │
│ 🔄 same_different cycle 415  acc 57%    │
│    loss ████████░░ 0.346                │
│    grad ░░░░░░░░░░ 0.0 ← DEAD          │
│                                         │
│ 🔥 binary_p_next  cycle 24   acc 90%    │
│    loss ██░░░░░░░░ 0.151                │
│    grad █████████░ 17.8 ← BREAKTHROUGH  │
└─────────────────────────────────────────┘
```
- Updates every 2 seconds (worker push)
- Grad norm color: green=healthy, yellow=low, red=dead, purple=spike
- Loss sparkline: last 20 cycles inline
- Click to see full cycle history chart

#### 2.2 Diagnostic Dashboard
**Data:** `/state/diagnostics`
**What:** Shows active signals and prescription effectiveness.
```
┌─────────────────────────────────────────┐
│ DIAGNOSTICS                             │
│                                         │
│ Active Signals:                         │
│   ⚠ parity: dead_grad (10 cycles)      │
│   ⚠ same_different: dead_grad (8 cyc)  │
│   🎯 binary_p_next: accelerating!      │
│                                         │
│ Prescription Win Rates:                 │
│   noise_injection: 0/4 (0%)             │
│   warm_restart: not tried yet           │
│   teacher_distill: not tried yet        │
│   focal_loss: 0/1 (0%)                  │
│                                         │
│ Next Rx: warm_restart for parity        │
└─────────────────────────────────────────┘
```
- Real-time signal detection visualization
- Win rate bars per prescription type
- "What's next" prediction

#### 2.3 Error Analysis Heatmap
**Data:** `/state/errors`
**What:** Per-task error pattern visualization.
```
┌─────────────────────────────────────────┐
│ ERROR PATTERNS                          │
│                                         │
│ parity:                                 │
│   By length: ███ 3  ██░ 5  ░░░ 8       │
│   By output: E=70%  O=26%  ← BIASED    │
│   Confidence: correct=0.88 wrong=0.65   │
│                                         │
│ arithmetic_next:                        │
│   By length: ██░ 3  █░░ 5  ░░░ 8       │
│   By output: balanced                   │
│   Confidence: correct=0.72 wrong=0.51   │
└─────────────────────────────────────────┘
```
- Mini heatmaps showing accuracy by input length
- Output bias indicator (mode collapse detection)
- Confidence scatter: correct vs wrong

#### 2.4 Teacher Effectiveness Matrix
**Data:** `/state/teacher_matrix`
**What:** Heatmap showing which teachers help which tasks.
```
┌──────────────────────────────────────────────────┐
│ TEACHER × TASK MATRIX                            │
│                                                  │
│              parity  arith  logic  same   binary │
│ mathstral     0%     87%    70%    50%    33%    │
│ qwen-math     0%      0%    43%    47%    47%    │
│ spec:logic    0%      —      —      0%     —     │
│ spec:same     —       —      —      —      47%   │
│                                                  │
│ Green = teacher better than our specialist       │
│ Red = teacher worse                              │
│ Gray = not evaluated                             │
└──────────────────────────────────────────────────┘
```
- Color intensity = accuracy
- Click cell to see if this teacher was ever used in lineage
- Shows which teachers are worth trying for stuck tasks

---

### Tier 3: Ambitious (new Firebase data needed)

#### 3.1 Live Genetic Tree

**Data:** `/snapshot/lineage` (already has parent pointers, config, won/lost, provenance)
**What:** Interactive tree per task showing the full evolution of configs.

```
parity — Genetic Tree
                                    
  [seed]──────────────────────────────────────────
  d64 L3 adamw CE wd=0.1                         
  round 0, 60%                                    
     │                                            
     ├──[champion r1] 62% ▲                       
     │     │                                      
     │     ├──[challenger r4] Lion+label_smooth    
     │     │   62% ✗ (champion held)              
     │     │                                      
     │     ├──[challenger r5] lr=9e-4             
     │     │   62% ✗                              
     │     │                                      
     │     └──[challenger r6] L=2, lr=9e-4        
     │         62% ✗ (arch change, fresh start)   
     │                                            
     └──[diagnostic r8] noise_scale=0.005 🔬       
         55% ✗ (dead_grad prescription failed)    
```

- **D3.js tree layout** — horizontal, scrollable
- **Node = one experiment** (champion or challenger)
- **Node color:**
  - Green = improved (won)
  - Red = lost (champion held)
  - Blue = diagnostic prescription
  - Gold = mastered (graduated)
  - Gray = inherited (no change)
- **Node size** = accuracy (bigger = better)
- **Edge label** = what changed (mutation diff)
- **Click node** = expand model card with full provenance
- **Live updates** — new nodes appear as challengers are tried
- **Filter:** show only winners, only diagnostics, only teacher mutations
- **Per-task tabs** — switch between tasks to see their individual trees
- **Cross-task view** — when `specialist:same_different` teaches `arithmetic_next`,
  draw a dashed edge between the two task trees

**Firebase data needed (already available):**
```javascript
db.ref('mamba3/snapshot/lineage').on('value', s => {
  // Each node has: parent, task, round, acc, best, role,
  // won, config, teachers, mutation, provenance
  drawGeneticTree(s.val());
});
```

The lineage data already has everything: parent pointers for the tree
structure, config for the node labels, won/lost for the colors,
provenance for the source tags, teachers for the cross-task edges.

---

#### 3.2 Knowledge Flow Graph (Cross-Task)
**Data:** `/state/knowledge_flow`
**What:** Interactive graph showing how knowledge flows between tasks.
```
    [mathstral-7b]
         │ 87%
         ▼
    [arithmetic_next] ──── 30% best
         
    [specialist:same_diff] ──→ [binary_pattern_next]
         87%                        94%
         
    [specialist:logic_gate] ──→ [logic_chain]
         100%                       98%
```
- D3.js force-directed graph
- Edge thickness = teacher weight
- Node size = task accuracy
- Animated: edges pulse when distillation is active
- Shows cross-task knowledge transfer in real time

#### 3.2 Model Card Viewer
**Data:** `/state/model_cards/{task}`
**What:** Click any task to see its full genealogy.
```
┌─────────────────────────────────────────┐
│ MODEL CARD: arithmetic_next             │
│                                         │
│ Config:                                 │
│   d_model: 64     ← seed (round 0)     │
│   layers: 3       ← seed (round 0)     │
│   lr: 0.002       ← GA (round 7, s=2)  │
│   noise: 0.005    ← diagnostic: dead_   │
│                      grad (round 8)     │
│   teacher: mathstral ← GA (round 9)    │
│   wd: 0.0         ← diagnostic: over_  │
│                      confidence (r6)    │
│                                         │
│ Teachers (inherited):                   │
│   mathstral-7b     weight=0.80  r9      │
│   spec:logic_gate  weight=0.51  r5      │
│                                         │
│ Diagnostics:                            │
│   dead_grad: noise tried 3x, won 0     │
│   next: warm_restart                    │
│                                         │
│ Performance: 30% best, 326 cycles       │
│ Lineage: 14 rounds, 8 challengers      │
└─────────────────────────────────────────┘
```
- Every config param color-coded by source
- Teacher inheritance tree
- Diagnostic history summary
- Full lineage timeline inline

#### 3.3 Convergence Race Chart
**Data:** `/realtime/cycles` (per-cycle accuracy)
**What:** Animated bar chart race showing all tasks competing over time.
- Y-axis: tasks, sorted by current accuracy
- X-axis: accuracy 0-100%
- Animated: bars grow as training progresses
- Tasks swap positions as accuracy changes
- Teachers flash gold when they graduate
- Play/pause/speed controls

#### 3.4 Training Cost Dashboard
**Data:** Computed from `cycle_history`
**What:** How much GPU time has each task consumed vs its progress.
```
┌─────────────────────────────────────────┐
│ COST EFFICIENCY                         │
│                                         │
│ Task              Cycles  Time   ROI    │
│ modus_ponens        5    8s    ████████ │
│ run_length_next     7    12s   ████████ │
│ logic_gate         13    22s   ██████   │
│ binary_p_next      24    40s   █████    │
│ same_different    415    11m   ██       │
│ parity            326    9m    ░        │
│ arithmetic_next   326    9m    ░        │
│                                         │
│ Total GPU time: 45 minutes              │
│ Efficiency: 6 teachers / 45 min = 0.13  │
└─────────────────────────────────────────┘
```
- Sort by ROI (improvement per cycle)
- Highlight tasks that should be deprioritized (high cost, low return)

#### 3.5 Replay Mode (Enhanced)
**Data:** `/history` + `/state/lineage`
**What:** Replay the entire training history as a movie.
- Timeline scrubber at bottom
- All charts animate as history replays
- Mutation events highlighted with popups
- Graduation celebrations (confetti?)
- Speed controls: 1x, 5x, 10x, 50x
- "Jump to interesting moment" — auto-detects breakthroughs

---

## Firebase Push Implementation

### Workers push real-time data (every cycle):
```python
# In specialist_trainer.py — already pushing some, add:
fb._put(f"mamba3/realtime/active/{task}", {
    "cycle": cycle, "accuracy": acc, "loss": loss,
    "grad_norm": grad_norm, "param_norm": param_norm,
    "lr": current_lr, "gpu_mem_mb": gpu_mem,
    "timestamp": time.time(),
})
```

### Orchestrator pushes snapshots (every round):
```python
# Already doing this. Add task_status + diagnostics:
fb._put("mamba3/state/task_status", {
    task: {"status": s["status"], "best": s["best_accuracy"],
           "cycles": s["total_cycles"], "config": s["current_config"]}
    for task, s in all_status.items()
})
```

### sync_to_firebase() pushes full state (every round):
```python
# Already pushes teachers + lineage. Add:
fb._put("mamba3/state/diagnostics", diagnostic_stats)
fb._put("mamba3/state/errors", latest_error_analysis)
fb._put("mamba3/state/teacher_matrix", teacher_eval_matrix)
fb._put("mamba3/state/model_cards", {
    task: db.build_model_card(task) for task in ALL_TASKS
})
```

### Event-based UI subscription:
```javascript
// UI subscribes — no polling
db.ref('mamba3/realtime/active').on('value', s => updateLiveMonitor(s.val()));
db.ref('mamba3/state/diagnostics').on('value', s => updateDiagDashboard(s.val()));
db.ref('mamba3/state/errors').on('value', s => updateErrorHeatmap(s.val()));
db.ref('mamba3/state/teacher_matrix').on('value', s => updateTeacherMatrix(s.val()));
db.ref('mamba3/snapshot').on('value', s => updateMainDashboard(s.val()));
```

Each subscription fires ONLY when the data changes — pure event-driven.
No polling, no wasted bandwidth, no stale data.

---

## Implementation Priority

1. **Push active_runs to Firebase** — enables Live Training Monitor
2. **Push task_status to Firebase** — enables richer leaderboard
3. **Push diagnostic_history** — enables Diagnostic Dashboard
4. **Push error_analysis** — enables Error Heatmap
5. **Push teacher_eval_cache** — enables Teacher Matrix
6. **Push model cards** — enables Model Card Viewer
7. **Build Knowledge Flow graph** — D3.js force-directed
8. **Build Convergence Race** — animated bar chart
9. **Enhanced Replay** — timeline with events

Each step is a separate commit. Each adds one Firebase path + one UI component.
All event-based. All lightweight.
