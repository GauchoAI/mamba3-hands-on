# Diagnostics Plan — Telemetry-Driven Targeted Mutations

## Context

The GA explores blindly — random mutations with random probability. When
a task has dead gradients (grad_norm=0.1), the GA might try changing
d_model when the real fix is noise injection. The diagnostic system reads
telemetry from `cycle_history` and prescribes the RIGHT mutation.

This is not a replacement for the GA. The GA handles creative exploration
(discovering that L=3 beats L=1). The diagnostic system handles medical
emergencies (dead gradients, exploding params, loss divergence).

## Data Source

All signals come from `cycle_history` table in SQLite:

```sql
SELECT task, cycle, accuracy, loss, grad_norm, param_norm, lr,
       forward_ms, backward_ms, gpu_mem_mb
FROM cycle_history
WHERE task = ? ORDER BY cycle DESC LIMIT 20
```

20 most recent cycles give us trends, means, variances.

---

## Signal 1: Dead Gradients

### Detection
```python
recent = get_cycle_history(task, last_n=10)
avg_grad = mean(r["grad_norm"] for r in recent)
if avg_grad < 0.1:
    diagnose("dead_gradients")
```

### What it means
Model is at a flat plateau in the loss landscape. No direction to improve.
More cycles with the same config will produce zero learning.

### Observed in
- same_different: grad_norm=0.0-0.1 for 400+ cycles at 87% best
- parity: grad_norm=0.1-0.2 for 200+ cycles at 63% best

### Prescribed mutations (in priority order)
1. **noise_scale=0.005** — perturb weights to escape the flat region.
   Cost: near zero. Can be combined with current config.
2. **warm_restarts=True** — periodic lr spikes. Each spike is a chance
   to escape. T_0=100 means a spike every 100 steps.
3. **teacher_model=best_available** — external gradient signal. The
   distillation loss provides gradients even when task loss is flat.
   Uses teacher_eval_cache to pick the best available.
4. **lr *= 10 for 1 cycle** — temporary spike. Risky but sometimes
   the model just needs a push. Revert lr after 1 cycle.

### Implementation
```python
def prescribe_dead_gradients(task, config, db):
    """Return a targeted challenger config for dead gradients."""
    cfg = config.copy()

    # Try noise first (cheapest)
    if cfg.get("noise_scale", 0) < 0.001:
        cfg["noise_scale"] = 0.005
        return cfg, "dead_grad: noise injection"

    # Try warm restarts
    if not cfg.get("warm_restarts"):
        cfg["warm_restarts"] = True
        return cfg, "dead_grad: warm restarts"

    # Try best available teacher
    best_teachers = db.get_best_teachers_for_task(task, min_accuracy=0.01)
    if best_teachers:
        cfg["teacher_model"] = best_teachers[0][0]
        return cfg, f"dead_grad: teacher={best_teachers[0][0]}"

    # Last resort: lr spike
    cfg["lr"] = cfg.get("lr", 1e-3) * 10
    return cfg, "dead_grad: lr spike 10x"
```

---

## Signal 2: Oscillating Loss (high variance, flat mean)

### Detection
```python
recent = get_cycle_history(task, last_n=20)
losses = [r["loss"] for r in recent]
mean_loss = mean(losses)
var_loss = variance(losses)
trend = losses[-1] - losses[0]  # is it trending down?

if var_loss > 0.01 and abs(trend) < 0.001:
    diagnose("oscillating_loss")
```

### What it means
Model is bouncing around a minimum but can't settle. The learning rate
is too high for the current landscape, or the batch is too small
(noisy gradients).

### Prescribed mutations
1. **lr *= 0.3** — reduce learning rate to allow finer convergence
2. **batch_size *= 2** — smoother gradients, less oscillation
3. **weight_decay += 0.05** — regularize toward simpler (more stable) solution

---

## Signal 3: Accuracy Oscillation (wild swings)

### Detection
```python
recent = get_cycle_history(task, last_n=20)
accs = [r["accuracy"] for r in recent]
spread = max(accs) - min(accs)
best = max(accs)

if spread > 0.3 and best > 0.7:
    diagnose("accuracy_oscillation")
```

### What it means
Model finds good solutions but can't stabilize. The representations
are fragile — small weight changes flip many predictions.

### Observed in
- binary_pattern_next: swings 55%-93% across rounds

### Prescribed mutations
1. **label_smooth** — softer targets prevent the overconfident
   all-or-nothing predictions that cause oscillation
2. **lr *= 0.5** — smaller steps, more stable convergence
3. **Increase eval count** — current 100 examples has high variance.
   Increasing to 200 gives more reliable accuracy signal.

---

## Signal 4: Loss Down, Accuracy Flat

### Detection
```python
recent = get_cycle_history(task, last_n=20)
loss_trend = recent[-1]["loss"] - recent[0]["loss"]  # negative = decreasing
acc_trend = recent[-1]["accuracy"] - recent[0]["accuracy"]

if loss_trend < -0.01 and abs(acc_trend) < 0.02:
    diagnose("loss_acc_divergence")
```

### What it means
Model is optimizing something that doesn't help exact-match accuracy.
Possible causes:
- Overfitting to easy examples while ignoring hard ones
- Learning confidence (sharper distributions) without learning correctness
- Loss function not aligned with the evaluation metric

### Prescribed mutations
1. **focal_loss** — focuses gradient on hard examples. If the model is
   acing easy ones and ignoring hard ones, focal fixes that.
2. **use_perp=True** — PerpGrad prevents "naive loss minimization" where
   the model reduces loss without learning the algorithm.

---

## Signal 5: Param Norm Growing

### Detection
```python
recent = get_cycle_history(task, last_n=20)
pnorms = [r["param_norm"] for r in recent]
growth = pnorms[-1] / max(pnorms[0], 1e-6)

if growth > 1.5:  # params grew 50%+ in 20 cycles
    diagnose("param_explosion")
```

### What it means
Weights are growing — the model is memorizing rather than generalizing.
Bigger weights = sharper decision boundaries = overfitting.

### Prescribed mutations
1. **weight_decay += 0.1** — actively shrink weights
2. **lr *= 0.5** — slower updates to prevent further growth
3. **noise_scale=0.001** — perturb to break memorization circuits

---

## Signal 6: Grad Norm Spike (sudden 10x+)

### Detection
```python
recent = get_cycle_history(task, last_n=10)
gnorms = [r["grad_norm"] for r in recent]
if len(gnorms) >= 2:
    ratio = gnorms[-1] / max(mean(gnorms[:-1]), 1e-6)
    if ratio > 10:
        diagnose("grad_spike")
```

### What it means
Two possibilities:
- **Breakthrough** (good): the model just learned something new. The
  spike is the rapid weight adjustment. binary_pattern_next showed
  this: grad_norm spiked to 17.8 when accuracy jumped 86%→90%.
- **Instability** (bad): the model is about to diverge. Loss will
  spike next.

### Prescribed action
- **Do nothing for 3 cycles** — wait and see if it's a breakthrough
- If loss spikes after: **lr *= 0.3** and **increase grad clip**
- If accuracy improves: **celebrate** — the model is grokking

---

---

## Signal 7: Stale Teacher

### Detection
```python
for teacher in model_card["teachers"]:
    teacher_score = db.get_teacher_score(teacher["model"], task)
    current_best = task_best[task]
    if teacher_score is not None and teacher_score < current_best:
        diagnose("stale_teacher", teacher=teacher["model"])
```

### What it means
The student has surpassed the teacher. The teacher's output distributions
are now WORSE than the student's own representations. Continuing to
distill from this teacher adds noise, not knowledge.

### Prescribed mutations
1. **Drop the stale teacher** — remove from config, train without it
2. **Swap to a stronger teacher** — check teacher_eval_cache for a
   teacher that scores higher than current best
3. **Promote to self-teaching** — if no external teacher is better,
   the model is the best available. It should train alone.

---

## Signal 8: Cycle Time Regression

### Detection
```python
recent = get_cycle_history(task, last_n=10)
times = [r["forward_ms"] for r in recent]
baseline = mean(times[:5])
current = mean(times[-3:])

if current > baseline * 1.8:  # 80% slower
    diagnose("cycle_time_regression")
```

### What it means
Not a training problem — an operational problem:
- VRAM pressure from another worker
- Model grew from architecture mutation (more layers = slower)
- CUDA contention from llama-server or other processes
- GPU thermal throttling

### Prescribed action
Not a training mutation — an operational alert:
1. **Log warning** — "task X cycle time 2x slower"
2. **Check pool size** — reduce max_concurrent if VRAM is tight
3. **Check for external processes** — llama-server still running?
4. **If arch mutation caused it** — note the time cost in lineage
   so future analysis can correlate "L=8 is 3x slower"

---

## Signal 9: Mode Collapse (identical accuracy)

### Detection
```python
recent = get_cycle_history(task, last_n=15)
accs = [r["accuracy"] for r in recent]
unique_accs = len(set(round(a, 2) for a in accs))

if unique_accs <= 2 and len(accs) >= 15:
    diagnose("mode_collapse")
```

### What it means
The model outputs the same prediction for every (or nearly every) input.
For parity, it might always say "E" (50% accuracy on a balanced dataset).
The model has collapsed to a trivial solution.

### Observed in
- Possible explanation for parity stuck at exactly 50% (coin flip)

### Prescribed mutations (aggressive — mode collapse is severe)
1. **noise_scale=0.01** — heavy noise to break the collapsed state
2. **lr = initial_lr * 5** — large lr to escape the trivial minimum
3. **loss_fn=focal** — focal loss penalizes the dominant prediction,
   forcing the model to differentiate
4. **Re-initialize last layer** — keep SSM weights but randomize
   the output head. The collapse is usually in the head.
5. **Teacher distillation** — external gradient signal forces diverse
   outputs (a teacher that outputs different things for different
   inputs breaks the "always predict X" pattern)

---

## Signal 10: Cross-Task Systemic Failure

### Detection
```python
dead_tasks = []
for task in active_tasks:
    recent = get_cycle_history(task, last_n=10)
    avg_grad = mean(r["grad_norm"] for r in recent)
    if avg_grad < 0.1:
        dead_tasks.append(task)

if len(dead_tasks) >= 3:
    diagnose("systemic_failure", tasks=dead_tasks)
```

### What it means
Multiple tasks dying simultaneously is NOT independent failures.
Possible systemic causes:
- GPU contention (workers fighting for compute)
- Shared optimizer bug
- Data generator producing degenerate examples
- CUDA driver issue (silent corruption)
- All tasks hit the same architectural ceiling (model too small)

### Prescribed action
1. **Alert — don't treat individually** — prescribing noise to 3
   tasks at once won't fix a systemic cause
2. **Check GPU health** — nvidia-smi, temperature, ECC errors
3. **Check pool size** — too many workers = each gets insufficient GPU
4. **Check model size** — if all tasks plateau at 60-70%, the model
   might just be too small. Prescribe a global d_model increase.
5. **Log to findings.md** — systemic failures are important events

---

---

## Signal 11: Error Analysis — WHAT is the model getting wrong?

### The gap
We compute accuracy as a single number: 63%. But we never ask which
examples fail and why. The error pattern directly maps to the fix.

### Data collection
During evaluation (every cycle), log each example's properties and result:

```python
# In specialist_trainer.py eval loop
error_log = []
for ex in eval_examples:
    tokens, sep = tok.encode_curriculum(ex)
    # ... forward pass ...
    correct = (predicted == expected)
    error_log.append({
        "correct": correct,
        "confidence": softmax_prob_of_predicted,
        "input_len": len(ex["input"].split(",")),
        "output": ex["output"],
        "predicted": predicted_char,
        "difficulty": ex.get("difficulty", 0),
    })
```

### Storage
```sql
CREATE TABLE IF NOT EXISTS error_analysis (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task TEXT NOT NULL,
    cycle INTEGER NOT NULL,
    n_correct INTEGER,
    n_total INTEGER,
    -- Error breakdown
    errors_by_length TEXT,    -- JSON: {"3": 0.95, "5": 0.80, "8": 0.20}
    errors_by_output TEXT,    -- JSON: {"E": 0.90, "O": 0.35}
    avg_confidence_correct REAL,
    avg_confidence_wrong REAL,
    -- Derived signals
    length_correlation REAL,  -- negative = fails on longer inputs
    output_bias REAL,         -- how skewed predictions are to one class
    overconfidence REAL,      -- confidence when wrong (high = bad)
    timestamp REAL NOT NULL
);
```

### Error patterns and prescriptions

**Pattern A: Length-dependent failure**
```
input_len=3: 95% correct
input_len=5: 80% correct
input_len=8: 20% correct
```
- Diagnosis: model can't track long sequences
- Prescription: increase `d_state` (larger SSM state), increase
  `n_kernel_layers` (more sequential processing), or use a teacher
  that handles long sequences well

**Pattern B: Output class bias**
```
Always predicts "E": accuracy on E-examples=100%, O-examples=0%
Overall accuracy: 50% (coin flip on balanced dataset)
```
- Diagnosis: mode collapse to majority class
- Prescription: focal loss (penalizes dominant class), label smoothing,
  aggressive noise, re-init output head

**Pattern C: Confident and wrong**
```
avg_confidence_correct: 0.92
avg_confidence_wrong: 0.85  ← almost as confident when WRONG
```
- Diagnosis: overconfidence / memorization
- Prescription: label smoothing, weight decay increase, PerpGrad
  (prevents naive loss minimization)

**Pattern D: Uncertain and wrong**
```
avg_confidence_correct: 0.88
avg_confidence_wrong: 0.52  ← genuinely uncertain
```
- Diagnosis: model is close but needs more capacity or training
- Prescription: more cycles (it's still learning), possibly more
  layers or wider model. NOT noise/lr changes — the model is
  doing the right thing, just needs more time.

**Pattern E: Random errors (no pattern)**
```
No correlation with length, output class, or difficulty.
Errors scattered uniformly across examples.
```
- Diagnosis: model has learned the algorithm but has a noisy
  implementation. Evaluation variance is high.
- Prescription: larger eval set (reduce variance), lower lr
  (stabilize weights), or just patience.

**Pattern F: Difficulty cliff**
```
difficulty < 0.3: 95% correct
difficulty 0.3-0.6: 60% correct
difficulty > 0.6: 10% correct
```
- Diagnosis: model mastered easy cases but can't generalize
  to harder ones. This is the curriculum problem.
- Prescription: the AdaptiveTeacher approach (progressive
  difficulty). Or a teacher that mastered the hard cases.

### Implementation in diagnostician
```python
def diagnose_errors(db, task):
    """Analyze error patterns and prescribe targeted fix."""
    recent = db.get_error_analysis(task, last_n=5)
    if not recent:
        return None

    latest = recent[-1]

    # Check output bias (mode collapse)
    if latest["output_bias"] > 0.8:
        return "mode_collapse", prescribe_mode_collapse(task)

    # Check length correlation
    if latest["length_correlation"] < -0.5:
        return "length_failure", prescribe_length_failure(task)

    # Check overconfidence
    if latest["overconfidence"] > 0.7:
        return "overconfidence", prescribe_overconfidence(task)

    # Check difficulty cliff
    errors_by_len = json.loads(latest["errors_by_length"])
    if errors_by_len:
        sorted_lens = sorted(errors_by_len.items(), key=lambda x: int(x[0]))
        if len(sorted_lens) >= 3:
            easy_acc = mean([v for k, v in sorted_lens[:len(sorted_lens)//3]])
            hard_acc = mean([v for k, v in sorted_lens[-len(sorted_lens)//3:]])
            if easy_acc > 0.8 and hard_acc < 0.3:
                return "difficulty_cliff", prescribe_difficulty_cliff(task)

    return None  # no clear error pattern — let GA explore
```

---

## Signal 12: Convergence Speed — Learning Curve Shape

### Detection
```python
recent = get_cycle_history(task, last_n=50)
if len(recent) < 20:
    return  # not enough data

accs = [r["accuracy"] for r in recent]

# Compute learning rate: improvement per cycle
early_rate = (accs[10] - accs[0]) / 10  # first 10 cycles
late_rate = (accs[-1] - accs[-10]) / 10  # last 10 cycles

if early_rate > 0.02 and late_rate < 0.001:
    diagnose("early_plateau")  # learned fast then stopped
elif early_rate < 0.001 and late_rate < 0.001:
    diagnose("never_learned")  # never got traction
elif late_rate > early_rate:
    diagnose("accelerating")   # grokking? getting faster
```

### What it means
- **early_plateau**: the model learned a shortcut fast (memorization?)
  and now can't improve. Needs: weight decay to erode shortcuts,
  or harder training data.
- **never_learned**: the config is fundamentally wrong for this task.
  Needs: radical mutation (different architecture entirely).
- **accelerating**: rare and exciting — the model is grokking. 
  DON'T TOUCH IT. No mutations, no interventions. Let it run.

### Prescribed action for "accelerating"
```python
if signal == "accelerating":
    # Protect this task — no mutations until acceleration stops
    db.set_config(f"protect_{task}", True)
    log("GROKKING DETECTED — protecting {task} from mutation")
```

---

## Signal 13: Resource Efficiency — Cost per Improvement

### Detection
```python
lineage = db.get_lineage(task)
improvements = [(e["round"], e["accuracy"]) for e in lineage
                if e["accuracy"] > (lineage[lineage.index(e)-1]["accuracy"]
                if lineage.index(e) > 0 else 0)]

total_cycles = sum(r["cycle"] for r in get_cycle_history(task))
total_improvements = len(improvements)
cycles_per_improvement = total_cycles / max(total_improvements, 1)
```

### What it means
If parity has spent 500 cycles across 15 rounds with 0 improvements,
its cost-per-improvement is infinite. If binary_pattern_next improved
3 times in 30 cycles, it's 10 cycles per improvement.

### Prescribed action
- **High cost (>100 cycles/improvement)**: deprioritize this task.
  Move it to a lower frequency (train every 3rd round instead of
  every round). Spend GPU time on tasks that are actually learning.
- **Low cost (<20 cycles/improvement)**: prioritize this task.
  Give it more cycles per round. It's the best return on GPU time.

This connects to pool sizing: instead of equal time per task,
allocate GPU time proportional to expected improvement.

---

## Diagnostic History — Don't Repeat Failed Treatments

### Storage
```sql
CREATE TABLE IF NOT EXISTS diagnostic_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task TEXT NOT NULL,
    signal TEXT NOT NULL,
    prescription TEXT NOT NULL,
    challenger_acc REAL,
    champion_acc REAL,
    won INTEGER NOT NULL,  -- 1=challenger won, 0=champion held
    timestamp REAL NOT NULL
);
```

### Logic
```python
def should_prescribe(db, task, signal, prescription):
    """Check if this prescription has been tried before and failed."""
    cur = db.conn.execute(
        "SELECT COUNT(*) as tries, SUM(won) as wins "
        "FROM diagnostic_history "
        "WHERE task=? AND signal=? AND prescription=?",
        (task, signal, prescription)
    )
    row = cur.fetchone()
    tries, wins = row[0], row[1] or 0

    if tries >= 3 and wins == 0:
        return False  # failed 3 times, stop prescribing

    return True
```

### After challenger comparison
```python
db.log_diagnostic(
    task=task, signal="dead_gradients",
    prescription="noise_scale=0.005",
    challenger_acc=challenger_acc,
    champion_acc=best,
    won=1 if challenger_acc > best else 0,
)
```

### Escalation
When a prescription fails 3 times:
1. Move to the next prescription in priority order
2. If ALL prescriptions for a signal have failed 3 times:
   - Escalate to radical GA mutation (severity=3.0)
   - Log: "all diagnostic treatments failed for {task}/{signal}"
   - This is valuable information — the task may need something
     fundamentally different (more layers, different architecture)

---

## Architecture

```
cycle_history table (populated by subprocess workers)
        ↓
    Diagnostician (runs after each batch in orchestrator)
        ↓
    For each active task:
        ├── Read last 20 cycles
        ├── Check all 6 signals
        ├── If diagnosed:
        │     ├── Prescribe targeted mutation
        │     ├── Create challenger with prescribed config
        │     ├── Log diagnosis in lineage: "dead_grad: noise injection"
        │     └── Champion-challenger comparison as usual
        └── If healthy:
              └── Normal GA mutation (random exploration)
```

### Priority
Diagnostic mutations take priority over random GA mutations.
If a task has dead gradients, don't waste a challenger on a random
d_model change — apply the noise injection first.

But: if the diagnostic mutation was already tried and the champion
held, fall back to random GA. Don't keep prescribing the same fix.

### Tracking
The `mutation` field in lineage records the diagnosis:
```
"dead_grad: noise injection"
"oscillating_loss: lr reduction"
"accuracy_oscillation: label smoothing"
```

This allows post-hoc analysis: "which diagnoses led to improvements?"

---

## Files to modify

| File | Change |
|------|--------|
| `state_db.py` | Add `get_diagnostics(task)` method |
| `three_populations.py` | Call diagnostician before GA mutation |
| `coordinator.py` | No change — GA stays as fallback |

## Deployment

No state loss. New code runs on next restart.
Diagnostic mutations go through champion-challenger — no risk.
cycle_history already populated (210+ entries and growing).

## Verification

1. Check `cycle_history` for tasks with grad_norm < 0.1
2. Verify diagnostician prescribes noise/warm_restart for those
3. Check lineage for "dead_grad:" mutation entries
4. Compare: did diagnostic mutations beat random GA mutations?
