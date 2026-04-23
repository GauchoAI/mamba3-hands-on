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

---

## Config Provenance — Every Parameter Has a Source

### The rule
The diagnostic system does NOT bypass the mutation gate. It biases
the mutation, making it more likely to pick the right change. The
mutation still goes through champion-challenger. The lineage still
records everything.

But to analyze whether diagnostics actually help, we need to tag
EVERY config parameter with its provenance: who decided this value
and why.

### Data model: config_provenance

Each config in a model card carries provenance per parameter:

```python
{
    "config": {
        "d_model": 64,
        "n_kernel_layers": 3,
        "lr": 0.002,
        "noise_scale": 0.005,
        "teacher_model": "mathstral-7b",
        "weight_decay": 0.0,
    },
    "provenance": {
        "d_model": {"source": "inherited", "from": "exp_r5", "round": 5},
        "n_kernel_layers": {"source": "inherited", "from": "exp_r5", "round": 5},
        "lr": {"source": "ga_mutation", "severity": 2.0, "round": 7},
        "noise_scale": {"source": "diagnostic", "signal": "dead_grad", "round": 8},
        "teacher_model": {"source": "ga_mutation", "severity": 1.5, "round": 9},
        "weight_decay": {"source": "diagnostic", "signal": "overconfidence", "round": 6},
    }
}
```

### Source types
- **`seed`** — from BASE_CONFIG at round 0
- **`inherited`** — unchanged from parent (no mutation touched it)
- **`ga_mutation`** — the GA's random exploration changed this param.
  Records: severity, round.
- **`diagnostic`** — the diagnostic system biased the mutation toward
  this value. Records: signal name, round, prescription.
- **`teacher_inherited`** — came with a teacher through breeding

### How it flows through generations

When a child is created:
1. Start with parent's config + provenance (all params marked "inherited")
2. For each param the GA mutation changes:
   - If the change was biased by a diagnostic → source="diagnostic"
   - If the change was random GA → source="ga_mutation"
3. Unchanged params keep their parent's provenance
4. The child's provenance accumulates history from all ancestors

### Storage

Add `provenance` column to lineage table:
```sql
ALTER TABLE lineage ADD COLUMN provenance TEXT DEFAULT '{}';
```

The `provenance` field is a JSON dict mapping param names to source info.

### Implementation in mutate_config

```python
def mutate_config(parent_config, ..., diagnostic_bias=None):
    child = parent_config.copy()
    provenance = parent_provenance.copy()  # inherit parent's provenance

    # Mark all params as inherited initially
    for k in provenance:
        if provenance[k]["source"] != "seed":
            provenance[k] = {"source": "inherited",
                             "from": parent_exp_id,
                             "round": current_round}

    # Apply GA mutations (random)
    if random.random() < 0.5 * amp:
        child["lr"] = new_lr
        provenance["lr"] = {"source": "ga_mutation",
                            "severity": severity,
                            "round": current_round}

    # Apply diagnostic bias (targeted)
    if diagnostic_bias:
        signal, prescription = diagnostic_bias
        for param, value in prescription.items():
            child[param] = value
            provenance[param] = {"source": "diagnostic",
                                 "signal": signal,
                                 "round": current_round,
                                 "prescription": str(prescription)}

    return child, provenance
```

### Analysis: did diagnostics help?

Query the DB:
```sql
-- How often did diagnostic-sourced params lead to improvements?
SELECT
    p.signal,
    COUNT(*) as times_tried,
    SUM(CASE WHEN l.accuracy > l.best_accuracy THEN 1 ELSE 0 END) as wins
FROM lineage l
JOIN json_each(l.provenance) p
WHERE p.value LIKE '%diagnostic%'
GROUP BY p.signal
```

This answers: "dead_grad diagnostic was prescribed 5 times,
led to improvement 2 times (40% win rate)."

Compare to GA mutations:
```sql
-- How often did random GA mutations lead to improvements?
SELECT
    COUNT(*) as times_tried,
    SUM(CASE WHEN accuracy > best_accuracy THEN 1 ELSE 0 END) as wins
FROM lineage
WHERE mutation LIKE '%ga_mutation%'
```

If diagnostics have a higher win rate than random GA, the system
is working. If not, the diagnostics need recalibration.

### Dashboard implication

The mutation timeline can color-code entries by source:
- Gray: inherited (no change)
- Orange: GA mutation (random exploration)
- Blue: diagnostic prescription (targeted intervention)
- Green: teacher-inherited (from breeding)

A model card on the UI would show:
```
arithmetic_next — round 12
  d_model: 64        ← inherited from seed
  layers: 3          ← inherited from seed
  lr: 0.002          ← GA mutation (round 7, severity 2.0)
  noise: 0.005       ← diagnostic: dead_grad (round 8)
  teacher: mathstral  ← GA mutation (round 9, severity 1.5)
  wd: 0.0            ← diagnostic: overconfidence (round 6)
```

This makes the lineage READABLE. Not just "what config" but
"why this config" — every decision traceable to its origin.

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

---

## Strict Data Schemas — Everything Typed, Parseable, Explorable

All data must be structured, not strings. Every field has a type,
every JSON blob has a schema. No free-form text in data fields.

### Schema: Provenance Entry
```json
{
    "param": "lr",
    "value": 0.002,
    "source": "ga_mutation",          // enum: seed|inherited|ga_mutation|diagnostic|teacher_inherited
    "round": 7,
    "from_exp": "parity_r5",          // nullable: parent experiment
    "diagnostic_signal": null,        // nullable: which signal triggered this
    "diagnostic_prescription": null,  // nullable: what was prescribed
    "severity": 2.0,                  // nullable: GA severity when mutated
    "timestamp": 1776900000.0
}
```

### Schema: Model Card
```json
{
    "task": "arithmetic_next",
    "exp_id": "arithmetic_next_r12",
    "parent_id": "arithmetic_next_r11",
    "round": 12,
    "config": {
        "d_model": 64,
        "d_state": 16,
        "headdim": 16,
        "n_kernel_layers": 3,
        "lr": 0.002,
        "weight_decay": 0.0,
        "optimizer": "adamw",
        "loss_fn": "ce",
        "noise_scale": 0.005,
        "use_perp": false,
        "warm_restarts": false,
        "batch_size": 256,
        "steps_per_cycle": 200
    },
    "provenance": [
        {"param": "lr", "value": 0.002, "source": "ga_mutation", "round": 7, "severity": 2.0},
        {"param": "noise_scale", "value": 0.005, "source": "diagnostic", "round": 8,
         "diagnostic_signal": "dead_grad", "diagnostic_prescription": "noise_injection"},
        {"param": "weight_decay", "value": 0.0, "source": "diagnostic", "round": 6,
         "diagnostic_signal": "overconfidence", "diagnostic_prescription": "wd_reduction"}
    ],
    "teachers": [
        {"model": "mathstral-7b", "weight": 0.8, "from_round": 9, "source": "ga_mutation"},
        {"model": "specialist:logic_gate", "weight": 0.5, "from_round": 5, "source": "ga_mutation"}
    ],
    "diagnostics": {
        "current_signals": ["dead_grad"],
        "history": [
            {"signal": "dead_grad", "round": 8, "prescription": "noise_injection", "won": true},
            {"signal": "overconfidence", "round": 6, "prescription": "wd_reduction", "won": false}
        ]
    },
    "metrics": {
        "accuracy": 0.45,
        "best_accuracy": 0.30,
        "loss": 0.847,
        "grad_norm": 2.1,
        "param_norm": 125.3,
        "cycles_total": 120
    }
}
```

### Schema: Diagnostic Event
```json
{
    "task": "parity",
    "round": 8,
    "signal": "dead_grad",            // enum: dead_grad|oscillating_loss|accuracy_oscillation|
                                      //       loss_acc_divergence|param_explosion|grad_spike|
                                      //       stale_teacher|cycle_time_regression|mode_collapse|
                                      //       systemic_failure|length_failure|overconfidence|
                                      //       difficulty_cliff|early_plateau|never_learned|accelerating
    "evidence": {
        "avg_grad_norm": 0.08,
        "n_cycles_observed": 10,
        "threshold": 0.1
    },
    "prescription": {
        "type": "noise_injection",    // enum: noise_injection|warm_restart|teacher_distill|
                                      //       lr_spike|lr_reduce|batch_increase|wd_increase|
                                      //       focal_loss|label_smooth|perpgrad|reinit_head|
                                      //       protect_from_mutation|deprioritize|alert
        "params": {"noise_scale": 0.005},
        "priority": 1,               // 1=highest, lower tried first
        "previously_tried": 0,
        "previously_won": 0
    },
    "outcome": {                      // filled after challenger comparison
        "challenger_acc": 0.42,
        "champion_acc": 0.30,
        "won": true
    },
    "timestamp": 1776900000.0
}
```

### Schema: Error Analysis
```json
{
    "task": "parity",
    "cycle": 217,
    "n_correct": 48,
    "n_total": 100,
    "accuracy": 0.48,
    "errors": {
        "by_length": {"3": 0.95, "4": 0.80, "5": 0.60, "6": 0.30, "7": 0.10, "8": 0.00},
        "by_output": {"E": 0.70, "O": 0.26},
        "by_difficulty": {"easy": 0.85, "medium": 0.45, "hard": 0.10}
    },
    "confidence": {
        "correct_mean": 0.88,
        "correct_std": 0.12,
        "wrong_mean": 0.65,
        "wrong_std": 0.20
    },
    "derived": {
        "length_correlation": -0.85,
        "output_bias": 0.73,
        "overconfidence_score": 0.65,
        "mode_collapse_score": 0.15
    },
    "timestamp": 1776900000.0
}
```

### Schema: Cycle Telemetry (existing, for reference)
```json
{
    "task": "parity",
    "cycle": 217,
    "accuracy": 0.48,
    "loss": 0.347,
    "distill_loss": null,
    "grad_norm": 0.1,
    "lr": 0.001,
    "forward_ms": 8.5,
    "backward_ms": 12.3,
    "eval_ms": 45.0,
    "gpu_mem_mb": 70,
    "param_norm": 117.0,
    "timestamp": 1776900000.0
}
```

All enums, all typed, all parseable. No free-form strings in data
fields. The `mutation` field in lineage will transition from the
current string format to structured provenance entries.

---

## Implementation Plan — Sequenced Commits

### Commit 1: Schema migration
**Files:** `state_db.py`
- Add `provenance` column to lineage table (TEXT, JSON, default '{}')
- Add `diagnostic_history` table
- Add `error_analysis` table
- Migration auto-runs on existing DBs
- All backward compatible — existing code works without provenance

### Commit 2: Diagnostician core
**Files:** `diagnostician.py` (new)
- `class Diagnostician` with methods per signal
- `diagnose(db, task) -> list[DiagnosticEvent]`
- `prescribe(signal, task, config, db) -> (config, provenance_entries)`
- Uses `cycle_history` and `error_analysis` tables
- Returns structured `DiagnosticEvent` dicts (not strings)
- `should_prescribe()` checks diagnostic_history for past failures

### Commit 3: Error analysis in specialist_trainer
**Files:** `specialist_trainer.py`
- During eval loop: collect per-example results with properties
- Compute: errors_by_length, errors_by_output, confidence stats
- Write to `error_analysis` table via StateDB
- No change to training logic — pure observation

### Commit 4: Provenance tracking in mutation
**Files:** `coordinator.py`, `three_populations.py`
- `mutate_config()` returns `(config, provenance)` tuple
- Each mutated param tagged with source + metadata
- Inherited params tagged with parent reference
- Diagnostic bias passed as `diagnostic_bias` parameter
- Provenance stored in lineage table

### Commit 5: Wire diagnostician into orchestrator
**Files:** `three_populations.py`
- Before creating challenger: run `diagnostician.diagnose(db, task)`
- If diagnostic found: bias the mutation with `diagnostic_bias`
- After challenger comparison: log to `diagnostic_history`
- Provenance recorded in lineage for every entry

### Commit 6: Model card builder uses provenance
**Files:** `state_db.py`
- `build_model_card()` includes provenance from lineage
- Walks ancestor chain, collects provenance per param
- Returns structured model card matching the schema above

### Commit 7: Firebase sync + UI
**Files:** `state_db.py`, `docs/index.html`
- Push model cards + diagnostic events to Firebase
- UI renders provenance color-coded (gray/orange/blue/green)
- UI shows diagnostic signals when active

### Each commit:
- Write code locally
- `git add && git commit && git push`
- `ssh H100 git pull` — next worker batch picks up changes
- No restart needed for specialist_trainer changes
- Restart (via PID lock) needed for three_populations changes
- DB state always preserved

## Verification

1. Check `cycle_history` for tasks with grad_norm < 0.1
2. Verify diagnostician prescribes noise/warm_restart for those
3. Check lineage provenance field shows structured source data
4. Check error_analysis table populates with per-example breakdown
5. Check diagnostic_history tracks which prescriptions won/lost
6. Compare: diagnostic mutation win rate vs random GA win rate
7. Verify model card has full provenance chain from seed to current
