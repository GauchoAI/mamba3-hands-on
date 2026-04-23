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
