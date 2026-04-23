# Model Card Plan — Genetic Lineage + Multi-Teacher Distillation

## Problem

When a challenger wins with `teacher_model: mathstral-7b`, that knowledge
should persist AND accumulate across generations:

1. The champion should keep distilling from its inherited teachers every round
2. When bred, the child inherits ALL teachers from its lineage
3. Teachers are weighted by recency and effectiveness

Currently: teacher_model is in the config but only used during the
challenger phase. The champion ignores it.

## Design: Model Card

Each experiment gets a **model card** computed at setup from its lineage:

```python
{
    "exp_id": "arithmetic_next_r15",
    "task": "arithmetic_next",
    "parent_id": "arithmetic_next_r12",
    "config": {"d_model": 96, "n_kernel_layers": 3, "lr": 1e-3, ...},
    "teachers": [
        {"model": "mathstral-7b", "weight": 1.0, "from_round": 12},
        {"model": "specialist:logic_gate", "weight": 0.5, "from_round": 8},
    ],
}
```

### How teachers accumulate

Walk the parent chain in the lineage DB:

```
exp_r15 (mutation: lr change)
  └─ parent: exp_r12 (won with teacher=mathstral-7b)
       └─ parent: exp_r8 (won with teacher=specialist:logic_gate)
            └─ parent: exp_r1 (base config, no teacher)
```

At setup for exp_r15:
- mathstral-7b: inherited from r12 (2 generations ago) → weight = 1.0 * 0.8^1 = 0.8
- specialist:logic_gate: inherited from r8 (4 gen ago) → weight = 1.0 * 0.8^3 = 0.51

Weight decay factor: 0.8 per generation. Recent teachers matter more.

### When breeding (mutation)

New mutation adds `teacher_model: specialist:same_different`:

```python
child.teachers = parent.teachers + [
    {"model": "specialist:same_different", "weight": 1.0, "from_round": 15}
]
```

The child distills from ALL three teachers simultaneously.

### During training

Each cycle:
1. Generate task data (normal)
2. For each teacher in model_card.teachers:
   - Get teacher's output distribution for the batch
   - Compute distillation loss (KL divergence)
   - Weight by teacher.weight
3. Combine: total_loss = task_loss + sum(teacher_weight * distill_loss)
4. Backward + optimize

Teachers with low weight contribute less to the gradient.
If a teacher hurts performance, the champion-challenger will drop it.

## Data Model

### StateDB: lineage table (already exists)

Add `teachers` column (JSON list of teacher entries):

```sql
ALTER TABLE lineage ADD COLUMN teachers TEXT DEFAULT '[]';
```

### Model card builder

```python
def build_model_card(db, task, round_num):
    """Walk lineage, collect inherited teachers, compute weights."""
    lineage = db.get_lineage(task)
    
    teachers = []
    seen = set()
    decay = 0.8
    
    # Walk backwards from current round
    for i, entry in enumerate(reversed(lineage)):
        cfg = entry["config"]
        teacher = cfg.get("teacher_model")
        if teacher and teacher not in seen:
            generations_ago = i
            weight = decay ** generations_ago
            teachers.append({
                "model": teacher,
                "weight": round(weight, 3),
                "from_round": entry["round"],
            })
            seen.add(teacher)
    
    return {
        "task": task,
        "round": round_num,
        "config": db.get_best_config(task)[0] or {},
        "teachers": teachers,
    }
```

## Implementation Plan

### Step 1: Add `teachers` to lineage entries

- Update `state_db.py`: add `teachers` column to lineage table
- Update `log_lineage()` to accept and store teachers list
- Add `build_model_card()` method

### Step 2: Champion training uses model card teachers

- In `three_populations.py`, before training each task:
  - Build model card from lineage
  - If model card has teachers, load them
  - Pass teachers to training function
- In `specialist_trainer.py`, add `teachers` parameter:
  - If teachers provided, compute blended loss each step:
    `loss = task_loss + sum(w * distill_loss(student, teacher, batch))`

### Step 3: Mutations breed teachers

- In `mutate_config()`, `teacher_model` mutations ADD to existing teachers
  (don't replace)
- New challengers inherit parent's teachers + any new mutation
- Champion-challenger comparison uses the full teacher set

### Step 4: Smooth deployment

- `state_db.py`: backwards-compatible (new column has default '[]')
- `specialist_trainer.py`: `teachers=None` parameter, no change if None
- `three_populations.py`: builds model card, passes teachers
- Deploy: `git pull` on H100, PID lock handles restart
- No state loss: DB preserved, checkpoints preserved, lineage preserved

### Step 5: Verify

- Check lineage entries have teachers field
- Check training log shows "Distilling from N teachers"
- Check model card accumulates teachers across generations
- Check champion-challenger works with multi-teacher distillation

## Files to modify

| File | Change |
|------|--------|
| `state_db.py` | Add teachers column, build_model_card() |
| `specialist_trainer.py` | Add teachers param, blended distillation loss |
| `three_populations.py` | Build model card, pass teachers to training |
| `coordinator.py` | teacher_model mutations accumulate, don't replace |

## Deployment

```bash
# On H100 — single command, no state loss
ssh H100 'cd /root/mamba3-hands-on && git pull'
# PID lock in three_populations.py auto-kills old, starts new
# DB schema auto-migrates (ALTER TABLE with DEFAULT)
# Existing lineage entries get teachers='[]' (empty, backward compat)
```
