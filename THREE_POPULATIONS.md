# Three Populations — The Path to a Generalist Puzzle Solver

> "Give me the brain of a fly. And I will give you tools to make the brain of a human."

---

## The Vision

Build a model that doesn't just solve puzzles — it *understands* them. A model
that, when it sees a novel pattern it's never encountered, recognizes echoes of
things it has — "this feels like counting plus mirroring" — and composes. That's
intuition. Not a toolbox. Not a router. A mind.

## What We Learned (the hard way)

1. **Multi-task training oscillates.** When one model trains on all tasks,
   gradients conflict. Parity goes up, same_different goes down. The model
   negotiates between conflicting directions and never converges. We watched
   this for hours.

2. **The genetic algorithm is essential.** It discovered d=64, L=3, PerpGrad,
   single-byte outputs — none of which we would have guessed. Evolution is
   the explorer. It finds the blueprint.

3. **Specialists are better teachers than the grader.** A Python eval says
   "right or wrong." A specialist that's 95% on parity says "85% chance it's
   S, 15% chance it's D" — and that distribution encodes the *structure* of
   parity in a way the grader never could. This is "dark knowledge."

4. **Distillation works.** Our first student hit 20.5% fresh in 4 cycles and
   SURPASSED a specialist on logic_gate (64% vs 58%) by finding shared
   internal primitives that serve multiple tasks.

5. **exp_1836 proved it's possible.** One model, evolved by the GA, achieved
   100% on parity, 100% on binary_pattern_next, 100% on same_different —
   plus non-zero on 6 more tasks. A natural generalist exists in the
   loss landscape. We just need a better way to find it.

---

## The Architecture: Three Populations

### Population 1: Workers (the explorers)

```
Purpose:     Find the best way to solve each individual task
Scope:       ONE task per worker, no multi-task interference
Method:      Genetic algorithm evolves architecture + hyperparameters
Graduation:  When accuracy reaches 100% → become a Teacher
```

Each task gets its own tournament. The GA explores:
- Architecture: d_model, d_state, n_layers, headdim
- Training: learning rate, weight decay, optimizer (AdamW/Lion)
- Loss function: CE, focal, label smoothing
- Strategies: PerpGrad, warm restarts, noise injection

**Why per-task?** No gradient conflict. A parity worker never sees
logic_gate examples. It converges fast because there's only one
objective. The GA discovers what works for each task independently.

Maybe parity likes PerpGrad but logic_gate likes grokking. The GA
finds that per-task instead of forcing one config for everything.

### Population 2: Teachers (the masters)

```
Purpose:     Teach students via distillation
Scope:       ONE task, mastered at 100%, frozen
Lifecycle:   Worker graduates at 100% → precomputes output distributions → waits
```

A teacher is a frozen specialist. It never trains again. Its value is
the full softmax distribution on thousands of examples — the "dark
knowledge" that encodes not just the right answer but the shape of
understanding:

- Which wrong answers are "close" (4 and 6 are neighbors of 5)
- How confident the model is (uncertainty = don't overfit this example)
- The internal structure of the task (comparison vs counting vs logic)

Teachers precompute 10K examples of output distributions and cache
them to disk. This means distillation runs without loading the teacher
model — just reading cached distributions. Fast.

### Population 3: Students (the generalists)

```
Purpose:     Absorb ALL teachers' knowledge into one model
Scope:       All tasks, all teachers
Method:      Distillation from cached teacher distributions + PCGrad
Evolution:   Population of students, also mutate architecture/hyperparams
```

The student starts as soon as the FIRST teacher graduates. It doesn't
wait for all 15 tasks to have teachers. When a second teacher arrives,
the student gets a new subject to learn. When a third arrives, another.
The student grows in knowledge as teachers accumulate.

**PCGrad** prevents the oscillation we saw in multi-task training.
Per-task gradients are computed separately, conflicting components
are projected out, and the student gets a clean combined signal.

**Students also evolve.** The first student might not be the best.
A population of students explores different architectures — maybe
the optimal generalist is d=96 L=4, even though the optimal parity
specialist was d=64 L=3. Evolution finds the best student shape.

---

## Resource Management

The H100 has finite VRAM. The three populations share it intelligently:

1. **Workers start.** 15 workers (one per task), each ~2GB VRAM.
2. **Worker graduates → free its slot.** When parity_worker_007 masters
   parity, ALL parity workers stop. Their VRAM is freed.
3. **Stopped workers become students.** They already have partial
   knowledge (70% on parity = a head start for distillation). The
   worker's weights become the student's initialization.
4. **Hardware stays constant.** Worker slots → student slots. The total
   compute is always 100% utilized.

As more workers graduate, more slots convert to students. Eventually:
- 0 workers (all tasks mastered)
- 15 teachers (one per task, frozen)
- N students (competing to be the best generalist)

---

## The Three Leaderboards

### Workers Leaderboard
```
| Task                  | Workers | Best Acc | Status    |
|-----------------------|---------|----------|-----------|
| parity                | 3       | 87%      | training  |
| same_different        | 2       | 92%      | training  |
| logic_gate            | 4       | 100%     | GRADUATED |
| mirror_detection      | 1       | 12%      | training  |
| ...                   |         |          |           |
```

### Teachers Leaderboard
```
| Task                  | Teacher ID     | Accuracy | Graduated At |
|-----------------------|----------------|----------|-------------|
| logic_gate            | w_logic_012    | 100%     | 14:32:01    |
| parity                | w_parity_007   | 100%     | 14:45:23    |
| same_different        | w_samediff_003 | 100%     | 15:01:44    |
| ...                   |                |          |             |
```

### Students Leaderboard
```
| Student ID | Fresh | Parity | Logic | Same/Diff | ... | Arch       |
|------------|-------|--------|-------|-----------|-----|------------|
| s_003      | 35.2% | 91%    | 78%   | 88%       | ... | d=96 L=4   |
| s_001      | 28.7% | 82%    | 65%   | 74%       | ... | d=64 L=3   |
| s_005      | 22.1% | 70%    | 55%   | 60%       | ... | d=64 L=3   |
```

---

## Implementation Plan

### Phase 1: Fix specialist_trainer.py
- Fix the NaN issue (use CE + weight_decay=0.1, not stable_ce + PerpGrad)
- Ensure precomputed teacher outputs are cached correctly
- Test locally on one task before deploying

### Phase 2: Run workers (15 parallel, one per task)
- Each worker uses the winning architecture from the GA (d=64, L=3)
- Each worker trains ONLY on its assigned task
- Monitor via Firebase (three_pop status)
- Expected: some tasks master in minutes, others take longer

### Phase 3: Teacher graduation
- When a worker hits 100% → checkpoint saved, output distributions cached
- All other workers for that task stop
- Teacher leaderboard updates in real-time

### Phase 4: Student distillation
- First student spawns when first teacher graduates
- Distills from precomputed teacher distributions (fast, no teacher inference)
- PCGrad on per-task gradients
- Monitor student vs specialists (is the whole > sum of parts?)

### Phase 5: Student evolution
- Spawn student variants with mutated configs
- GA explores student architecture (maybe d=96, L=4, L=5)
- Convert stopped workers into students (they have prior knowledge)
- Student leaderboard tracks generalist performance

### Phase 6: The test
- Once the best student plateaus, evaluate on the BOSS TASKS
  (18 unseen task types the student has never seen)
- If it generalizes → the model has learned to learn
- If it doesn't → add more task diversity and repeat

---

## Success Criteria

The three populations succeed if:

1. **Workers master all 15 tasks.** Each task has a 100% specialist.
2. **The student matches or surpasses specialists** on tasks it was
   distilled on. (Already demonstrated: logic_gate 64% > specialist 58%)
3. **The student generalizes to boss tasks** it was never trained on.
   This proves the shared internal primitives are real.
4. **The student can learn a new task from 2-3 examples** (few-shot).
   This proves it learned to learn, not just to solve.

If all four hold, we have the brain of a fly.

---

## Backup Strategy

All data is preserved:
- **SQLite metrics.db** — complete training telemetry
- **Firebase** — append-only event store (replayable)
- **Checkpoints** — all model weights on TB4 disk (298MB backup)
- **exp_1836** — the natural generalist (100%/100%/100%), backed up

The three populations system pushes to Firebase in real-time.
Three leaderboards visible at https://gauchoai.github.io/mamba3-hands-on/
