# GA tournament / mass-evolution / multi-task arc — findings journal

The middle period of the project: from "we can solve parity" to "we
have a 50-experiment population evolving across 15 tasks on an H100,
with teacher mutation, GPU saturation, kernel bugs, and the eventual
14/15-tasks-mastered milestone."

Entries 14–17 and 19–24 of the original findings.md (entry 18 was
skipped in the original numbering). The PTX engine arc that followed
(entries 27–54) lives at `engine/ptx/findings.md`.

Originally inline in the root `findings.md`; moved here as part of the
structuring pass (2026-04-30).

---

## Entry 14 — The roadmap: from puzzle solver to formal reasoning

### Where we are

The curriculum trains the model on 15 specific task types, one at a time.
Parity is mastered. Binary patterns mastered in fewer examples (transfer!).
The model is climbing the difficulty ladder.

But 15 task types is still a fixed set. The real test is what comes next.

### The three leaps

**Leap 1: Generalized puzzle solver (meta-learning)**

The model should solve tasks it has *never been trained on*, given only
a few examples. This is in-context learning / few-shot reasoning:

```
Example: 3 5 → 8, 2 7 → 9, 4 1 → ?
Answer: 5  (the model inferred "addition" from two examples)
```

How to get there:
- The curriculum builds atomic skills (compare, count, detect patterns)
- The boss tasks test whether these skills compose into new abilities
- True few-shot eval: present 2-3 examples of a novel task type,
  ask the model to infer the rule and apply it
- If it can do this, it has learned *how to learn patterns*, not just
  specific patterns

The key metric: **zero-shot accuracy on novel task types.** After
training on 15 types + 18 boss types, generate a *34th* type the model
has never seen. Can it figure it out?

**Leap 2: Language as interface**

The model goes from *doing* to *explaining*. Four levels:

```
Level A: Solve silently (current — outputs just the answer)
Level B: Solve + state the rule ("the pattern repeats every 3")
Level C: Express rules formally ("∀n: a(n+3) = a(n)")
Level D: Chain rules into proofs ("periodic ∧ arithmetic → closed form")
```

This requires switching to the bilingual byte-level LM (`mamba3_lm.py`)
and training on reasoning traces where the model shows its work.

The seeds we built with Cerebras (thinking + answer + python) were
exactly this format. The curriculum provides the *grounding* — when
the model writes "I'll compare adjacent pairs," it has actually learned
what comparison means from Level 0-1 training.

**Leap 3: Formal mathematics**

Once the model can express reasoning in language, the next step is
formal notation:

```
Natural:  "the sequence increases by 3 each time"
Formal:   a(n) = a(0) + 3n
Proof:    ∀n ∈ ℕ: a(n+1) - a(n) = 3  (by induction on training examples)
```

The path: logic gates → propositional logic (modus ponens, already in
curriculum) → predicate logic → simple proofs → induction → algebraic
manipulation (SymPy traces from BOOTSTRAP.md Level 5).

### Checkpoint strategy

The curriculum checkpoint is small (~2MB for 405K params). Can be
committed directly to git. To continue training:

```python
cfg = Mamba3Config(d_model=128, d_state=16, expand=2, headdim=16)
model = PlainModel(cfg).to(device)
ckpt = torch.load("curriculum_best.pt", map_location=device)
model.load_state_dict(ckpt["model"])
# Continue training on new data (language, formal math)
```

The SSM state representations carry forward. Skills learned in Level 0
(parity, patterns) become the substrate for Level 1+ reasoning.

### The ultimate validation

Give the model a problem it has *never* seen, in *natural language*:

```
User: "Is 371 an Armstrong number?"

Model thinking:
  An Armstrong number has digits whose cubes sum to itself.
  371 → digits: 3, 7, 1
  3³ = 27, 7³ = 343, 1³ = 1
  27 + 343 + 1 = 371 = original
  → YES

Model output: Yes, 371 is an Armstrong number.
  3³ + 7³ + 1³ = 27 + 343 + 1 = 371 ✓
```

The thinking block uses real operations the model learned:
decomposition (Level 0), exponentiation (Level 4), summation (Level 2),
comparison (Level 1). Language (Level 6) is the interface.

If this works, we have a system that reasons — not because it memorized
Armstrong numbers, but because it bootstrapped from parity to formal
mathematics through six levels of increasingly powerful computation.

The brain of a fly, becoming the brain of a human.

---

## Entry 15 — First 100K run complete, overnight launched

### First run results (100K steps, plain vs augmented)

**Plain model timeline:**
```
Step 2K:    parity MASTERED (100% fresh, 30K examples)
Step 14K:   binary_pattern_next MASTERED (100%, 13K examples — 2x faster!)
Step 24K:   same_different UNLOCKED (was 49% before any training — transfer!)
Step 48K:   same_different peaked at 75%
Step 62K:   parity hit difficulty 1.0 (length 12, still 100%)
Step 100K:  parity 100%, binary_pattern 100%, same_different ~57-70%
```

same_different never hit 90% mastery in 100K steps. The model learned
to compare numbers 0-3 at ~70% accuracy but couldn't break through.
The transition from binary (0/1) to numeric reasoning is the current
bottleneck.

**Augmented model: failed**
```
Step 100K:  train=70%, fresh=3.8%, parity only 38%
            Spikes: reg=0.74, mem=0.23
            Wildly unstable — loss swinging between 0.3 and 1.8
```

The augmented model never stabilized. Registers and memory added
complexity without providing useful structure for these tasks. The
plain model dramatically outperformed it. The augmented architecture
needs fundamental rethinking — possibly interleaving registers with
the scan rather than post-processing, or only activating registers
at Stage 3+ when sequence memory is actually needed.

**Key observations from the 100K run:**

1. **Transfer is real.** same_different was at 49% before any training,
   purely from parity/binary transfer. sequence_completion and
   alternating_next also showed small transfer (6%).

2. **Catastrophic forgetting is temporary.** Parity dropped to 43%
   around step 38K while learning same_different, but recovered to
   100% by step 62K. The teacher's retreat mechanism works, just slowly.

3. **Binary → numeric is the hardest transition.** The model spent 24K
   steps in a pure binary world. Moving to numbers 0-3 required
   learning entirely new token representations.

4. **Learning to learn (partial).** Task 1: 30K examples. Task 2: 13K
   examples. Half the examples needed. But the step count was the same
   (2000 steps) because task 2 shared the batch with task 1.

### Overnight run launched

Resumed from the 100K checkpoint at step 100K. 600K total steps (~10
hours). The teacher restarted fresh (old code didn't save teacher
state) but the model weights carry all learned representations. Parity
should re-master instantly.

Added checkpoint resume support: model weights + optimizer state +
teacher state + grokfast EMA all saved and restored. Future runs will
have seamless continuation.

---

## Entry 16 — The evolutionary tournament

### From one experiment to a civilization

What started as a single model training on parity became something
bigger. By the end of this session, the H100 is running an
**evolutionary tournament**: 76+ experiments competing, breeding,
and dying on the same GPU simultaneously.

### The infrastructure

```
Workers (35 running)     → SQLite metrics.db ← Coordinator
    ↓ train independently        ↓ evolves population
    ↓ write per-cycle metrics    ↓ pauses losers
    ↓ write checkpoints          ↓ spawns mutant children
    ↓                            ↓ inherits parent weights
    ↓                            ↓ manages 50GB disk budget
    └──────────────── GPU 100% ──┘
                                 ↓
                            Renderer → index.html + dashboard.md
                                 ↓
                         http://localhost:9090 (SSH tunnel)
```

**Map-Reduce architecture:** Workers are the map (train independently),
coordinator is the reduce (read metrics, evolve, prune). Filesystem
is the message bus. Workers survive coordinator restarts. Watchdog
auto-restarts the coordinator on crash.

**Genetic evolution:** Winners reproduce — their config gets mutated
(tweak lr, batch_size, d_model, weight_decay, loss_fn, optimizer,
backend). Children inherit parent weights when architecturally
compatible (same d_model, d_state, headdim, n_layers). Losers get
paused (not killed) — they can be resumed if resources free up.
200-cycle grace period protects new experiments from premature death.

**Auto-scaling:** The coordinator monitors GPU utilization and spawns
more workers until the GPU hits 90%. Started at 6 workers, scaled
to 35. From 11% GPU to 100%.

### The DAME bug

The model learned parity at 89% per-byte accuracy but only 57%
exact match. Every "SAME" prediction came out as "DAME" — the model
always output 'D' as the first byte because DIFF was easier to
learn and the first byte got stuck in a local minimum.

**Root cause:** Multi-byte outputs ("SAME" = 4 bytes) where one byte
carries all the information and the other 3 are deterministic. The
gradient signal to flip from D→S was too weak relative to the signal
for the remaining bytes.

**Fix:** Single-byte outputs — "S" instead of "SAME", "D" instead of
"DIFF". The kernel doesn't need to spell out words. One byte = one
decision. The cortex will later learn to translate S→"SAME"→"igual".

### Five new training strategies

The tournament started with two strategies: grokking (weight decay)
and PerpGrad. We added five more, all competing in the same
evolutionary pool:

1. **SAM** (Sharpness-Aware Minimization) — seeks wide flat valleys
   in the loss landscape. Wide = robust = generalizes.

2. **Label smoothing** — trains on "90% S, 10% everything else"
   instead of "100% S". Prevents overconfidence. Directly targets
   the DAME bug pattern.

3. **Lion optimizer** — Google's optimizer using sign of momentum.
   50% less memory than Adam, often faster on small models.

4. **Warm restarts** — periodically reset learning rate to high.
   Lets the model escape wherever it's stuck.

5. **Noise injection** — add random perturbation to weights every
   cycle. Forces robust solutions, escapes local minima.

Each strategy can combine with any other via mutation. Evolution
can discover that Lion + label smoothing + warm restarts beats
AdamW + grokking. Or it won't. The data decides, not us.

### Current state of the tournament

```
76+ experiments, 35 running, 100% GPU
Best fresh: 11.6% (exp_059, d=96, wd=0.1)
Parity best: 82% (exp_065)
Same_different: 82% (transfer, never trained directly)
Arithmetic_next: 4% (transfer to Stage 4!)
Fastest parity mastery: 5,600 steps (exp_108)

Grokking dominates top 5 (weight decay = 0.05-0.1)
PerpGrad competitive at #7 (exp_095, d=48, 29K params)
Evolution discovered d=96 beats d=64 (exp_059)
Evolution trying 3-layer models (exp_076, d=96, 214K params)
```

They're competing against 70+ existing grokking experiments right
now. Evolution will cross-breed the winners. The genetic tournament
just got a lot more interesting.

### The progressive model

The architecture grew too:

- **Byte-level tokenizer** (260 vocab) — universal, no vocabulary wall
- **Progressive growing** — start with 1 layer (45K params), add more
  as needed. The model earns its complexity.
- **Kernel/cortex split** — kernel layers for reasoning, cortex layers
  for language. Shared embedding bridges both.
- **Near-identity layer init** — new layers start invisible (scale=0.01)
  and gradually learn to contribute.

### What we learned

1. **The training signal matters more than model size.** 405K params
   on an H100 memorized but didn't generalize. 45K params with the
   right curriculum + grokking + byte-level tokenization does better.

2. **Multi-byte outputs are a trap.** The DAME bug wasted hours of
   compute. Single-byte outputs eliminate an entire class of local
   minima.

3. **Evolution works.** Let 76 experiments compete instead of hand-
   tuning one. The genetic algorithm discovered d=96 > d=64, found
   that wd=0.05 is competitive with wd=0.1, and is now exploring
   Lion, focal loss, and warm restarts without human intervention.

4. **Transfer is real.** same_different at 82% without any direct
   training. The model learned comparison from parity training alone.

5. **GPU utilization requires multi-process parallelism.** A tiny
   model on a huge GPU can't saturate it. Multiple independent
   processes on the same GPU solve this (11% → 100%).

6. **The cortex and kernel should evolve separately.** Two different
   specializations, shared embedding. Like left brain, right brain.

---

## Entry 17 — The breakthrough: 31.5% fresh, 13/15 tasks active

### What happened

After restarting all workers with single-byte outputs, the ceiling
shattered. In under 2 hours:

```
Before restart (old code):     After restart (new code):
  Fresh: 12.3% ceiling           Fresh: 31.5% and climbing
  Tasks active: 2/15             Tasks active: 13/15
  Parity: 82% (stuck)            Parity: 75% (still learning)
  same_different: 82% transfer   same_different: 92% MASTERED ✅
  Grokking dominated             PerpGrad dominates
  Architecture: d=64, L=1        Architecture: d=64, L=3
```

### The three keys

1. **Single-byte outputs** — "S"/"D" instead of "SAME"/"DIFF". Eliminated
   the DAME bug entirely. One byte = one decision = no local minimum from
   correlated multi-byte outputs.

2. **3-layer architecture** — evolution discovered d=64, L=3 (116K params)
   massively outperforms L=1 (45K params). Three layers can represent
   multi-step reasoning. One layer could only do one transformation.

3. **PerpGrad** — with single-byte outputs, PerpGrad's faster convergence
   wins over grokking. PerpGrad dominates all top 5. Weight decay was
   compensating for the multi-byte output problem — once that was fixed,
   the simpler optimizer (PerpGrad) was better.

### Curriculum explosion

13 of 15 tasks showing non-zero accuracy, most via pure transfer:

```
✅ same_different:      92%  — MASTERED (Stage 1)
🔄 binary_pattern_next: 88%  — nearly mastered (Stage 0)
🔄 parity:              75%  — solid (Stage 0)
🔄 modus_ponens:        71%  — propositional logic! (Stage 5)
🔄 logic_chain:         56%  — chained gates (Stage 5)
🔄 repeat_count:        44%  — counting (Stage 3)
🔄 logic_gate:          36%  — AND/OR/XOR (Stage 5)
🔄 odd_one_out:         33%  — outlier detection (Stage 1)
🔄 geometric_next:      31%  — ratio detection (Stage 4)
🔄 sequence_completion: 18%  — pattern prediction (Stage 2)
🔄 alternating_next:    17%  — interleaved sequences (Stage 4)
🔄 arithmetic_next:     12%  — step detection (Stage 4)
🔒 pattern_period:       0%  — still locked
🔒 run_length_next:      0%  — still locked
```

The model learned Stage 5 logic (modus ponens at 71%) without EVER
being trained on it. The curriculum only trains parity (Stage 0).
Everything else is transfer.

### Meta-evolution deployed

We added four strategies to the evolution itself, inspired by
training optimizer concepts:

**Deployed and running:**

1. **Momentum tracking** — experiments scored by trajectory, not just
   snapshot. `effective_score = best_fresh + 0.5 * momentum`. A fast
   climber ranks higher than a stagnant leader.

2. **Focal task attention** — 30% chance to breed from a task
   specialist instead of the overall winner. If exp_049 is best at
   modus_ponens, it might get to reproduce even if its overall fresh
   is lower. Preserves genetic diversity for different capabilities.

3. **Temperature annealing** — early generations: breed randomly
   (explore). Late generations: breed from the top (exploit).
   `temperature = 5 * exp(-generation / 50)`. The population starts
   creative and gradually becomes selective.

4. **Lineage dropout** — if >50% of running experiments share the
   same great-grandparent, force breed from a different lineage.
   Prevents monoculture. Currently all top 10 are d=64 L=3 PerpGrad
   clones — lineage dropout will push diversity.

### Training strategies: what's deployed vs what's planned

**Deployed and competing:**
- ✅ Grokking (weight decay 0.05-0.2) — was dominant, now #2
- ✅ PerpGrad (orthogonal gradient) — current champion
- ✅ StableMax (numerically stable softmax) — used by all
- ✅ Focal loss — available, some experiments using it
- ✅ Label smoothing — available via mutation
- ✅ Lion optimizer — available via mutation
- ✅ Warm restarts — available via mutation
- ✅ Noise injection — available via mutation
- ✅ Regular cross-entropy — available as alternative to StableMax

**Not yet implemented (future work):**
- ❌ SAM (Sharpness-Aware Minimization) — code exists in strategies.py
  but not wired into the worker. Needs 2-forward-pass training loop.
- ❌ Ensemble predictions — combine multiple experiments' outputs
- ❌ Knowledge distillation — best model teaches others
- ❌ Tinygrad backend — was crashing (numpy fixed), needs more testing
- ❌ Cortex training — language LM side not started
- ❌ Few-shot evaluation — test on truly novel tasks

### Infrastructure delivered

```
Component                     Status
────────────────────────────  ──────
Workers (map)                 ✅ 60+ parallel processes
Coordinator (reduce)          ✅ Genetic evolution + meta-strategies
Watchdog                      ✅ Auto-restart on crash
SQLite metrics                ✅ All telemetry persisted
HTML dashboard                ✅ Chart.js, auto-refresh
Markdown dashboard            ✅ For Claude to read
Renderer (decoupled)          ✅ Reads SQLite, writes HTML/MD
GitHub Pages                  ✅ Public docs at gauchoai.github.io
Byte-level tokenizer          ✅ Universal 260-token vocab
Progressive model             ✅ Grows layers on demand
Kernel/cortex split           ✅ Selective freeze, shared embedding
Triton SSM kernel             ✅ GPU-native scan
Weight inheritance            ✅ Compatible children inherit checkpoints
Disk budget (50GB)            ✅ Evicts paused experiments
VRAM management (75% cap)     ✅ Pauses losers to prevent OOM
Grace period (200 cycles)     ✅ New experiments can't be killed early
Per-byte eval                 ✅ Granular teacher feedback
Single-byte outputs           ✅ Eliminated DAME bug
Ratatouille CLI               ✅ Interactive web UI for model
Training strategies doc        ✅ Bilingual EN/ES guide
```

### What we learned (session summary)

1. **Data representation > model size > hyperparameters.** Single-byte
   outputs (data fix) broke a ceiling that 100+ experiments with
   different hyperparameters couldn't break.

2. **Depth > width for algorithmic tasks.** 3 layers of d=64 (116K
   params) massively outperforms 1 layer of d=128 (143K params).
   The model needs sequential processing steps, not wider vectors.

3. **PerpGrad > grokking with clean data.** Grokking was compensating
   for the multi-byte output problem. With clean single-byte outputs,
   PerpGrad's direct approach wins.

4. **Evolution finds architecture.** We didn't design d=64 L=3 — the
   genetic algorithm discovered it by trying d=32/48/64/96/128 and
   L=1/2/3 and seeing what survives.

5. **Transfer is the real metric.** modus_ponens at 71% without any
   training is more impressive than parity at 100% with training.
   The model is building general representations, not task-specific
   circuits.

6. **Meta-evolution is natural.** Every training strategy concept
   (momentum, temperature, focal attention) has a direct analogue
   for managing the experiment population.

---

## Entry 19: GPU Saturation — Lessons Learned the Hard Way

**Date:** 2026-04-22

A full day of experiments on the H100, trying to maximize GPU utilization
and training speed. Key lessons, all learned empirically:

### GPU utilization is a vanity metric

4 subprocess workers showed 99% GPU utilization but each trained at
31s/cycle — 5x slower than 1 worker at 28% GPU doing 1.8s/cycle.
Total throughput was LOWER with 99% GPU. The tiny models (100K params)
can't saturate an H100; the 99% was from CUDA context-switching overhead.

### Threads vs subprocesses vs single-process

| Approach | GPU% | Cycle time | Throughput | Result |
|----------|------|-----------|------------|--------|
| 1 subprocess (sequential) | 28% | 1.7s | Best per-task | 5 teachers in 15min |
| 4 subprocesses | 99% | 31s | Worst | 2 genuine teachers in 10h |
| 4 threads | 25% CPU | 31s | Same as subproc | GIL killed it |
| 8 models in for-loop | 98% | 7s avg | Medium | Slow shootout |

**Verdict:** Sequential is best for small models. GPU% doesn't matter.

### The inherited best_acc bug

When weight inheritance copies a checkpoint from task A to task B, the
child inherited task A's `best_acc=96%`. The pool thought task B had
96% accuracy when it actually had 30%. Caused 3 false graduations.

**Fix:** Reset best_acc and cycle count when loading cross-task checkpoint.

### Models must persist across rounds

The original three_populations.py created a NEW model every round (10
cycles), then threw it away. 19 rounds × 10 cycles = 190 cycles of
training, but across 19 separate models that never accumulated learning.
Loss was literally 0.347 from cycle 1 to cycle 190 — flat.

**Fix:** Save checkpoint after each round, load it on next round.

### Never delete training state

Twice in one session, earned mastery (5 graduated teachers) was destroyed
by deleting the teachers directory as a "quick fix" for code bugs. Each
teacher represents hours of GPU time. The fix is always to fix the code,
never to delete the data.

---

## Entry 20: State Management + External Teachers

**Date:** 2026-04-22

### SQLite state database

All training state now lives in SQLite (`three_pop/training.db`):
- `teachers` table: append-only. First graduation wins. Never deleted.
- `lineage` table: every round, every config, every result. Permanent.
- `experiments` table: full metadata for every champion and challenger.
- `runtime_config` table: hot-reload settings (no process restart needed).

To change training parameters without restarting:
```sql
sqlite3 training.db "UPDATE runtime_config SET value='20' WHERE key='cycles_per_round'"
```
Process reads it next round. No kill, no restart, no lost state.

### Champion-challenger mutations

When a task plateaus (no improvement for 3 rounds), the GA creates a
challenger with a mutated config. The champion keeps training alongside.
The challenger must BEAT the champion's best accuracy to take over.
Otherwise, the champion's checkpoint is restored. No more destroying
91% models with bad mutations.

Results from first champion-challenger rounds:
- parity: Champion 63% held vs challenger 57%
- binary_pattern_next: Champion 92% held vs challenger 92%
- same_different: Champion 87% held vs challenger 56%
- odd_one_out: Champion 74% held vs challenger 17%

All champions held — the mutations weren't good enough yet.

### External teacher experiment: LLMs as teachers

Built `external_teacher.py` with llama.cpp integration. Tested two models
on our 15 reasoning tasks:

| Task | Qwen-Math 1.5B | Mathstral 7B | Our Specialist |
|------|----------------|-------------|----------------|
| arithmetic_next | 0% | **87%** | 30% |
| logic_gate | 43% | **70%** | 100% |
| same_different | 47% | 50% | **87%** |
| binary_pattern_next | 47% | 33% | **92%** |
| parity | 0% | 0% | **63%** |
| modus_ponens | 10% | 0% | **100%** |

**Key finding:** Mathstral 7B beats our specialist on arithmetic_next
(87% vs 30%) and logic_gate (70% vs our training accuracy). But it
can't do parity at all (0%). LLMs understand math sequences but not
bit counting.

**Design:** External teachers become a mutation option. The GA can try
`"teacher_model": "mathstral-7b"` or `"specialist:same_different"`.
Teacher evaluation results are cached — idempotent (compute once, reuse).
Only teachers that surpass our current best for a task are eligible.

### Current state

- 5 teachers graduated: run_length_next, geometric_next, logic_gate,
  logic_chain, modus_ponens
- binary_pattern_next at 94% — 1% from graduating
- same_different at 87%, mirror_detection at 78%
- Parity stuck at 63% across all configs tried
- llama.cpp + Mathstral 7B running on H100 alongside training
- SQLite DB protects all state, Firebase syncs for UI

---

## Entry 21: Teacher-as-Mutation Hypothesis

**Date:** 2026-04-22

### The hypothesis

External teachers (LLMs, our own specialists) can be offered as a
mutation option in the genetic algorithm. The GA discovers which
teacher helps which task, automatically. No manual selection needed.

### Why we believe this will work

1. **Mathstral 7B scores 87% on arithmetic_next.** Our specialist is
   stuck at 30%. The LLM genuinely understands number sequences better
   than our 100K-param model. Its output distributions encode math
   knowledge our model can't discover on its own.

2. **Cross-specialist transfer is real.** run_length_next loaded
   same_different's weights and scored 100% on first cycle (Entry 18).
   Related tasks share representations. A specialist that mastered
   logic_gate might help logic_chain through distillation.

3. **The cache makes it free.** Teacher evaluation is expensive (30
   forward passes per task). But once cached, the GA can check
   instantly whether a teacher is worth trying. Bad teachers get
   rejected from cache — zero wasted compute on repeat evaluations.

4. **Champion-challenger protects against bad teachers.** If Mathstral's
   output distributions confuse more than help, the challenger loses
   and the champion's checkpoint is restored. No damage.

### The mechanism

```
Task plateaus → GA mutation fires
  ↓
15% chance: teacher_model = random([
    "specialist:same_different",
    "specialist:logic_gate",
    "mathstral-7b",
    "qwen-math-1.5b",
    ...
])
  ↓
Check cache: does this teacher beat our current best?
  ├─ Cached at 0% → skip instantly
  ├─ Cached at 87% > our 30% → use it!
  └─ Not cached → evaluate (30 examples), cache result
  ↓
Train challenger with teacher guidance
  ↓
Champion-challenger comparison
  ├─ Challenger wins → new config with teacher
  └─ Champion holds → teacher wasn't helpful in practice
```

### What we expect to see

- **arithmetic_next**: Mathstral teacher should push past 30% ceiling.
  This is our strongest prediction — the LLM has genuine math knowledge
  our small model lacks.
- **logic_gate/logic_chain**: Cross-specialist teaching might help.
  Both are logic tasks, shared representations likely.
- **parity**: No teacher can help (both LLMs score 0%, specialists
  for other tasks don't transfer). Parity needs the curriculum
  approach (progressive difficulty), not a teacher.
- **same_different**: Already at 87%, close to self-graduating.
  External teachers unlikely to help at this level.

### How to verify

Monitor the lineage files. When a teacher_model mutation fires,
the lineage entry will show it:
```
| 8 | C | 45% | 30% | ... | severity=2.0 changes={'teacher_model': 'mathstral-7b'} |
```

If arithmetic_next breaks past 30% with a Mathstral teacher, the
hypothesis is confirmed. If it doesn't improve, the champion holds
and we learn that raw distillation from LLM logits doesn't transfer
to byte-level SSM architectures (also valuable knowledge).

### Current scoreboard

| Task | Our Best | Mathstral 7B | Qwen 1.5B | Best Available Teacher |
|------|----------|-------------|-----------|----------------------|
| arithmetic_next | 30% | **87%** | 0% | Mathstral |
| logic_gate | 100% (teacher) | 70% | 43% | Our specialist |
| same_different | 87% | 50% | 47% | Our specialist |
| binary_pattern_next | 94% | 33% | 47% | Our specialist |
| mirror_detection | 90% | — | — | Our specialist |
| parity | 63% | 0% | 0% | None — needs curriculum |

---

## Open threads

- **binary_pattern_next at 94%.** One percent from graduation.
- **mirror_detection at 90%.** New high — checkpoint resume is paying off.
- **Teacher-as-mutation deployed.** Waiting for GA to try it on stuck tasks.
- **Parity ceiling at 63%.** See Entry 22 — architecture insight.
- **Distillation.** 6 teachers ready. Student can start learning.
- **Cortex development.** Still waiting.
- **Boss tasks.** 18 unseen tasks for generalization eval.
- **Register inspector.** Built, deployed, pushes to Firebase.
- **Stateless orchestrator.** Deployed. Workers self-sufficient.

---

## Entry 22: The Architecture Dimension — What the GA Cannot See

**Date:** 2026-04-23

### The experiment

`parity_experiment.py` — a 7-line model: embed(2→32) + one Mamba3Block + head(32→2).
Feed raw bits in, running parity out. No tokenization, no strings, no bytes.

| Model | Params | Steps | Accuracy |
|-------|--------|-------|----------|
| **Mamba-3** | **8,074** | **400** | **100%** |
| Mamba-2-like | 7,690 | 400 | 49.5% (random) |

400 steps. 8K params. Perfect parity. Position-wise accuracy: 100% at
every position from 0 to 15. Mamba-2-like degrades from 100% at pos 0
to 49.5% at pos 15 — it cannot track state.

### Why our specialist is stuck at 63%

The specialist has the SAME Mamba-3 architecture. Same RoPE, same
trapezoidal gate, same state dynamics. But it's stuck at 63%.

The difference:

**The experiment feeds raw bits:**
```
input: tensor([0, 1, 1, 0, 1])  → 5 tokens, each is 0 or 1
```

**The specialist feeds byte-encoded strings:**
```
input: "0 1 1 0 1" → [48, 32, 49, 32, 49, 32, 48, 32, 49] → 9 tokens
```

The specialist must learn THREE things simultaneously:
1. 48 means "zero" and 49 means "one" (byte decoding)
2. 32 means "space" and should be ignored (noise filtering)
3. Count the ones and output parity (the actual task)

The experiment only needs to learn #3.

### What this means

The model architecture is POWERFUL ENOUGH. 8K params, 1 layer, 400
steps — parity solved perfectly. The SSM's 2,048 registers with
data-dependent decay CAN implement a running XOR.

But the GA cannot change the INPUT REPRESENTATION. It mutates
d_model, layers, lr, optimizer, loss_fn — all training parameters.
It cannot change HOW the data is presented to the model. The byte
tokenization is fixed. The spaces are fixed. The ASCII encoding
is fixed.

This is a blind spot. The GA explores a 15-dimensional config space
but the most impactful dimension — data representation — is not in
the search space.

### The SSM as a computing machine

The Mamba-3 SSM has:
- **2,048 registers per layer** (8 heads × 16 headdim × 16 d_state)
- **Data-dependent decay** — the model controls how much to remember
  vs forget at each timestep
- **Trapezoidal blending** — second-order integration, can look at
  current AND previous input
- **RoPE dynamics** — complex-valued rotations that enable state
  tracking across arbitrary lengths

Parity needs exactly 1 register as a flip-flop. The model has 2,047
more than it needs. The bottleneck is not capacity — it's learning
to ROUTE information through the projection matrices.

The parity experiment succeeds because the embedding is trivial:
embed(2, 32) maps {0, 1} to two 32-dim vectors. The model immediately
knows "this is a zero" vs "this is a one."

The specialist fails because embed(260, 64) maps 260 possible bytes
to 64-dim vectors. The model must discover that bytes 48 and 49 are
special (they represent bits), byte 32 is noise (space), and bytes
256-259 are control tokens. That's a much harder optimization problem
with many local minima.

### What we can do about it

1. **Task-specific tokenization** — for parity, embed bits directly
   instead of going through byte strings. The GA could mutate the
   tokenization scheme as part of the config: `"tokenizer": "raw_bits"`
   vs `"tokenizer": "byte_string"`.

2. **Pre-trained byte embeddings** — initialize the embedding so that
   48→"zero concept" and 49→"one concept" are already separated. The
   model doesn't have to discover this from scratch.

3. **Curriculum on representation** — start with raw bits (easy), then
   gradually shift to byte encoding. The model learns the algorithm
   first, then learns to parse the encoding.

4. **Architecture as mutation** — the GA already mutates d_model and
   layers. It could also mutate the embedding type, the input format,
   or the number of heads. The model architecture itself becomes part
   of the search space.

### The bigger picture

We are teaching this SSM two things at once:
1. **How to compute** — the algorithm (XOR, comparison, counting)
2. **How to instrument** — how to use its own registers, gates, decays

The parity experiment shows that #1 takes 400 steps when #2 is trivial
(raw bits). Our specialists show that #1 + #2 together take 60,000+
steps and often plateau.

This is analogous to teaching someone to play piano AND teaching them
music theory simultaneously. The experiment says "here are the keys,
play this" — 400 steps to mastery. The specialist says "here are some
symbols on paper, figure out what they mean, then play what they
represent" — 60,000 steps and still struggling.

The architecture — the SSM with its registers and gates — is the piano.
It's a beautiful instrument. The question is how to teach someone to
play it efficiently.

### How the SSM actually solves parity (the three-level lock)

When we say "the model learns parity," what physically happens in the
weights is three coupled optimization problems solved simultaneously:

**Level 1 — The B matrix (routing):**
The B projection matrix controls which registers receive which input.
The model cannot say "put this bit in register 7." It has to learn
weight values such that when input 1 arrives, the matrix multiplication
HAPPENS to route a signal to a specific register. With raw bits
(vocab=2), B is a 2×32 matrix — 64 numbers to optimize. With bytes
(vocab=260), B is 260×64 — 16,640 numbers, and 258 of the 260 input
channels are noise the model must learn to ignore.

**Level 2 — The A matrix (decay/memory):**
The A parameter controls how fast each register forgets. The model
must learn that the XOR register should have HIGH decay (close to 1.0,
meaning "hold this value") while scratch registers should have LOW
decay (forget quickly). This is baked into the weights — not a runtime
decision. The model discovers the right decay values through gradient
descent.

**Level 3 — The C matrix (reading):**
The C projection matrix controls which registers contribute to the
output prediction. The model must learn to read from the XOR register
and ignore the other 2,047. If it reads from the wrong register, the
output is noise regardless of whether B and A are correct.

**The coupling problem:**
These three matrices INTERACT. Changing B (routing) changes which
register receives the signal, which changes what A (decay) is relevant,
which changes what C needs to read. It's a coupled optimization over
three matrices simultaneously. With 3 layers, the output of layer 1's
C feeds into layer 2's B — creating a pipeline of
routing→decay→reading→routing→decay→reading where all weight matrices
across all layers must align.

**The phase transition:**
This coupling explains the "flat then sudden" learning pattern we
observe. same_different had zero gradients for 400+ cycles, then
spiked to 9.3, then mastered. The model was wandering in a flat
landscape where B, A, and C were misaligned — no combination of small
adjustments improved the output. Then the tumblers clicked: B learned
to route, A learned to hold, C learned to read, all at once. The
gradient spike IS the moment of alignment.

### The ridiculousness quantified

|                          | Parity (raw bits) | same_different (bytes) |
|--------------------------|-------------------|----------------------|
| Vocab size               | 2                 | 260                  |
| B matrix size            | 2×32 = 64         | 260×64 = 16,640      |
| Noise channels           | 0                 | 258 (97% noise)      |
| Params                   | 8,074             | 103,539              |
| Steps to mastery         | 400               | ~60,000+             |
| Time on H100             | <1 second         | ~6,000 seconds       |
| Accuracy                 | 100.0%            | 95% (just graduated) |

The same architecture. The same mathematical capability. The same
registers and gates. 400 steps vs 60,000 steps. The only difference:
how many useless byte channels the B matrix has to learn to ignore.

And yet — same_different DID graduate. 6,000 seconds of an H100
burning through gradient descent, 400 cycles of zero progress, one
glorious spike to 9.3, and then mastery. The model taught itself to
fly a 2,048-knob cockpit with no manual, no labels, starting from
random noise. That's simultaneously ridiculous and magnificent.

---

## Entry 23: Triton Kernel Bug — CUDA Cannot Learn What CPU Can

**Date:** 2026-04-23

### The experiment

`test_cpu_vs_cuda.py` — same model, same seed, same config, same code.
Only difference: device.

```
Exact same config (d=32, dS=16, hd=16), seed=0, 400 steps

  cpu:  acc_all=100.0%  acc_last=99.7%  loss=0.0014  time=84.8s
  cuda: acc_all= 54.4%  acc_last=51.0%  loss=0.6816  time=1.4s
```

**CPU: 100% accuracy. CUDA: 54% (random). Same model.**

### What this means

The SSM scan has two implementations:
1. `ssm_scan_jit` — Python/PyTorch loop, runs on CPU/MPS. Mathematically
   correct. The parity experiment works perfectly on this.
2. `ssm_scan_triton` — Triton GPU kernel, runs on CUDA. Fast but
   numerically different enough that the model CANNOT learn parity.

The dispatch logic in `ssm_triton.py`:
```python
def ssm_scan(inp, decay, C, x, z, D):
    if inp.is_cuda and HAS_TRITON:
        return ssm_scan_triton(inp, decay, C, x, z, D)  # ← broken
    return ssm_scan_jit(inp, decay, C, x, z, D)         # ← works
```

Every task that trained on the H100 used the Triton kernel. Every task
that struggled for thousands of cycles was fighting BOTH the byte
encoding AND a numerically imprecise scan kernel.

### The 7 teachers that graduated

They graduated DESPITE the Triton kernel, not because of it. The tasks
that converge easily (modus_ponens in 5 cycles, logic_gate in 8 cycles)
are simple enough that numerical imprecision doesn't matter. The tasks
stuck at 62-93% (parity, alternating_next) might be stuck BECAUSE of it.

### Register sizes tested

```
test_register_sizes.py — all on CUDA (Triton kernel):

config                          params  acc_all  acc_last  registers
tiny d=16 dS=2 hd=4              2,122    51.7%    48.4%         64
tiny d=16 dS=4 hd=4              2,210    53.3%    51.3%        128
small d=32 dS=8 hd=8             7,794    53.2%    50.9%        512
orig d=32 dS=16 hd=16            8,074    54.4%    51.0%      1,024
current d=64 dS=16              29,138    53.0%    46.2%      2,048
```

ALL configurations fail on CUDA. Every register size. The issue is not
the model architecture — it's the kernel implementation.

### Possible causes

The Triton kernel (`ssm_triton_kernel.py`) performs the scan:
```
h[t] = decay[t] * h[t-1] + inp[t]
```

In GPU registers using fp32. Possible sources of numerical divergence:
- **Operation ordering**: the JIT computes sequentially in Python;
  Triton may reorder or fuse operations differently
- **Reduction precision**: the output projection `sum(h * C)` may
  accumulate differently on GPU
- **The silu gate**: `y = (y + D*x) * z * sigmoid(z)` is fused in
  Triton but separate in JIT. Fused ops may lose precision.
- **Decay precision**: `exp(A * dt)` computed once vs recomputed.
  Small differences in decay compound over the sequence length.

### The fix: backend as a mutation

Rather than choosing one or the other globally, make the scan backend
a mutation option in the GA config:

```python
{"scan_backend": "triton"}   # fast, possibly imprecise
{"scan_backend": "jit"}      # slower, mathematically correct
```

The GA can try both. For parity, JIT wins (100% vs 54%). For tasks
where precision matters less (logic_gate, modus_ponens), Triton is
fine and 60x faster.

This is added to the config space alongside d_model, layers, lr, etc.
The champion-challenger system protects: if JIT produces better
accuracy, it wins. The lineage records which backend was used.

### The deeper lesson

We spent hours optimizing the training loop — bigger batches, more
workers, smarter mutations. The real bottleneck was a numerical bug
in the innermost computation. The SSM's state update — the one
operation that must be EXACT for state tracking to work — was
approximate on CUDA.

This is why observability matters. Without the CPU vs CUDA comparison,
we would have blamed the byte encoding, the model size, the learning
rate. The actual cause was invisible at the training level — it's in
the kernel implementation, below the abstraction layer.

---

## Entry 24: 14/15 Tasks Mastered — What the GA Discovered

**Date:** 2026-04-23

### The scoreboard

14 out of 15 tasks mastered. Only repeat_count remains at 70%.

### Winning configurations — discovered by the GA, not by humans

```
task                         d  L  dS  backend device   acc
------------------------------------------------------------
alternating_next           144  3  16      jit   cuda 100%
arithmetic_next            192  5  32      jit   cuda  99%
binary_pattern_next         64  3  16   triton   cuda  96%
geometric_next              64  3  16   triton   cuda 100%
logic_chain                 64  3  16   triton   cuda  98%
logic_gate                  64  3  16   triton   cuda 100%
mirror_detection            96  4  16      jit    cpu  96%
modus_ponens                64  3  16   triton   cuda 100%
odd_one_out                 48  2  32      jit    cpu  98%
parity                      64  4   8      jit   cuda 100%
pattern_period              64  8  16      jit   cuda  97%
run_length_next             64  3  16   triton   cuda 100%
same_different              64  3  16   triton   cuda  95%
sequence_completion         96  6  32      jit   cuda 100%
```

### What the GA discovered — three categories

**Easy tasks (6/14): default config is enough**
geometric_next, logic_gate, modus_ponens, run_length_next,
binary_pattern_next, same_different, logic_chain — all mastered
with the original BASE_CONFIG (d=64, L=3, dS=16, triton, cuda).
No mutation needed. These tasks are simple enough that even the
imprecise Triton kernel can solve them.

**Hard tasks needing JIT (8/14): precision matters**
Every task that needed a mutation to master also needed the JIT
backend. NOT ONE hard task mastered on Triton alone. The numerical
precision of the scan kernel is the difference between convergence
and permanent plateau.

**Tasks needing CPU (2/14): GPU arithmetic itself is insufficient**
mirror_detection and odd_one_out needed CPU execution. Even JIT on
CUDA wasn't precise enough. These tasks require exact floating point
behavior that only CPU provides.

### Architecture insights — each task has its own optimal shape

**Parity (d=64, L=4, dS=8):** Needed MORE layers (4 vs 3) but
FEWER registers (dS=8 vs 16). The running XOR needs sequential
processing depth, not state width. Fewer registers = smaller
search space for the B matrix = faster convergence.

**Arithmetic_next (d=192, L=5, dS=32):** Needed EVERYTHING bigger.
Numbers require wide representations (d=192) and lots of state
(dS=32 = 15,360 registers per layer) to track sequences. The GA
discovered this via a severity 3.0 radical mutation.

**Odd_one_out (d=48, L=2, dS=32):** SMALLER model, fewer layers,
but MORE registers. The task needs to store and compare many values
(find the outlier) but the comparison is simple — more state, less
processing.

**Pattern_period (d=64, L=8):** The deepest model. Finding the
period of a repeating pattern requires many sequential processing
steps — the model needs to hypothesize a period length and verify
it across multiple repetitions. 8 layers of pipeline.

**Sequence_completion (d=96, L=6, dS=32):** Wide, deep, lots of
state. Completing sequences requires representing the pattern
(d=96), verifying it (L=6 deep), and tracking position (dS=32).

### The bigger picture

No human designed these configurations. The GA explored the space
through mutation + champion-challenger, and the lineage records
every attempt. The key mutations:

- `scan_backend: "jit"` — unlocked 8 tasks that Triton couldn't solve
- `device: "cpu"` — unlocked 2 tasks that CUDA couldn't solve
- `n_kernel_layers: 4→8` — unlocked tasks needing deeper reasoning
- `d_model: 64→192` — unlocked tasks needing wider representations
- `d_state: 16→32` — unlocked tasks needing more memory

Each of these was discovered through the mutation gate, validated
by champion-challenger, and recorded in the lineage with full
provenance. The system found the right architecture for each task
automatically.

### This is a known problem in the Mamba community

Research confirmed this is not unique to our implementation:

- **Tri Dao** (Mamba's creator) documented that SSM dynamics are
  "very sensitive to numerical precision" — cumulative sums can be
  large, and without the right implementation, "the basic SSD
  algorithm produces NaNs immediately during training."
  (Source: tridao.me/blog/2024/mamba2-part3-algorithm)

- **PyTorch/IBM** documented that fused Triton kernels "internally
  use fp16 for some computations that the original kernels used fp32
  for" and there are "slight differences in output between the fused
  kernel and reference solution that depend on the GPU."
  (Source: pytorch.org/blog/accelerating-mamba2-with-kernel-fusion)

- **The standard fix**: "accumulate partial sums in higher precision
  (fp32) for numerical stability." Our kernel likely accumulates
  the state `h = decay * h + inp` without sufficient precision,
  or the fused silu gate introduces compounding rounding errors.

### What we're doing about it

**Immediate (deployed):** `scan_backend` as a GA mutation. The GA
tries JIT vs Triton per task. JIT is mathematically correct, Triton
is fast. The champion-challenger decides. Already live — the GA
tried JIT for alternating_next (which graduated at 100%).

**Next:** Fix the Triton kernel itself. The audit path:
1. Check if the state accumulation loop uses fp32 throughout
2. Check if the silu gate is fused in a way that loses precision
3. Compare intermediate values (h at each timestep) between
   JIT and Triton for the same input
4. Fix and verify: parity must hit 100% on CUDA

**Community contribution:** Once fixed, this kernel is valuable.
The Mamba community on Hugging Face (kernels-community/mamba-ssm)
is building community kernels. A correct, tested kernel with a
parity regression test would be a real contribution.

---
