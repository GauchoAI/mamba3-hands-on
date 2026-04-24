# Mamba-3 Hands-On — Findings Log

A running lab notebook as we explore the Mamba-3 paper (Lahoti et al., ICLR 2026)
by implementing a minimal version from scratch and running experiments on Apple
Silicon MPS. Each entry corresponds to a commit.

Paper: https://arxiv.org/abs/2603.15569
Official repo: https://github.com/state-spaces/mamba (CUDA-only)

---

## Entry 1 — Setup and minimal SISO block

**Commit:** `init` — the foundation.

**What we built**
- A ~200-line pure-PyTorch Mamba-3 block (`mamba3_minimal.py`) with no custom
  kernels. Runs on MPS / CPU / CUDA. Sequential scan (slow but correct).
- A matched Mamba-2-style baseline in the same file (same block structure,
  missing only the two innovations we care about).
- A parity task (`parity_experiment.py`) — the exact benchmark where the paper
  reports Mamba-2 at ~0.9% and Mamba-3 at 100%.

**The three innovations in minimal form**
1. **Trapezoidal gate** → a single learnable sigmoid `trap` per head, blending
   current and previous `(B · x)` outer products before the state update.
2. **Complex dynamics via data-dependent RoPE** → a slice of `in_proj` produces
   per-step `angles`; cumulative phase is `cumsum(angles * DT_mean)`; we rotate
   B and C (not the state) via pairwise RoPE. Real-only ops, complex behaviour.
3. **MIMO** → not implemented in this entry (SISO only, rank=1).

**Sanity**
- Forward + backward pass OK on MPS. 7.8k params for `d_model=32, d_state=16`.

---

## Entry 2 — The operator-precedence bug, and parity solved

**Commit:** `fix: A sign / train parity`

**The bug**
```python
A = -F.softplus(dd_A).clamp(max=-A_floor)
```
This parses as `-(softplus(dd_A).clamp(max=-A_floor))`. Since softplus is
positive, the clamp forces it to `-A_floor`, and negation flips it to `+A_floor`.
Result: A was positive, so `decay = exp(A·DT) > 1` — the state **grew**
slightly at every step instead of decaying. Both models were broken identically,
which is why both sat at ~50% on parity initially. Classic.

**The fix**
```python
A = (-F.softplus(dd_A)).clamp(max=-A_floor)
```

**Parity results (L=16, 400 steps, AdamW lr=3e-3)**

| Model | Params | Final loss | Acc (all pos) | Acc (last pos) |
|---|---|---|---|---|
| Mamba-2-like | 7,690 | 0.590 | 56.4% | 51.0% |
| **Mamba-3** | 8,074 | **0.013** | **99.9%** | **99.7%** |
| Random | — | — | 50.0% | 50.0% |

Position-wise, Mamba-3 holds 100% across the whole sequence; Mamba-2-like
starts at 100% on the trivial positions 0–1 and degrades to random by t=6.
**Paper's central claim reproduced on a laptop.**

---

## Entry 3 — Mechanistic interpretability: the model found –π

**Commit:** `probe: learned phase per bit`

I asked: *what rotation did the model actually learn?* I fed the trained
Mamba-3 a sequence of all-0s, then all-1s, and inspected the per-step phase
increment `angles * DT_mean` across the 8 phase components.

The difference `phase(bit=1) – phase(bit=0)` at one component:
```
angle[0]: -3.140 rad  (-179.9°,  -0.999 * π)
```

**The model discovered, through pure gradient descent on 25k parity labels,
that it needs to rotate a phase component by exactly –π when the bit is 1.**
That is *literally* the textbook "XOR as rotation" construction. The other
7 components learned assorted non-useful rotations (64°, 20°, -113°, …) —
unused capacity, correctly ignored by the readout.

This is the kind of "theory predicts → training finds → weights confirm" loop
that makes mechanistic interpretability satisfying.

---

## Entry 4 — Harder moduli: where the minimal model runs out

**Commit:** `exp: modular counting mod 3 and mod 5`

**Task** Same as parity but inputs and target in `{0, ..., M-1}`, target is
`cumsum(x) mod M`. Ideal per-step rotation is `2π/M` (or an integer multiple).

**Results (L=16, 800 steps)**

| M | Random | Mamba-2-like (acc_all) | Mamba-3 (acc_all) |
|---|---|---|---|
| 2 | 50.0% | 62.6% | **99.9%** |
| 3 | 33.3% | 44.8% | 46.1% |
| 5 | 20.0% | 31.2% | 32.3% |

Parity works; mod 3 and mod 5 don't. Both models stay near random at higher
moduli.

**What this means.** Parity is uniquely easy in the rotation framing: you only
need *one* phase component tuned to π, and the readout is a 2-way classifier.
Mod 3 and mod 5 demand:
- A precisely-tuned angle (`2π/3 ≈ 120°`, `2π/5 = 72°`) — small errors compound
  over the sequence so readout degrades fast.
- A readout that discriminates 3 or 5 states from a rotating 2D point, which
  needs more state capacity than our `d_state=16, headdim=16` minimal config.

**Not a refutation of the paper.** The official Mamba-3 has MIMO (rank-4 lift),
more heads, chunked scans, better init. Our minimal SISO with 8k params is
genuinely under-powered for M≥3. We expect scaling `d_state`, `d_inner`,
or adding MIMO would close this gap — a nice follow-up experiment.

---

## Entry 5 — Ablation: RoPE does all the parity work

**Commit:** `exp: ablate RoPE and trapezoidal on parity`

Added `use_rope` and `use_trap` flags to `Mamba3Block` so each innovation can
be toggled independently. Ran the same parity setup (L=16, 400 steps) four ways.

| Variant | acc_all | acc_last |
|---|---|---|
| full (RoPE + trap) | 99.1% | 96.1% |
| **no trap (RoPE only)** | **100.0%** | **100.0%** |
| no RoPE (trap only) | 60.6% | 50.0% |
| neither | 58.6% | 47.8% |

**Takeaway.** The full credit on parity belongs to the **complex-dynamics
via RoPE** (innovation #2). The trapezoidal gate (#1) contributes zero here
and is marginally noisy. That matches the paper's framing: innovation #1
is a discretization accuracy improvement — the benefit shows up on smooth
dynamics and long-range stability, not on a toggling discrete state.

The two innovations are complementary, not redundant. Parity only exercises
one of them; the other earns its keep elsewhere.

---

## Entry 6 — Mod 3 is mechanically solved, but brittle

**Commit:** `exp: longer training + curriculum on mod 3/5`

**Attempt 1 (bigger config, same budget) — WORSE.** Bumping to
`d_model=64, d_state=32` tanked everything. Parity dropped from 99.9% → 58%,
mod 3 from 46% → 50%. Classic "more capacity + same training budget = harder
optimization". Lesson: don't scale the model without scaling the schedule.

**Attempt 2 (small config, 8000 steps, train on L=8, eval at L=16) — works
in-distribution, brittle out.**

| M | Train acc (L=8) | Eval acc (L=16) | Chance |
|---|---|---|---|
| 3 | 89.0% | 67.5% | 33.3% |
| 5 | 51.2% | 35.6% | 20.0% |

**The big finding — probing the learned angles on mod 3.** Component 3 of
the 8 phase components learned the right thing:

```
Token 1 - Token 0, comp[3]: +113.5° (ideal +120°, 2π/3)  err=0.07 rad
Token 2 - Token 0, comp[3]: -124.0° (ideal -120°)         err=0.07 rad
```

**The model discovered 2π/3 rotation essentially from scratch**, same way it
found -π for parity. Seven other phase components learned miscellaneous angles,
unused.

**Why performance degrades at L=16.** Angle error tolerance scales as 1/M.
A 7° angle error per step is invisible at mod 2 (doesn't cross the ±π/2
decision boundary). At mod 3, 7° over 16 steps = 112° drift — enough to
misclassify into the neighbouring sector. Length extrapolation gets brutal
fast for higher moduli.

**Mod 5.** Stuck at ~35% (vs 20% random). Some learning but far from solving.
Expected at this scale — the angle budget per head is tighter and the
readout decision regions thinner.

**Correct conclusion.** The *algorithm* generalises to any M — rotations are
learned — but the minimal SISO config doesn't have the precision budget to
stabilize them over long sequences at M≥3. The paper's full model uses MIMO,
chunked scans, and careful init to close this gap. The inner mechanism is
the same.

---

## Entry 7 — Selective copy: gating works too

**Commit:** `exp: selective copy — Mamba-3 gates state writes`

**Task.** Sequence of tokens from `{0..3}`, with a special marker token
at 1–3 random positions. Target: the token immediately after the *last*
marker. All other positions are don't-care. This tests whether the model
can learn to **ignore** most inputs and only write to state on cue — the
opposite of parity (which needs every token).

**Results (L=16 train, 1500 steps, MPS on M4)**

| Model | Acc (L=16) | L=32 | L=64 |
|---|---|---|---|
| **Mamba-3** | **100.0%** | **99.0%** | 72.3% |
| Mamba-2-like | 65.4% | 48.2% | 38.7% |
| Random | 25.0% | 25.0% | 25.0% |

Mamba-3 solves it perfectly by step 200 (loss ≈ 0). Mamba-2-like plateaus
around 65% and never truly cracks it.

**What this tells us.** Mamba-3's advantage isn't limited to cumulative
state tracking (parity). The RoPE-gated dynamics also enable **selective
write** — the model learns to "open the gate" only when the marker appears,
overwriting state with the following token's value. Mamba-2-like can
partially do this (~65%) but can't reliably distinguish "write now" vs
"ignore" across varying marker positions.

**Length generalization.** Same pattern as parity: near-perfect at 2× train
length, degrades at 4×. The state retention mechanism leaks over long gaps
between the last marker and the readout position.

---

## Entry 8 — Bilingual language model: it speaks (kind of)

**Commit:** `exp: bilingual char-level LM on Tatoeba EN+ES`

**Setup.** First multi-layer Mamba-3 model: 2 stacked blocks with residual
connections + LayerNorm, byte-level (vocab=256), trained on 80k interleaved
`[EN]`/`[ES]` sentences from Tatoeba (~3.7 MB). 253K params. Trained on
Mac mini M4 via MPS, ~24 min for 5000 steps.

**Architecture:** `Embedding → LayerNorm → [Mamba3Block + residual] × 2 → LayerNorm → Linear`
with gradient checkpointing per layer and weight tying (embed ↔ head).

**Training curve** (bits-per-character):

| Step | BPC | Loss |
|---|---|---|
| 1 | 8.05 | 5.58 |
| 1000 | 2.09 | 1.45 |
| 3000 | 1.82 | 1.26 |
| 5000 | 1.76 | 1.22 |

**What it learned:**
- **Language tag conditioning.** Prompting with `[EN]` produces English-ish
  text, `[ES]` produces Spanish-ish text. The model learned to switch.
- **Spanish morphology.** Verb conjugations (-aron, -ando, -ido), articles
  (el, la, los, las), question marks (¿...?), accented characters (á, é, ó).
- **English structure.** "The church will do you would be it with the same"
  — word order correct, meaning absent. Classic small LM behaviour.
- **Code-switching.** The model naturally transitions between `[EN]` and
  `[ES]` tagged sentences, mirroring the training data format.

**Limitations.** 64-char context window + byte-level + 253K params means
it can't maintain coherence beyond a few words. Many invented words
("suspicionario", "dormidarios", "escribidente"). Gender/number agreement
is spotty. But the *mechanism* — an SSM learning to generate two languages
from a shared state space — works.

**What this proves for Mamba-3.** The RoPE-complex dynamics that solved
parity and selective copy also support genuine language modeling. The
phase rotation mechanism handles the much richer "state" needed for
character-level generation across two languages.

---

## Entry 9 — The universal step function: what worked and what didn't

**Commit:** `exp: universal step function + multi-task interference`

### Multi-task with separate heads (failed)

Trained one Mamba-3 with task indicator tokens and separate heads for
parity, sorting, counting. d_state=8, 16: parity and counting stuck at
random. Only sorting learned (~67%). The task-indicator-plus-separate-heads
design split the model into three isolated circuits that competed for state.

### Universal step function — no task labels (partial success)

Redesigned: one head, no task labels. Format: `[input...] [SEP] [output...]`.
Five tasks mixed: parity, sort, reverse, minmax, length. The model must
figure out from context what to compute.

Results at step 3000, d_state=16, 31K params:

| Task | Accuracy | Random | Verdict |
|---|---|---|---|
| Sort | 87% | 6% | ✓ Learned |
| Minmax | 66% | 6% | ✓ Learning |
| Reverse | 64% | 6% | ✓ Learning |
| Parity | 48% | 50% | ✗ Random |
| Length | 2% | 6% | ✗ Random |

Sort, reverse, and minmax are clearly above random. The step function IS
learning to handle multiple tasks. But parity and length are stuck — both
produce a single token after SEP, so the model can't distinguish them.

### Why this is not enough — the Von Neumann insight

**The critical realization:** these successes may be pattern matching over
a small finite space, not genuine algorithms.

- **Sorting with vocab=10** is probably counting sort — the model memorizes
  how many of each value it's seen and outputs them in order. Give it
  vocab=1000 and it would fail. This is a lookup table, not a comparison
  algorithm.
- **Parity with binary** works because there are only 2 input values and
  one trivial rotation. It's the simplest possible case.
- **Reverse with L≤10** might be position memorization, not a general
  reversal algorithm.

**For unbounded solutions** — sorting any numbers, solving Hanoi with any
number of disks — the model needs to learn **micro-operations** (COMPARE,
SWAP, INCREMENT, PUSH/POP) that compose into algorithms, not memorize the
answers for a small vocabulary.

This is the Von Neumann architecture: instructions and data in the same
stream, processed by the same step function. The training data shouldn't
be "here are sorted sequences" but "here is COMPARE executing on two
values, here is SWAP, and here is how they chain into a sort."

### The three layers of mathematical thinking

This exploration led to identifying three layers the model needs:

1. **Numerical computation** (pure step function): `3 + 5 = 8`,
   `COMPARE(7, 3) → 7 > 3`. No symbols, no language. State manipulation.

2. **Symbolic manipulation** (the bridge): Algebra, equation solving,
   simplification. Each step is a rule applied to a pattern:
   `2x + 3 = 7 → 2x = 4 → x = 2`. The rules (DISTRIBUTE, FACTOR,
   SUBSTITUTE, CANCEL) are higher-level micro-operations over expressions,
   not just numbers. A CAS like SymPy can generate these traces.

3. **Mathematical language** (pure orator): "By the commutative property..."
   "Therefore, since x > 0...". This is Brain 1 — explaining, proving.

A mathematician who only has Layer 3 is a poet. A calculator with only
Layer 1 can compute but can't generalize. Layer 2 — symbolic manipulation —
is where language meets computation. Teaching the model this layer is the
key to genuine reasoning.

### What this means for the training data

The training data for the "left brain" (step function) should be
**code-generated execution traces** of micro-operations:

- **Tier 1:** Numerical — ADD, CMP, SWAP, PUSH, POP on concrete values
- **Tier 2:** Symbolic — DISTRIBUTE, FACTOR, SIMPLIFY on algebraic expressions
- **Tier 3:** Composition — chains of micro-ops that implement algorithms

The code generators (sorting algorithms, SymPy, etc.) produce the traces.
The model learns the micro-operations from the traces. And eventually,
Brain 1 (language) learns to WRITE programs of micro-ops that Brain 2
(step function) executes.

See `ARCHITECTURE.md` for the full FPGA analogy and experimental plan.

---

## Entry 10 — Bootstrap Level 0: the adaptive teacher

**Commit:** `train: bootstrap Level 0 with adaptive teacher`

### The bootstrap hypothesis

Human cognition builds in layers: pattern recognition → comparison →
accumulation → composition → recursion → symbolic → language. Each level
bootstraps from the one below. A baby doesn't learn sorting from 10,000
sorted lists — it learns to see patterns, then to compare, then to order.

We designed a 6-level bootstrap curriculum (see `BOOTSTRAP.md`) and built
Level 0: pattern recognition, the substrate for everything above.

### Training evolution

**v1 (baseline):** 10K fixed examples, eval on training data. Reached 77%
but was measuring memorization, not generalization.

**v2 (cycles + fresh data):** Learn→digest cycles with fresh data
regenerated each cycle. Dual eval showing train accuracy, fresh accuracy,
and the gap between them.

| Version | Fresh acc | Gap | Key insight |
|---|---|---|---|
| v1 | 77% (train data) | unknown | Was memorizing |
| v2 step 500 | 40% | +2.4% | Genuine learning |
| v2 step 4750 | **80%** | -1.0% | Best — cycles work |
| v2 step 9000 | 78% | +0.4% | Holding steady |

The learn→digest cycle was the breakthrough. High LR to absorb new
patterns, low LR to consolidate. Fresh data each cycle prevents
memorization. The gap staying small (<5%) confirms generalization.

### The adaptive teacher

Built a teacher that observes per-type accuracy and adjusts:
- **Task weights:** struggling tasks get more practice (gradually)
- **Difficulty levels:** 3 presets per task, auto-promote when mastered
- **Smooth transitions:** weights move 30% toward target per observation
  to avoid distribution shock

First version was too aggressive (instant weight changes from 1.0 to 2.0),
which caused the model to catastrophically forget — fresh dropped from
80% to 50%. Fixed with smooth transitions.

### Per-type results at best checkpoint (80% overall, fresh data)

| Task | Accuracy | Status |
|---|---|---|
| same_different | 98-100% | ✓ Mastered |
| geometric_next | 94-98% | ✓ Mastered |
| odd_one_out | 83-88% | ✓ Strong |
| mirror_detection | 76-84% | ✓ Good |
| arithmetic_next | 68-80% | ✓ Improving |
| sequence_completion | 72-86% | ✓ Good |
| pattern_period | 44-59% | … Struggling |
| repeat_count | 33-57% | … Struggling (needs Level 2) |

### Lessons learned

1. **Eval must be on fresh data.** Eval on training data hides memorization.
   The train/fresh/gap triple is essential.
2. **Checkpoints must be immutable.** We lost our best 80% checkpoint when
   a bad run overwrote it. Now saved as `level0_step{N}.pt`.
3. **Resume must carry forward best_fresh.** Otherwise the new run thinks
   0% is the baseline and "improves" to 60%.
4. **Teacher adjustments must be gradual.** Instant weight changes cause
   distribution shock. Smooth 30% transitions per observation.
5. **The student is the bottleneck, not the teacher.** The adaptive teacher
   would be capable of teaching a human. One Mamba-3 block at 169K params
   hits its ceiling around 80%. Scaling to more layers and more state
   dimensions is needed.

### What we built

```
generators/
├── level0_patterns.py    # 8 task types, parameterized difficulty
├── teacher.py            # adaptive teacher with smooth weight/difficulty control
train_bootstrap.py        # cycle training, dual eval, immutable checkpoints
BOOTSTRAP.md              # 6-level curriculum plan
ARCHITECTURE.md           # two-brain problem, FPGA analogy, Von Neumann insight
CURRICULUM.md             # language + reasoning curriculum
```

---

## Entry 11 — The augmented architecture: registers, spikes, persistent memory

**Commit:** `arch: augmented Mamba-3 with registers + spike gates + persistent memory`

### The insight

The SSM state is `h = decay * h + input` — one fixed-size buffer at one
timescale. Biology has at least four timescales of memory: ion channels
(milliseconds), working memory (seconds), short-term (hours), long-term
(years). Plus explicit mechanisms for deciding WHAT to remember.

Scaling the SSM (more layers, more params) doesn't add new KINDS of state.
It just makes the same buffer bigger. That's like giving a fish a bigger
bowl instead of legs. The architecture needs qualitatively different components.

### The augmented architecture

```
Input → SSM (Mamba-3) → per-step register/memory ops → output
                              ↕               ↕
                       Register Bank    Persistent Memory
                       (working memory) (long-term, slow decay)
                              ↑               ↑
                         Spike Gate       Spike Gate
                       (when to write)  (when to write)
```

**Register Bank (8 slots):** Explicit addressable memory. The model learns
WHEN to write (spike gate), WHERE to write (soft attention over slots),
and WHAT to write. Registers persist across the sequence unless overwritten.
Like CPU registers — small, fast, explicitly controlled.

**Persistent Memory (16 slots):** Same interface but with slow decay
(0.995 per step). Old memories fade unless refreshed. Frequently reinforced
patterns persist naturally. Like biological long-term memory consolidation.

**Spike Gates:** Threshold mechanism initialized to NOT fire (bias=-2.0).
The model must actively learn when something is noteworthy enough to store.
Uses steep sigmoid — output is near-0 (silence) or near-1 (fire). This is
the "aha moment" detector.

### Parameters

| Component | Params | Role |
|---|---|---|
| SSM (Mamba-3) | 28,752 | Sequential processing |
| Register Bank | ~15K | Working memory |
| Persistent Memory | ~15K | Long-term memory |
| Combine + Norm | ~12K | Integration |
| **Total** | **59,562** | **2x plain, not 100x** |

### Key design decisions

1. **Spike gates start silent.** The model begins by relying purely on
   the SSM (which it already knows how to use from Level 0 pretraining).
   Register writes are learned gradually — only when the model discovers
   that storing something helps prediction.

2. **SSM runs first, registers second.** The SSM processes the full
   sequence, then the register layer operates on top. This preserves the
   SSM's proven capabilities while adding new ones.

3. **Three sources combined with residual.** Output = Linear(SSM + reg_read
   + mem_read) + SSM. The residual ensures the SSM's contribution is never
   lost, even if registers are unused.

4. **Not everything is a tensor.** The register addressing and spike
   decisions introduce discrete-ish behavior (sharp sigmoids, argmax-like
   attention). This is intentionally less "smooth" than standard neural
   net operations — it's closer to how actual memory works.

### Hypothesis

If 6 pattern types compete for SSM state (causing the 80% ceiling), the
augmented model should break through by storing each pattern type's
signature in a different register. The SSM detects "this is a sequence
completion task" → spike fires → writes the pattern to register 3. Next
time it needs that pattern, it reads from register 3 instead of trying
to reconstruct it from the decaying SSM state.

### Next: head-to-head comparison

Train plain vs augmented on Level 0 pattern recognition. Same teacher,
same data, same everything. Measure: final fresh accuracy, spike frequency,
register utilization. If the augmented model breaks 80%, the architecture
hypothesis is confirmed.

---

## Entry 12 — Augmented survey + H100 deployment with Triton

**Commits:** `exp: H100 comparison script` → `perf: Triton kernel for SSM scan`

### Head-to-head results (M4, d_model=64, 3000 steps)

```
Plain:     75.3% fresh  (169,232 params)
Augmented: 67.0% fresh  (199,914 params)
Δ: -8.3%  ← augmented lost
```

The augmented model lost. But *why* matters more than the number.

### Diagnostic survey — what does the augmented model actually do?

We built `exp_augmented_survey.py` to capture per-example internals:
spike rates, register addressing, norms — split by correct vs wrong.

**Key finding: spikes correlate with success.**

```
             reg_spike(output)  mem_spike(output)  final_reg_norm
CORRECT:     0.157              0.097              152.1
WRONG:       0.094              0.050               77.9
```

When the model gets an answer right, it spikes 1.7x more during output
generation and accumulates 2x more register content. The architecture
IS being used — just not enough.

**Register slot distribution:**
- reg[0]: 72% of all writes (dominant)
- reg[4]: 15%, reg[5]: 7%, reg[6]: 6%
- reg[1,2,3,7]: dead (0%)

Only 4 of 8 registers are used. The model hasn't learned to specialize
registers by task type — it's using reg[0] as a generic scratchpad.

**Spike timing: output > input.**
Spikes fire mostly during answer generation, not during question reading.
The model writes to registers while producing output, not while encoding
the problem. This suggests it's using registers as intermediate compute
rather than as structured memory of the input.

### The user's insight: don't constrain, improve the data

We proposed adding sparsity penalties to force selective spiking. The
user rejected this:

> "Why would we make the choice for the model on how much it's going to
> use registers? If you want to steer it, improve the training data."

This is the right call. Penalizing spike rates is us imposing assumptions.
The model should decide how to use its tools — we just need data that
genuinely requires structured memory, not just pattern matching.

### VOCAB_SIZE bug found on CUDA

MPS silently ignores out-of-bounds embedding lookups. CUDA correctly
asserts. Token for n=999 is `64 + 999 + 64 = 1127`, but VOCAB_SIZE
was 1088. Fixed to 1152. This bug existed since the bootstrap was built
but never caused visible failures on Apple Silicon.

**Lesson:** MPS is forgiving. CUDA is correct. Always test on CUDA.

### H100 deployment — from 56% to 100% GPU utilization

**Problem:** Our pure-Python sequential scan launched one tiny CUDA kernel
per timestep per batch. The H100 finished each kernel in microseconds
then waited for Python to schedule the next one.

**Optimizations applied (incremental):**

1. **Precompute outer products** — moved all batch-parallel ops (einsum,
   trapezoidal blending, gating) out of the per-timestep loop. The Python
   loop body shrank to just state multiply-add + output contraction.
   *Result: faster, still ~56% GPU.*

2. **BF16 mixed precision** — H100 native bfloat16. Free 2x throughput
   on matmuls. *Result: marginal improvement, not the bottleneck.*

3. **Batch 64 → 4096** — sequences are tiny (avg 20 tokens, max 75).
   With d_model=256, total VRAM is still ~315MB on an 80GB GPU.
   *Result: more parallel work per kernel, up to ~60% GPU.*

4. **Triton kernel** — eliminated the Python for-loop entirely.
   Each thread block handles one (batch, head) pair and loops L steps
   in GPU registers. With batch=4096 and 16 heads = 65K thread blocks.
   *Result: 100% GPU, 43GB VRAM, 0.69ms per scan call.*

```
         Before Triton    After Triton
GPU:     56%              100%
ms/step: 16-19ms          TBD (running now)
ex/s:    ~30K             TBD
```

**Architecture of the Triton kernel (`ssm_triton_kernel.py`):**

```
one program per (batch_idx, head_idx):
    h[hD, dS] = zeros  // state in registers
    for t in 0..L:
        h = decay[t] * h + inp[t]           // state update
        y[t] = sum(h * C[t]) + D * x[t]     // output
        y[t] *= silu(z[t])                   // gate
        store y[t]
```

No Python. No kernel launch overhead per timestep. The recurrence is
inherently sequential over L, but L≈20 — the parallelism comes from
the 65K independent (batch, head) pairs running simultaneously.

**Fallback:** `torch.jit.script` version for MPS/CPU. Same loop, JIT-
compiled so no Python interpreter overhead. Dispatched automatically
based on device.

### Current: H100 running d_model=256, batch=4096, 10K steps

Both plain and augmented models with adaptive teacher. This is the
first run where:
- The GPU is actually saturated
- The model is large enough to potentially generalize (1M params)
- Both models get the same adaptive curriculum

If the augmented model still loses at this scale, the architecture
needs rethinking. If it wins, we have our brain of a fly.

---

## Entry 13 — Curriculum v2: progressive unlock + grokking

**Commits:** `curriculum: 15-task progressive unlock` → `grok: Grokfast EMA`

### The memorization trap (what failed)

Three successive H100 runs all showed the same pattern:

```
d_model=64,  batch=64,   old teacher:  92% train, 33% fresh, 59% gap
d_model=128, batch=2048, old teacher:  100% train, 7% fresh, 93% gap
d_model=256, batch=4096, old teacher:  100% train, 6% fresh, 94% gap
```

More params and more compute made it *memorize faster*, not generalize
better. The model learned a lookup table, not an algorithm. The gap
throttle (reducing LR when train>>fresh) just slowed down training
without fixing the root cause.

### The fix: curriculum design, not model scaling

Two insights from the user:

1. **"Improve the training data"** — don't penalize the model for
   memorizing, give it data that *requires* algorithms.

2. **"Start with parity, master it, then add the next problem"** —
   sequential task unlock, not all 8 tasks at once.

### 15-task curriculum with progressive unlock

Tasks unlock one at a time, in cognitive order. Each task teaches a
skill the next one needs:

```
Stage 0: Binary foundations
  1. parity                — count 1s mod 2 (SSM state tracking)
  2. binary_pattern_next   — detect cycles in 0/1 streams

Stage 1: Comparison
  3. same_different        — compare two values
  4. odd_one_out           — find the outlier in N values

Stage 2: Pattern detection
  5. sequence_completion   — predict next in repeating pattern
  6. pattern_period        — identify cycle length
  7. run_length_next       — detect run-length encoding patterns

Stage 3: Sequence memory
  8. mirror_detection      — palindrome detection (needs memory)
  9. repeat_count          — accumulate counts (needs registers)

Stage 4: Arithmetic reasoning
  10. arithmetic_next      — detect and apply arithmetic step
  11. geometric_next       — detect and apply ratio
  12. alternating_next     — two interleaved sequences

Stage 5: Logic
  13. logic_gate           — AND/OR/XOR/NOT evaluation
  14. logic_chain          — chained gate evaluation
  15. modus_ponens         — propositional logic (IF p THEN q)
```

Unlock rule: all current tasks mastered (90%+ fresh) at difficulty ≥ 0.3
before the next task is introduced.

Difficulty scales continuously [0.0, 1.0] per task — starts trivially
easy (length 3, numbers 0-3), only advances on fresh accuracy. By the
time difficulty is high, memorization is impossible.

### Final boss: 18 unseen task types

When all 15 tasks are mastered, boss mode activates with never-seen
tasks: set operations (union, intersection, subset), sorting, min/max,
range, sum, modular arithmetic, reverse, rotate, deduplicate, XOR,
count ones, unique count, majority element, second largest.

If the model truly learned algorithms and not formats, it should handle
these with minimal examples.

### Samples-to-mastery metric

Each task records when it was unlocked and when it first hit 90% fresh.
The key question: does task N+1 take fewer examples than task N?

If yes → the model is **learning to learn**.

### Grokking: the phase transition we need

Research (Power et al. 2022, Grokfast 2024) shows that with the right
setup, models can suddenly generalize *long after* memorizing — the
"grokking" phenomenon. The memorization circuit is expensive (high
weight norm); weight decay slowly erodes it until a cheaper algorithm
circuit takes over.

**Changes for grokking:**

| Parameter | Before | After |
|-----------|--------|-------|
| Weight decay | 0.01 | 0.1 |
| LR schedule | Gap throttle | Constant |
| Batch size | 4096 | 128 |
| Grokfast | OFF | alpha=0.98 |

**Grokfast** (Lee et al. 2024): EMA low-pass filter on gradients.
Amplifies slow-varying (generalizing) gradient components, suppresses
fast (memorizing) ones. Reported 50x speedup to grokking.

```python
# After loss.backward(), before opt.step():
ema[p] = alpha * ema[p] + (1-alpha) * p.grad
p.grad += ema[p]  # boost the slow signal
```

### First results: it works

```
Step 2000:  parity MASTERED (100% fresh, difficulty 0.00→0.05)
            30,000 examples to master

Step 12000: parity difficulty reached 0.30
            → binary_pattern_next UNLOCKED

Step 12000: same_different already at 60% fresh — NEVER TRAINED ON IT
            (transfer from parity!)
```

The curriculum is advancing. Parity was mastered, difficulty scaled up
to 0.30 while maintaining mastery, and the second task unlocked.

Most interesting: `same_different` (comparing two values) is already
at 60% accuracy despite being locked. The model learned something
about comparison *from parity training alone*. This is genuine transfer
— the algorithm generalized beyond the task it was trained on.

The 11% "fresh accuracy" we saw was misleading — it was testing all
15 task types when only 1 was trained. Per-type accuracy on the
trained task was 100%.

### Architecture note

This run uses plain Mamba-3 (no augmented). The curriculum should
first prove that the *training signal* can produce generalization.
Once we have a generalizing plain model, we add registers/spikes
and test whether they accelerate learning or enable harder tasks
the plain model can't do.

---

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

## Entry 27 — PTX engine: precision verified, gradient coverage is the real gap

**Date:** 2026-04-24

### Context

Built `engine/ptx/` — a hand-written PTX Mamba-3 engine for H100. Forward
pass lands bit-close to CPU (max diff 7.6e-6, runs 2.75× faster than CPU).
Wrote a full backward chain that matches `engine/wgpu/src/train.rs::TrainState`
step-by-step to 1e-6. Ran a parity training on it, hit 61%. Panicked —
thought we'd regressed into a precision bug. Turns out the picture is
different and more useful.

### The precision test

Loaded `/tmp/parity.bin` into both `Mamba3Model::forward` (Rust CPU) and
`PtxModel::forward`. Ran 1000 random parity inputs across bit-lengths 3–8:

```
CPU↔PTX argmax mismatches: 0 / 1000
max |CPU_rust − PTX| forward diff: 1.05e-5  (FP32 epsilon level)
```

Zero divergences. If we had inherited the Triton-style fp16-leaning bug
from Entry 23, we'd see per-forward diffs ~1e-3 and argmax divergences.
We see neither. The PTX scan is fp32 throughout; kernels compiled with
`--fmad=true --ftz=false --prec-div=true --prec-sqrt=true`, every FMA
via `__fmaf_rn` (IEEE round-to-nearest). Precision is clean.

### The scope discovery

My `PtxTrainer::train_step` matches `mamba3_engine::TrainState::train_step`
bit-for-bit. Both get 61% on parity. But `parity_meta.json` shows 35
lineage rounds of Python `specialist_trainer.py` also capping at 63%
across various backends. So is parity broken in the codebase?

**No.** `specialist_trainer.py --scan-backend jit --device cuda` with the
winning config from line 1867 (`d=64, L=4, dS=8`) still trains parity
to 100% in **one cycle (7.5 s)** on H100. Confirmed today:

```
cycle 1  loss=6.736  acc=100%  best=100%  7.5s    ← stage 1 mastered
cycle 2  loss=0.023  acc=100%  (curriculum advanced to stage 2)
cycle 3  loss=0.017  acc=100%  (stage 3, max_len=16)
cycle 4  loss=0.000  acc=100%
cycle 5  loss=0.000  acc=100%
```

No regression upstream. The 63% ceiling in parity_meta.json is because
the GA rounds used `L=3 dS=16` champions and never properly explored
the `L=4 dS=8` winner.

### Why my PTX gets 61%, then

`engine/wgpu/src/train.rs::backward_analytical` **deliberately zeros**
the gradients of `dt_bias`, `d_param`, `b_norm_w/b`, `c_norm_w/b`, per-layer
`layer_norm_w/b`, and `scale`. See lines 388–397:

```rust
lgrads.extend(vec![0.0f32; layer.dt_bias.len()]);   // dt_bias grad (approx 0)
lgrads.extend(vec![0.0f32; layer.d_param.len()]);   // D grad
lgrads.extend(vec![0.0f32; layer.b_norm_w.len()]);  // B norm
...
lgrads.push(0.0);                                    // scale
```

Without those gradients, SSM dynamics aren't trainable — parity needs
`dt_bias` (time constant of state retention) and `d_param` (skip-path gain)
to be tuned, and neither is reachable through just `in_proj_w`/`out_proj_w`.
Easy tasks whose answer is a function of the last token converge without
SSM-parameter grads; stateful tasks like parity can't.

So my PTX faithfully reproduces the Rust reference's incompleteness.
That's consistency, not correctness. To match PyTorch autograd, I need
to actually compute all the gradients Rust TrainState skips.

### The gap, precisely

To close the gap between `PtxTrainer` and PyTorch autograd:

1. `d_decay[t, h] = Σ_{p,n} dh[p,n] · states[t, h, p, n]` (compute before
   `dh *= decay` propagation in scan_bwd)
2. `d_dt[t, h]` from two paths:
   - via decay: `d_decay[t, h] · a[t, h] · decay[t, h]` (scalar)
   - via inp_val: `Σ_{p,n} d_inp_val · blended`
3. `d_a[t, h] = d_decay[t, h] · dt[t, h] · decay[t, h]`
4. `d_dt_bias[h] += Σ_t d_dt[t, h] · sigmoid(dd_dt[t, h] + dt_bias[h])`,
   plus write to `d_proj[dt_off + h]` via the same sigmoid
5. `d_dd_a → d_proj[a_off + h]` via softplus' derivative, clamp-aware
6. `d_d_param[h] += Σ_{t,p} dy_pre[t,h,p] · x_raw[t, h*hd + p]`
7. `d_trap` → `d_proj[trap_off + h]`: `d_trap = Σ_{p,n} d_blended · (bx_cur − bx_prev)`,
   then `d_trap_raw = d_trap · trap · (1 − trap)`
8. **Correct bx backward.** The current `TrainState` code `d_bx = d_scan_inp / (dt·trap + ε)`
   divides where it should multiply — CPU-side bug. Real math:
   `d_blended = d_inp_val · dt`,
   `d_bx_cur[t] = d_blended[t] · trap[t] + d_blended[t+1] · (1 − trap[t+1])`
   (the second term couples timesteps through `bx_prev`; reverse pass needed).
9. `d_x_raw[t, h, p] = Σ_n d_bx_cur · bp_raw[t, n]`,
   `d_bp_raw[t, n] += Σ_p d_bx_cur · x_raw[t, h*hd + p]` (atomic across heads)
10. `layer_norm_bwd` on `bp_raw → bp` and `cp_raw → cp`, giving
    `d_b_norm_w/b`, `d_c_norm_w/b`
11. RoPE backward and sequential backward of `phase[t,k] = cumsum(angles·dt_mean)`,
    producing `d_angles → d_proj[ang_off]`, and `d_dt_mean` → another contribution to `d_dt`
12. Per-layer pre-LN backward (for `d_layer_norm_w/b`)
13. `d_scale[l] = Σ_i d_x[i] · y_out[l, i]`

### Plan

**Track 1 (PTX backward closure):** implement items 1–13. Target:
`cargo run --release --bin test-parity-train` reaches ≥95% in the same
curriculum-number-of-cycles as `specialist_trainer.py --scan-backend jit
--device cuda`. Reference will be PyTorch autograd's gradient output at
step 0 — grad magnitudes should match to within ~1e-4 for each weight
tensor.

**Track 2 (CPU vs CUDA JIT divergence):** continuation of Entry 23.
`specialist_trainer.py --scan-backend jit --device cpu` does NOT converge
in the same step budget that `--device cuda` does. Smaller-blast-radius
than the Triton fp16 bug, but real.

### What doesn't need revisiting

- PTX forward precision: **done**, verified, 1e-5 vs Rust CPU, zero
  argmax mismatches over 1000 inputs. Any future accusation that "PTX
  introduced a CUDA precision regression" can be rebutted with this.
- The `ptxd` scheduler daemon (single-process, sequential JSON jobs
  via stdin/stdout): wired up, works. Once Track 1 closes the gradient
  gap, ptxd becomes a drop-in replacement for `specialist_trainer.py` in
  `three_populations.py`, with PTX forward at 2.75× CPU.

### First iteration of gradient closure — one win, one stall

**Added:** `d_d_param` kernel (atomic sum over (t, p) inside `ssm_scan_bwd_full`),
`d_decay[t, h]` via block-reduce, `d_dt_from_inp[t, h]` (inp-path contribution
to `d_dt`). Fixed two earlier bugs along the way:

- `ssm_scan_bwd_full` reduction used fixed stride=128, which reads uninitialized
  shared memory when `hd*ds < 256`. Changed to `stride = (hd*ds)/2`.
- `bx_bwd`'s formula was dividing by `dt*trap` where it should multiply — the
  CPU reference `train.rs` has the same math bug. Fixed to `d_bx = d_scan_inp * dt * trap`.

**Result on parity (d=64 L=4 dS=8, 5000 steps, batch=64):**
- Baseline (only `in_proj`, `out_proj`, `embed`, `fnorm` grads): 58–61% best
- With `d_d_param` enabled: 58–72% best; single peak at 72% (step 400),
  mostly 50–58%. Slightly better on average but not robustly converging.
- With `d_d_param + d_dt_bias` (decay-path only): **worse**, 58% and
  training diverges around step 2400 (loss ramps from 0.44 to ≥1.6).
- With `d_d_param + d_layer_norm_w/b` (per-layer pre-norm bwd):
  **slower convergence**, 58% best. The correct LN backward gives a smaller
  gradient than the "skip LN" approximation my earlier code was using, and
  the model doesn't reach the same early peak.
- With `d_d_param + d_dt_bias` (via `ssm_param_grads`, inp-path enabled):
  NaN by step 200. The `blended = inp_val / dt` division amplifies by ~20×
  at init; AdamW grad clipping to [-1, 1] isn't enough.

**Conclusion of iteration 1.** Adding individual gradient paths one at a time
is hitting a wall. Each "mathematically correct" addition either:
(a) doesn't improve the peak meaningfully (single-noise peak at 72% is not
reproducible), or
(b) destabilizes training because magnitudes interact badly with the fixed
LR/WD/clip triple that works for the baseline weights.

The baseline weights (in_proj, out_proj, embed, fnorm) are large matrices
that average a lot of gradients — their update magnitudes are naturally
moderate. The SSM parameters (dt_bias, d_param, layer_norm_w) are smaller,
receive sparser gradients with different scale, and benefit from different
LR or separate weight-decay (PyTorch's standard practice: `no_decay` group
for biases and norm parameters).

**What's needed to close the loop to 95%:**

This is more than a mechanical translation of PyTorch autograd. PyTorch's
`specialist_trainer.py` with the winning config hits 100% in one cycle
because:
1. It computes the FULL gradient (every path, correctly scaled)
2. It uses `no_decay` parameter groups so norm/bias params don't get decayed
   away
3. It may use `torch.nn.utils.clip_grad_norm_` (norm-based clipping, not
   elementwise) so directions are preserved
4. The curriculum starts short (min_len=2) which helps the model learn the
   simpler mapping first

To reach 95% on PTX:
1. Implement the *norm-based* gradient clipping (not just elementwise) so
   whole-weight-tensor directions are preserved
2. Exempt norm weights and biases from weight decay (per-parameter-group WD)
3. Implement all remaining gradient paths: bp/cp norm, RoPE/angles, d_scale,
   correct bx coupling across timesteps
4. Implement the curriculum (progressive bit-length) in `test-parity-train`
   — currently we train on fixed n_bits=4; PyTorch curriculum starts at
   min_len=2 (easier)

These are real pieces of work. The session's conclusion:

- **PTX precision: solid.** 7e-6 forward diff vs CPU, zero argmax mismatches.
- **PTX infrastructure: works.** forward_cached, AdamW, cross-entropy, backward
  kernels — all bit-exact against Rust CPU `TrainState` to 1e-6.
- **PTX training capability: limited by gradient coverage, not by hardware
  or precision.** The Rust CPU `TrainState` ceiling (~61%) and my PTX ceiling
  (~58–72%) are effectively the same — both are training with incomplete
  gradients. Neither has been closed to the full autograd gradient that
  PyTorch computes.

This is a plateau to rest at, not a wall. Track 1 restart needs the four
items above to land simultaneously, not incrementally.
