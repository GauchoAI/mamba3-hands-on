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

## Open threads

- **Von Neumann training data.** Design and implement code generators for
  micro-operation execution traces. Start with comparison-based sorting
  (not counting sort) and stack operations.
- **Symbolic math traces.** Use SymPy to generate algebraic manipulation
  traces. Teach DISTRIBUTE, FACTOR, SUBSTITUTE as learnable operations.
- **Scaling test.** Re-run sorting with vocab=100+ to distinguish genuine
  comparison sorting from counting sort / lookup table memorization.
- **The compilation problem.** How does Brain 1 learn to emit programs
  that Brain 2 executes? This is the hardest open question.
- **Length generalization.** Still unsolved from Entry 3.
- **MIMO.** Still not implemented.
