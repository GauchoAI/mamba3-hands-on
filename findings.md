# Mamba-3 Hands-On — Findings Log

A running lab notebook as we explore the Mamba-3 paper (Lahoti et al., ICLR 2026)
by implementing a minimal version from scratch and running experiments on Apple
Silicon MPS. Each entry corresponds to a commit.

> **Note (2026-04-26):** The CUDA/PTX engine and ptxd-related entries below were
> the H100/vast.ai era. That work has moved to the `pod-archive` branch. This
> Mac-only branch keeps the historical findings text intact for context but
> the production daily-driver path is PyTorch + MPS (`specialist_trainer.py`).

---

## Entry — JEPA-Cortex: thought-distillation regularizes a small bilingual LM, JEPA-off baseline mode-collapses (2026-04-29)

> **Cross-project takeaway** (full experiment journal in
> [`jepa/findings.md`](jepa/findings.md)):
>
> Pseudo-label distillation onto a 1M-param byte-level Mamba-3 from
> Qwen-2.5-1.5B over a tiny ~80 MB corpus produces fluent bilingual
> text in ~6 hours on 4× RTX 4070 Ti. *Without* a JEPA-style latent
> regularizer the small-corpus byte-CE objective drives the student
> into **mode collapse**: 4 different prompts produce near-identical
> responses; the model falls back on its highest-confidence learned
> attractor (or onto the unary `***:aaaa…` task) for any prompt outside
> its strongest modes. *With* JEPA pressure (λ=0.3 was the qualitative
> sweet-spot), runs preserve prompt-conditioning and improve
> monotonically on held-out bilingual byte CE while pure-byte runs
> oscillate. The result was confounded by an unintentional batch-size
> mismatch (gpu3 at 32, others at 64), so the *trajectory shape* —
> mode-collapse vs prompt-differentiation — is the trustworthy signal,
> not the absolute byte-CE numbers.
>
> The findings live in `jepa/findings.md` with hypothesis-before,
> live-observations, and (eventually) conclusion sections. Reproduce
> via `DEPLOYMENT.md` + `jepa/README.md`.

---

## Entry — Cortex / Counter / LoopCounter arc → moved to `docs/findings/cortex.md`

The 6-entry arc on the Cortex residual-primitive thesis was moved as
part of the structuring pass (2026-04-30). See
[`docs/findings/cortex.md`](docs/findings/cortex.md) for the full arc.

**One-paragraph cross-project takeaway:** The Cortex thesis — small LMs
extend their algorithmic reach via primitives that ride the residual
stream, not via more parameters — has a working existence proof (772-
param counter, 151k Mamba LM, byte-perfect counting to N=500, 16.7×
OOD). The LoopCounter primitive went through three iterations: additive
injection → gated injection → parameter-free `torch.where` on sign of c
(truly unbounded; HANOIBIN n=100k byte-perfect, 5000× extension).
EOS-bias gating broke the bounded-counter ceiling — train n≤20,
generalize to n=230 on Hanoi-binary; FIB(40) byte-perfect via per-
position iter_token. The arc graduated into the JEPA-Cortex experiment
(see jepa/) where the Counter primitive rides a bilingual byte-level
LM trained against a Qwen teacher.

---

## Entry — Renderer LM + the digit-copy failure (2026-04-28)

Closed the third leg of the harness: a tiny Mamba-3 conditional LM that
renders structured tool results as natural-language sentences. Trained
on synthetic prefix-LM pairs:

```
<payload>\x01<sentence>\x02
e.g.  hanoi_solver|n=12|optimal=4095|params=45318|timing=2864 \x01 The optimal solution to Tower of Hanoi with 12 disks requires 4,095 moves. \x02
```

74,400 params (2 layers, d=64), 600 training steps on M4 Pro CPU,
best val loss 0.189. **It doesn't work cleanly** — and the failure is
informative.

**The failure mode that surfaced.** On every test prompt, the renderer
produces fluent template text but **drops or substitutes the specific
digits**:

```
input  : "Solve Tower of Hanoi with 12 disks"
payload: hanoi_solver|n=12|optimal=4095|...
output : "The optimal solution to Tower of Hanoi with 18 disks requires 2,047 moves."
```

The LM has clearly learned the *shape* of the answer ("The optimal
solution to Tower of Hanoi with K disks requires M moves") but it
hasn't learned to *copy* the specific K and M from arbitrary
positions in the byte prefix. It guesses globally-common values.

**Why this is the right family of failure to expect.** A small Mamba
SSM compresses the prefix into a fixed-size hidden state. Specific
digits at arbitrary prefix positions are exactly what SSMs without
attention struggle to copy out — the canonical Mamba selective-copy
result. We've seen this throughout this repo: token-stream Hanoi was
unblocked only when we added EOS-bias gating + parameter-free
LoopCounter; HANOIBIN n=100k worked once we stopped trying to make
the SSM count and let the orchestrator count instead.

**Mitigation: a payload-fidelity guard.** `render_mamba` now checks
that every canonical numeric from the payload appears verbatim in the
LM's output. If anything is missing, fall back to `render_template`
and log what got dropped:

```
> Solve Tower of Hanoi with 12 disks
The optimal solution to Tower of Hanoi with 12 disks requires 4,095 moves.
The Hanoi GRU (45,318 parameters) reproduced this in 2864 ms.
[renderer-guard: LM dropped ['12', '4,095']; used template]
```

The pipeline never publishes a corrupted number. The guard is honest:
the trace shows that the LM failed and the template did the work.

**The thesis-aligned fix (next, not now).** The right shape is
**templates with placeholders**: the LM emits a skeleton like

```
The optimal solution to Tower of Hanoi with $N disks requires $OPTIMAL moves.
```

…and the orchestrator substitutes the actual values from the payload.
That removes the small-SSM copy failure entirely. The LM only has to
produce *language form*, which it's already doing fluently. The
specifics come from the payload via deterministic substitution. This
is the cleanest realization of "language as translation layer" — the
LM owns the shape of the sentence, the orchestrator owns the values,
neither tries to do the other's job.

**State of the harness.** Three Mamba-3-class stages totaling ~165k
parameters plus the 45,318-param GRU specialist:

  - **Router**: 45,459-param Mamba-3 byte classifier, val_acc 100 %.
  - **Specialist**: 45,318-param order-invariant GRU.
  - **Renderer**: 74,400-param Mamba-3 LM, val_loss 0.189, guarded.

Three demoable prompts work end-to-end with auditable per-stage
traces. Files: `train_tool_renderer.py`, `assistant.py` updates.
Checkpoint: `checkpoints/tool_renderer_mamba3.pt` (363 KB). Commit
`afa4dd2`.

---

## Entry — Tool routing inside Mamba-3: regex stub replaced by a 45k-param byte classifier (2026-04-28)

Took the next step on yesterday's harness: the router that picks which
specialist to call is now a Mamba-3 forward pass over hidden state, not
keyword scoring. `train_tool_router.py` generates synthetic prompts
(many phrasings per tool, mixed casing/punctuation/Spanish/English/
decoys), trains a 45,459-param Mamba-3 byte-level classifier (1 block,
d=64) over 96-byte prompts, mean-pools over non-pad positions, and
softmaxes over `{hanoi_solver, gcd, gcdhanoi}`.

Trained on `m4-mini` via `cluster_dispatch.py`, 1500 steps in 1322 s
(~22 min). Best val acc 100.0000 % on a 1024-example held-out synthetic
split, reached at step 200 and stable through step 1500.

`assistant.py` now takes `--router-checkpoint`. With it, the trace line
shows the per-tool softmax probabilities instead of keyword scores:

```
> Solve Tower of Hanoi with 12 disks
  [trace] router(mamba3): probs={hanoi_solver=1.000, gcd=0.000, gcdhanoi=0.000};
          chose=hanoi_solver; args={'n': 12}
The optimal solution to Tower of Hanoi with 12 disks requires 4,095 moves.
The Hanoi GRU (45,318 parameters) reproduced this in 2,849 ms.
```

**The harness is now two Mamba-3-class models in series, both ~45k
params:** the byte-level router decides *which* specialist to call;
the order-invariant GRU specialist does the recursive computation.
The combined system answers natural-language Hanoi prompts at any n
within `n_max_pad`, with a fully auditable trace.

**Out-of-distribution paraphrase check.** Phrasings the router never
saw during training, picked correctly:

  - "I'd like the move count for an 8-disk tower puzzle" → hanoi_solver
  - "common divisor of 144 and 60 please" → gcd

The router generalizes beyond the literal training templates, which is
the property we needed it to have for the Mamba-3-as-router substitution
to be honest. Argument extraction (the `n`, the `(a, b)` pair) stays
as regex on purpose — pulling integers out of text is not the part
that benefits from a learned head.

**What's next on this thread.** The renderer is still templates. The
substitution path is the same shape: replace `render(tool, args, result)`
with a forward pass through the bilingual char-LM, conditioned on the
structured payload. That demonstrates the third leg of the thesis:
language is the *output translator* of the inner computation, not the
medium it happens in.

Files: `train_tool_router.py`, `assistant.py` updates. Checkpoint:
`checkpoints/tool_router_mamba3.pt` (182 KB). Commit `af99190`.

---

## Entry — First-class harness: language as translation layer, demoed end-to-end (2026-04-28)

Wired the thesis up as a working artifact. `assistant.py` is a 220-line
single-file harness over a tool registry:

  natural language → router → specialist → renderer → natural language

The router and renderer are stubs (regex matching + string templates).
The specialists are real:

  - `hanoi_solver` calls the 45,318-param order-invariant GRU we trained
    earlier today; given an arbitrary n it returns the optimal move
    count.
  - `gcd` is `math.gcd` for now (placeholder for the GCD step Lego).
  - `gcdhanoi` is a composite that chains the two; demonstrates that
    the orchestrator can route to *compositions* of specialists, not
    just one at a time.

Three demo prompts, all working end-to-end on the M4 mini via
`cluster_dispatch.py`:

```
> Solve Tower of Hanoi with 10 disks
  [trace] router: scored=[('hanoi_solver', 4)]; chose=hanoi_solver; args={'n': 10}
  [tool ] calling Hanoi GRU via hanoi_solver({'n': 10})
  [spec ] hanoi_invariant_gru_offtrace (45,318 params, order-invariant GRU)  timing=895 ms
The optimal solution to Tower of Hanoi with 10 disks requires 1,023 moves.

> What is gcd of 462 and 252?
  [trace] router: scored=[('gcd', 1)]; chose=gcd; args={'a': 462, 'b': 252}
  [tool ] calling GCD tool via gcd({'a': 462, 'b': 252})
gcd(462, 252) = 42.

> Compute the gcd of Hanoi 6 and Hanoi 9
  [trace] router: scored=[('hanoi_solver', 1), ('gcd', 1), ('gcdhanoi', 1)]; chose=gcdhanoi; args={'a': 6, 'b': 9}
Hanoi(6) needs 63 moves; Hanoi(9) needs 511 moves; gcd of the two = 7.
```

The trace lines are the point: every step is auditable from the
command line — what the router scored, what it chose, which specialist
was called, with what args, and how long the inner computation took.
The model isn't pretending to "think out loud"; it's literally calling
inner specialists and translating their outputs back to language.

**Why this is the right shape, not just a CLI demo.** The substitution
path for both stubs is clean. The router is currently a regex score
over keywords, but its signature is `(text) → (Tool, args)`; replacing
it with a small Mamba-3 head over hidden state changes nothing
elsewhere. The renderer is currently a template, but its signature is
`(tool, args, ToolResult) → text`; replacing it with the bilingual
char-LM (already trained on April 20) changes nothing elsewhere.

**Two protocols that fell out naturally:**

  - `Tool(name, description, keywords, run, specialist_label)` — what a
    specialist looks like to the registry.
  - `ToolResult(ok, payload, timing_ms, specialist)` — what a
    specialist returns. `specialist` is a human-readable label that
    makes the trace honest about *which* underlying model was called
    (e.g. `"hanoi_invariant_gru_offtrace (45,318 params, order-invariant GRU)"`).

**Cluster sanity-check.** The whole flow works under
`cluster_dispatch.py`. One catch surfaced: `cluster_sync.py` excludes
`checkpoints/` (sensible default for the 100s of `.pt`s we have), so
specialist checkpoints have to be `rsync`'d explicitly. The Hanoi GRU
is 182 KB; not a deal. We should probably add an explicit
`--include-checkpoint` flag at some point, or a `specialist_checkpoint`
field on the `Tool` that triggers a per-tool sync.

**Where this leaves us, the actual situation.** The thesis is
demoable. Mamba-3 LM does language; the Lego library (and the GRU)
does reasoning; the harness composes them. Next concrete step is
upgrading the router from regex to a learned classifier head — small
job — followed by the renderer using the bilingual char-LM.

Files: `assistant.py`. Cluster: m4-mini at 192.168.0.170 via
`cluster_sync.py` + `cluster_dispatch.py`. Commit `dc5144b`.

---

## Entry — Hanoi reasoning arc → moved to `docs/findings/hanoi.md`

The 9-entry arc on teaching Mamba-3 to solve Tower of Hanoi was moved
out of the root file as part of the structuring pass (2026-04-30). Same
content, same dates, just relocated. See
[`docs/findings/hanoi.md`](docs/findings/hanoi.md) for the full arc:
the n=40 cliff discovery, decoder-bug ruling, last-digit-attention
diagnosis, the role-MLP plateau and GRU fix, and the eventual
Hanoi-step + Hanoi-exec wins.

**One-paragraph cross-project takeaway:** A 1574-param MLP trained on
Hanoi-step (n=2..6, 1.9 s training) hits 100% AR at n=12 (4095 moves)
under a Python orchestrator. The byte-level `progressive_model.py` LM
that tried to learn the algorithm autoregressively hit a wall at n=40
that traced through three layers of debugging — first thought to be
the output decoder, then a rightmost-byte attention shortcut, finally
diagnosed as the SSM at this scale being unable to extract `n` as an
unbounded counter. The fix was an order-invariant GRU over the disk-peg
sequence (45k params, 100% prediction at n=23 / 8.4M states).

---

## Entry — Lego library arc → moved to `docs/findings/lego.md`

The 4-entry arc on the "Lego" library of step-function specialists was
moved as part of the structuring pass (2026-04-30). See
[`docs/findings/lego.md`](docs/findings/lego.md) for the full arc.

**One-paragraph cross-project takeaway:** Five step-function specialists
(Hanoi+GCD+Conway+Bubble+Maze) total ~2.2k params, ~5 s combined
training. Composite tasks happen via a Python orchestrator with zero
retraining. Speed regime: NumPy beats neural-batched MPS 3-5× on
trivially-vectorizable rules (Conway, WireWorld at 1000²×100). Neural
wins as rule branchiness grows. The Light-CA Lego stretched the pattern
into regression (1009-param structured MLP, 5 materials × 4 dirs × RGB,
Cornell-flat in 70 ms) — first multi-channel state Lego. 3D Cornell
under hard-gating: byte-perfect 32k voxels × 96 steps, max diff 0.0000.

---

## Entry — Trajectory-distillation: stabilizes training, doesn't induce program (2026-04-27)

After multiple architectural attempts (output-history attention,
explicit registers) all NaN'd around stage 5 of the Hanoi curriculum,
we tried the user's "FD-style" idea: train against a programmatic
*oracle trajectory* that tells the model what its register state
should be at every timestep, not just what the final output should
be. Implementation: explicit register bank + auxiliary MSE loss
between actual register-write trajectory and a binary encoding of
2^n−1 from a hardcoded oracle.

**The training-stability finding is real and useful.** The trajectory
loss took the architectural addition from "NaN at stage 5 (n=12)"
to "trains cleanly to stage 9 (n=75) and registers' mix factor
grows 500×." The aux loss provides a much richer gradient signal
than end-to-end CE alone. Worth documenting as a methodology win:
when adding a new architectural pathway to an SSM, supervise its
*intermediate state* against a known-correct trajectory if you have
one.

**The extrapolation claim, however, falls.** We then ran the
decisive test: train on n ∈ [1, 20] only with the full trajectory
oracle, evaluate out to n=100. If the oracle teaches the recurrence,
the model should extrapolate. If it teaches templated outputs, the
model fails past 20.

Result:

```
n=1..20:  100% accuracy   (in trained range)
n=21:     '3'             (random)
n=25:     '255'           ← that's 2^8−1, the n=8 answer
n=29:     '511'           ← n=9's answer
n=30:     '127'           ← n=7's answer
n=40:     '1023'          ← n=10's answer
n=49:     '511'           ← n=9's answer again
```

The model produces **answers FROM the trained range for unseen
inputs**. It's pattern-matching surface features of n and emitting
the closest seen output. The trajectory oracle didn't make it
generalize the recurrence — it just gave it a stronger lookup
table within [1, 20].

**Synthesis.** The user's strict pressure test ("a true program
runs to any input") holds robustly across:
- baseline SSM + curriculum
- baseline SSM + scheduled sampling
- output-history attention
- explicit registers (no oracle)
- explicit registers + trajectory oracle
- explicit registers + trajectory oracle + restricted curriculum

Every variant produces *bounded program-shape* behavior over the
training range and *memorized-template* behavior outside it. The
75k-param Mamba-3 architecture with these training methodologies
cannot induce a program that extrapolates.

What's left to try, in increasing scope:
1. Discrete registers with hard write semantics (push/pop/load/store)
   — what we currently have is soft attention over a continuous bank.
2. Scheduled sampling on the *recurrence steps themselves* — the
   model conditions on its own intermediate values, not just final
   tokens. This is the autoregressive-loss extension to trajectory
   distillation.
3. Universal Transformer / Neural Turing Machine architecture —
   accept that small Mamba-3 has a fundamental ceiling for unbounded
   computation and switch primitive.

The architectural surgery for (3) is real research; (1) and (2) are
shippable in days. But the cleanest finding here is the negative
one: training methodology improvements (oracle, scheduled noise)
help with stability and within-range performance, not with
extrapolation. **Extrapolation requires architectural primitives
that can encode unbounded counters discretely**, and we don't have
those at this scale.

---

## Entry — Cluster transparent partition (cluster_dispatch + cluster_sync) (2026-04-27)

User pushed for "transparent" multi-node operation: jobs should
fan out across the M4 Pro + M4 mini cluster naturally, not
require manual SSH per experiment. cluster_dispatch.py was a
scaffold from earlier; refactored to manifest-driven and added
cluster_sync.py for repo rsync.

**cluster_sync.py**: rsync the repo to every non-local node
(excludes `.venv/`, `runs/`, `checkpoints/`, `three_pop/`,
`__pycache__`, `*.pt`, `.git/` — node-local state stays local).

**cluster_dispatch.py**: takes a JSON manifest of {node, name, cmd}
tuples and runs each via SSH (or local subprocess) in parallel.
Single ad-hoc shorthand: `--node X --name Y --cmd '...'`.
Stdout streams to `/tmp/cluster_dispatch_logs/{name}.log` on the
orchestrator side, so the M4 Pro sees mini's training in real
time without a separate log fetch.

**End-to-end verified.** Today's run had FIBD-decimal training
on M4 Pro, parallel HANOIBIN regression on the mini (testing
the new iter_bias=+50 init for behavioral parity with the +15
HANOIBIN_v2 model). Both nodes doing real training simultaneously,
stable, no SSH drops.

The pattern is now ready for any sweep. Future experiments
should default to writing a tiny manifest (one job per node)
rather than running serially here. cluster_nodes.json has
the actual M4 Pro + mini IPs and repo paths; just `cluster_sync`
before launching to ensure code parity.

---

## Entry — Synapse v2 / AttendBridge arc → moved to `docs/findings/synapse.md`

Three entries on the AttendBridge synapse primitive (one-to-one,
one-to-many, multi-specialist composition) moved out of root as part
of the structuring pass (2026-04-30). See
[`docs/findings/synapse.md`](docs/findings/synapse.md) for the full arc.

**One-paragraph cross-project takeaway:** A 1.1k-param adapter
(W_recv + gate) attached to a router gives +30 points at d_model=16
and is near-redundant at d_model=32+. The synapse is the cheap way to
extend small organisms — not a magic bullet for any size. Also
validated multi-specialist composition: at training time both
specialists' gates open with selectivity > 0.99. Foundational for the
"ecology of small models that route among themselves" thesis in
VISION.md.

---

## Entry — Parity / RoPE foundational arc → moved to `docs/findings/parity_rope.md`

The original 7-entry arc (entries 1-7) on the project's foundational
mechanics — minimal SISO block, parity-via-rotation, mechanistic
interpretability of the –π discovery, mod-N modular counting,
selective-copy gating — moved out of root as part of the structuring
pass (2026-04-30). See
[`docs/findings/parity_rope.md`](docs/findings/parity_rope.md) for
the full arc.

**One-paragraph cross-project takeaway:** The minimal SSM block can
solve parity by rotating exactly –π in a single phase channel — a
mechanistically interpretable result that became the seed for every
"unbounded computation by phase" trick the project later relied on
(LoopCounter via torch.where on sign of c, EOS-bias gating, the
counter primitive). The early architecture work is a small body of
evidence for what survives at scale: bounded counters fail past their
training range; pure rotations don't.

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

## Entry — GA tournament / multi-task arc → moved to `docs/findings/ga_tournament.md`

Entries 14-24 (10 entries; original numbering skipped 18) on the
multi-task GA tournament era moved as part of the structuring pass
(2026-04-30). Spans the H100/vast.ai period, GPU saturation lessons,
external teacher mutations, the architecture dimension, the Triton
kernel bug, and the 14/15-tasks-mastered milestone. See
[`docs/findings/ga_tournament.md`](docs/findings/ga_tournament.md)
for the full arc.

**One-paragraph cross-project takeaway:** With 50 fresh
experiments evolving in parallel under a GA, mastering 14 of 15
tasks is achievable in ~24 hours of H100 time. Three lessons
generalize: (1) GPU saturation is the binding constraint long
before the population is "done" — saturate the card with diverse
specialists first, only then layer mutation pressure; (2) teacher
mutation outperforms hand-tuned teacher schedules; (3) some tasks
(`bool_expr_depth3` was the last) are genuinely hard for a 130M
SSM and benefit from architecture mutations not just hyperparameter
ones. The PTX engine arc (entries 27-54, see engine/ptx/findings.md)
followed directly from the Triton kernel bug discovered here.

---

## Entry — PTX engine arc (entries 27–54) → moved to `engine/ptx/findings.md`

The PTX engine work (2026-04-24 → 2026-04-26) was a continuous
~28-entry arc that grew larger than any other subproject in this
notebook. To keep the root findings.md readable, those entries were
moved verbatim into [`engine/ptx/findings.md`](engine/ptx/findings.md)
during the structuring pass (2026-04-29). Same content, same dates,
same numbering — just relocated.

**One-paragraph cross-project takeaway:** A hand-written PTX Mamba-3
engine for CUDA, owning forward + backward + scheduler, was built
from scratch over ~3 days. It achieved bit-parity with CPU on forward
and 1e-6 parity on backward, converged on parity training, ran ~14×
faster than the same stream on a PyTorch baseline, picked up a
streaming batch protocol that made it task-agnostic over the existing
30+ problems, learned to resume from PyTorch .pt checkpoints, gained
real (non-stub) Lion/Focal/LabelSmooth kernels for the GA mutation
surface, and shipped a hot-plug `ptxd` daemon that the orchestrator
could submit jobs to without restart. The H100/vast.ai pod era then
ended (instability), the engine moved to the `pod-archive` branch,
and the daily-driver path moved back to PyTorch + MPS via
`specialist_trainer.py`. The PTX engine remains usable when a CUDA
box is provisioned.
