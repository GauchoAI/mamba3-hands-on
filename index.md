---
title: Repository Index
chapter: Prologue
status: living
summary: "Live book context: {{lab.tasksTracked}} tasks tracked, {{lab.mastered}} mastered, {{lab.models}} model records, {{lab.streams}} archive streams, and {{lab.labRuns}} LabRun records are available from the current Firebase surface."
---

# Index — what ML lives in this repo, where, and how well it works

This repo started April 20, 2026 — two days after the Mamba-3 paper
landed — as a "can we replicate this on a Mac mini?" study, and grew,
in 9 days and 496 commits, into an ecology of small specialists for
algorithmic tasks plus the substrate to host them. Below is the bet
that drives the work, the 9-day arc as it actually unfolded, and a
section-by-section snapshot of the architectures with their empirical
ceilings.

For deeper narrative on any line item, the lab notebook is `findings.md`
(54+ entries from the engine era plus the recent Lego/GRU entries); the
long-form vision is `VISION.md`; per-file architectural notes are in
`ARCHITECTURE.md`, `BOOTSTRAP.md`, and `CURRICULUM.md`.

---

## Why we're doing this — the bet

Transformer-class models have **quadratic memory in context length**.
The community spends large amounts of energy on context engineering —
at 1M tokens, even the best frontier setups need expensive compaction
strategies (e.g., calling a frontier model like Claude Haiku to track
dependency graphs across a long session). That cost gets worse as
conversations grow.

When the **Mamba-3 paper** (Lahoti et al., ICLR 2026) landed in
mid-April 2026 promising linear memory and improved precision via
data-dependent RoPE on B and C, this project started two days later.
The bet was that a linear-memory recurrent SSM, paired with the right
composition layer, could replace the quadratic context window with
something cheaper and more honest — a model whose state evolves with
the work it's doing rather than ballooning with the inputs it's seen.

Underneath the architectural bet are four convictions about what
intelligence in software ought to look like:

- **Overparameterized models memorize. Constraining size forces the
  pattern to surface.** A model with enough capacity can always look up
  the answer rather than learn the rule. We deliberately keep models
  small so the gradient has nowhere to hide. Empirically, the minimal
  model is the one that extends — n=12, n=20, n=100,000 — because it
  has actually captured the rule, not pattern-matched to a training
  shape. We've now proved this many times in this repo (HANOIBIN
  n=100k, FIB-decimal F(40), the 45k-param Hanoi GRU at n=23).

- **Language is the translation layer, not the reasoning substrate.**
  We treat natural language as a thin layer between humans and machines
  (and between machines), not as the medium reasoning happens in. The
  reasoning is an inner computation; the language is the API. The next
  step here — already in progress — is integrating reasoning
  specialists with the Mamba-3 LM as forward-pass tools, so a sentence
  is the output of an inner logical computation rather than the
  computation itself.

- **Composable, not monolithic.** Instead of one large model that
  pretends to be intelligent because it has memorized all the patterns,
  we want a constellation of minimal experts — some on reasoning, some
  on language — collaborating at runtime via a harness. The current
  name for this pattern in industry is **"super-learner."** The clearest
  recent public expression is **David Silver's new company** (the
  AlphaGo creator), which raised $1B on exactly this thesis the week we
  wrote this. The architecture is *orthogonal* to the LLM paradigm —
  rather than training on a static human-generated corpus, a
  super-learner interacts continuously with structured environments
  (engineering simulations, formal systems, scientific sandboxes),
  generates its own hypotheses, tests them, receives feedback signals
  from the environment, and updates its beliefs accordingly. Our work
  is in the same family: we don't try to memorize the world; we train
  small specialists against environments where the answer is *checkable*
  (a Hanoi state has an optimal next move, a Conway tick has a
  deterministic successor, a sort step is byte-comparable). Validators
  in this repo are byte-perfect oracles, not human-language judges. We
  also share the LeCun-style world-model intuition: ground the system in
  compact, composable predictors and let language emerge from them.

- **The harness is a first-class citizen.** Tool-calling shouldn't be
  bolted on around a model. It should be an inner primitive — the
  model finds and *creates* new intelligence at runtime according to
  the use case. Spawning a new specialist is itself a tool use.
  Producing a sentence is a tool use (the translation tool). We're not
  ideologically against transformers or diffusion models — we use the
  minimum tool that solves the task at the highest precision and
  smallest size, because small + precise is what converges to the
  optimal representation in the first place.

The point of this repo, in one line: **build the smallest possible
substrate that can host reasoning and language separately, then let the
harness compose them at runtime.**

---

## The 9-day arc — what we actually did

| Date | Commits | What happened |
|---|---|---|
| 2026-04-20 | 19 | **Replicate the Mamba-3 parity result from the paper.** It worked. Trained in seconds on an M4 Mac mini — the speed was the first surprise. |
| 2026-04-20 (cont.) | | **Bilingual char-level LM (EN+ES) from raw bytes.** Converged to recognizable Spanish + English in ~20 minutes on the same Mac mini, with no tokenizer. The second surprise; this is when we knew Mamba-3 was worth committing to. |
| 2026-04-20–04-21 | 104 | **Curriculum with increasing-difficulty logical gates** + AugmentedMamba3 — our extended Mamba-3 with side primitives the SSM doesn't have natively: a *register bank* (small discrete read/write memory), *spike gates* (sparse activation control), *persistent memory* (state that survives across sequences), and a *loop counter* (an explicit "how many times have I iterated" channel for unbounded computation). Reasonable success on simple logic; this is where `ProgressiveModel` + `ExplicitRegisters` + `LoopCounter` got built. |
| 2026-04-22 | 43 | **"Three Populations" architecture** — three pools of models that train each other in a loop: **Workers** propose new candidate specialists (random GA-mutated configs), **Teachers** are the high-accuracy survivors of past rounds, and **Students** are new models distilled from the Teacher pool. Each population pushes results to Firebase; lineage is tracked end-to-end. Same day shipped a **stateless orchestrator** (the orchestrator reads the database rather than holding state, so it can crash and resume cleanly) and a **diagnostician** that watches training telemetry and emits *typed prescriptions* — concrete config mutations targeted at the specific failure mode it detects (e.g. plateau → bump LR by ×1.5; gradient noise → enable label smoothing). |
| 2026-04-23–04-25 | 227 | **H100 disappointment → PTX engine.** Pushed to vast.ai H100s expecting a speedup; got training hiccups instead. Diagnosed as precision issues in upstream Mamba-3 code (a community-reported problem at the time). The fix that worked: write our own GPU kernels in **PTX** — NVIDIA's GPU intermediate assembly, the same low-level layer DeepSeek used to outperform PyTorch on their training runs. End result was ~60,000 tokens/sec on H100 (forward + backward) — but the M4 Mac mini was already doing 16,000. Three Mac minis ≈ one H100 in throughput, with no rental costs and no vast.ai instability. |
| 2026-04-25 | 67 | **PTX engine production-grade**: a **Tetris-style slot scheduler** (packs concurrent training jobs onto one GPU's CUDA streams the way Tetris pieces fit together, instead of serializing them); a **BTCH** streaming batch protocol (a tiny binary format we defined so any task in Python can pipe its training data into the Rust trainer over stdin without bespoke glue); pluggable Loss / Optimizer / Schedule enums; real-teacher knowledge distillation; **hot-plug daemon mode** (jobs added without restarting the trainer); optimizer-state round-trip. The "four-gate methodology" — *parity, perplexity, training-curve match, end-to-end task PASS* — shipped through Entries 39–54 in `findings.md`. |
| 2026-04-26 | 29 | **The Mac-only split.** vast.ai instability tipped us toward betting on a local cluster: three Macs now, more coming. Multi-machine collaboration via Firebase was already in place. PTX engine archived to a long-running git branch called `pod-archive` for the day a local cluster of Mac minis re-justifies it. |
| 2026-04-27 | 33 | **EOS-bias gating + parameter-free `LoopCounter`** unblock the *bounded-counter ceiling* (a hard limit we'd hit where Mamba-3 couldn't extract n as an unbounded counter from a token stream). The two pieces: nudge the model's end-of-sequence-token logit upward exactly when the loop counter says it's done; and make the counter parameter-free by computing it as `torch.where(c >= 0, ...)` instead of learning an addition. With those: **HANOIBIN** (Tower of Hanoi rendered as binary token strings) to n=100,000 byte-perfect — 5000× the longest training length. **FIB-decimal F(40)** 9-digit perfect, **FIB-unary F=6765** (123× extrapolation). The "tool-use over neural memory" pattern also emerged (a Python tool owns the state, the model proposes the move) — the cleaner alternative to teaching the SSM to remember discrete state internally. |
| 2026-04-28 | 41 | **Shallow MLPs converge in seconds for the right tasks.** Discretize the pattern into well-known steps, train a 1–3-layer MLP, run it inside a Python orchestrator. Works for Hanoi, GCD, sorting, Conway's Game of Life, WireWorld, light propagation. We started calling these models **"Legos"** — tiny snap-together specialists that compose for new tasks without retraining, the way Lego bricks compose for new shapes. The **Lego library** is the set of them: 5 specialists totaling ~2.2k parameters, ~5 s combined training, composing at runtime via a Python orchestrator. Same day, the **order-invariant GRU** closes the Hanoi story: 100 % on canonical traces n=15..23 (8.4M states) trained only on n=2..15. The 45k-param GRU + off-canonical augmentation solves arbitrary Hanoi starting positions at n=22 optimally. |

---

## 1. Mamba-3 SSM (sequence model)

The original target of the repo. Pure-PyTorch implementation of the
three innovations from Lahoti et al. (ICLR 2026):

- **Trapezoidal (2nd-order) discretization via a data-dependent gate `trap`.** Mamba-2 turns the continuous-time recurrence into discrete steps with a 1st-order Euler-like rule. Mamba-3 uses the 2nd-order trapezoidal rule instead, blended per timestep by `trap` (a sigmoid the model predicts from the input). The blend is data-dependent: the model decides per token how much "current input" vs "previous derivative" to mix. Better numerical accuracy on long sequences without adding parameters.
- **Complex-valued dynamics via data-dependent RoPE on B and C.** B and C are the SSM's input/output projections — **B** writes the current input into the hidden state, **C** reads the hidden state out. In Mamba-2 these are real-valued, so the hidden state can decay, accumulate, or gate, but it cannot *rotate*; anything periodic (parity, mod-counting, phase structure) is hard to learn. Mamba-3 predicts a small set of rotation angles from the input itself at every step and applies them to B and C as Rotary Position Embeddings *before* the scan. The result is complex-valued dynamics: the hidden state evolves through phase rotations the model controls based on what it just saw. This is what made parity at L=16 jump from ~56 % (Mamba-2-like) to 99.9 % (Mamba-3); the phase probe in `findings.md` Entry 3 caught the model learning exactly −π on one component for bit=1 — a textbook signature of "I'm using rotation to encode parity."
- (MIMO is omitted in our reference; we use the SISO rank-1 version for clarity.)

Sequential scan; no fused kernels in the daily-driver path.

**Reference implementation:**

- `mamba3_minimal.py` — `Mamba3Block`, `Mamba2LikeBlock`, `Mamba3Config`. The reference SSM. Runs CPU/MPS/CUDA without custom kernels.
- `mamba3_lm.py` — `Mamba3LM` language-model wrapper.
- `mamba3_augmented.py` — `AugmentedMamba3` with `RegisterBank`, `LoopCounter`, `PersistentMemory`, `SpikeGate`. Hardware additions for unbounded computation (commit `bf58fb7`).
- `progressive_model.py` — `ProgressiveModel` (the main training surface), with `ExplicitRegisters`, `OutputHistoryAttention`, `LoopCounter`, `MultiChannelStateFeedback`, `RegisterBank`.

**Engine variants:**

- `ssm_scan_native.py` — pure-PyTorch sequential scan.
- `ssm_triton.py`, `ssm_triton_kernel.py` — Triton kernel for SSM scan, eliminates the Python for-loop on GPU (commit `c9efd8c`); required by `mamba3_minimal.py` on MPS too.
- `ssm_cuda_kernel.py` — CUDA path; the production CUDA/PTX engine lives on the `pod-archive` branch.

**Where it works:**

- **Parity (L=16)** — Mamba-3 99.9 %, Mamba-2-like ~56 %; paper's central claim reproduced after fixing an operator-precedence bug on A (commit `fdeaa84`). Ablating RoPE and trapezoidal both hurt (commit `f87219e`).
- **Phase probe** — the model learned exactly −π on one component for bit=1; mechanistic interpretability win.
- **Modular counting** — mod 3, 5 with curriculum (commit `ba563d5`). Initially failed in minimal config; honest limitation logged in `findings.md`.
- **Selective copy** — Mamba-3 gates state writes (commit `72c46c5`).
- **Bilingual char-level LM** — Mamba-3 speaks EN+ES (commit `cda723a`).
- **HANOIBIN** to n=256 byte-perfect with EOS-bias gating; **n=100,000 byte-perfect** (5000× extrapolation) once `LoopCounter` was made parameter-free via `torch.where(c >= 0)` instead of learned addition (commits `8acbd46`, `7f29b0e`).
- **FIB-unary** to F=6765 (123× length extrapolation).
- **FIB-decimal F(40)** 9-digit perfect via per-position `iter_token`.

**Where it doesn't (and the fix that unstuck it):**

- Tower-of-Hanoi as a token-stream task hit a *bounded-counter* ceiling. Output decoder and last-digit attention shortcuts were ruled out as the cause.
- The fix that *did* work was structural: EOS-bias gating + parameter-free `LoopCounter`. Documented in `findings.md` Entries 51–54 plus the Hanoi/EOS entries.
- Lesson saved to memory: additive oracle injection is dominated by the SSM. New pathways need direct logit override or gating, not addition.

---

## 2. PTX engine — Rust + CUDA Mamba-3 (archived)

A from-scratch GPU training engine for Mamba-3, written in Rust and
NVIDIA PTX (the low-level GPU assembly language that sits below CUDA
C). Built over ~3 days (April 23–25) when the upstream PyTorch path
hit precision issues on H100 that matched a community-reported bug.
The ptxd "daemon" runs as a long-lived process: Python sends it
training jobs over stdin as JSON, ptxd executes them on the GPU and
streams loss / accuracy / lineage events back. It's archived on the
`pod-archive` git branch (kept dormant but intact) since the
April 26 split to a local Mac cluster, but its scaffolding stayed
on `main` for the day a multi-Mac-mini cluster re-justifies it.

**What's in there:**

- **Layer 1 — training kernels** (commits `8024b08`, `8e9aef8`, `91cb002`):
  - `adamw_step` + `cross_entropy_fwd_bwd`.
  - `forward_cached` + `ssm_scan_cached` + `TrainScratch`.
  - All four backward kernels + `PtxTrainer`.
  - Cooperative multi-block persistent forward (one launch for the whole model).
  - Tiled `matmul_t` with 16×16 SMEM staging.
- **Layer 2 — daemon** (commits `ef21be1`, `2bca93e`):
  - `ptxd` JSON stdin/stdout trainer.
  - Per-instance stream on `PtxModel`.
- **Tetris slot scheduler** (`scheduler.rs`, commits `0fa6806`, `2bca93e`, `3830ed9`, `febcec0`):
  - Concurrent jobs on one GPU via stream multiplexing.
  - Split advance into prepare/finalize for true multi-stream.
  - Removed per-batch loss sync (concurrency unblock).
  - 8–17 % throughput gain via `std::thread::scope` parallel prepare phase.
- **Streaming batch protocol** (BTCH format, commit `7a486fe`):
  - Task-agnostic; all 30+ tasks plug in for free.
- **Pluggable enums** (commit `a10a30a`):
  - `Loss::{Ce, CeKd, Focal, LabelSmooth}`.
  - `Optimizer::{AdamW, Lion}`.
  - `Schedule::{WarmupFlat, WarmRestarts}`.
- **Knowledge distillation** (`Loss::CeKd`, commit `83fb1af`):
  - Math fixed in `a62d85a` (additive Hinton KD with T multiplier).
- **Optimizer state round-trip** (Phase 5, commit `f5a0e8b`).
- **Hot-plug daemon mode** — jobs added without daemon restart (commit `9d11008`).
- **Telemetry tail** (`ptxd_tail.py`) — POST scheduler ticks to Firebase.
- **Regression guard** — never clobber a higher-accuracy `.pt` (commit `76ce353`).
- **In-process auto-tuner** — reacts to diagnostic event stream (commit `ebc60bd`).

**REPAIRS arc** — curriculum-mode parity-from-scratch debugged through
four documented repairs (commits `75f6c7f`, `24efa97`, `18c8dba`,
`94c16ee`). R-3 cracked it: parity from scratch in 35 s.

**Status:** integration-ready (Entries 51–54). Can resume the existing
82 `.pt` checkpoints via `ckpt_bridge`. Demoted from "daily driver" to
"overnight GA" tool when vast.ai instability forced the Mac-only split
on April 26.

---

## 3. Three Populations — Workers → Teachers → Students

A genetic-algorithm-style training loop with three pools of models that
feed into each other:

- **Workers.** Newly-spawned candidate specialists, each running a
  GA-mutated config (different LR, depth, regularization, etc.). They
  train independently on the same task. Most don't converge well; a few
  do.
- **Teachers.** The high-accuracy survivors of past Worker rounds.
  They're frozen and used as targets for knowledge distillation.
- **Students.** New, smaller models distilled from the Teacher pool.
  When a Student matches its Teachers' accuracy at lower cost, it
  graduates and becomes a Teacher itself.

The loop is: Workers → some succeed → become Teachers → distil to
Students → best Students join the Teacher pool. The architecture means
we're never running just one experiment; we're running a population of
experiments and selecting the survivors. Built April 21 (commit
`f518e0d`). Reduces a "specialist zoo" to one general distillation
pipeline.

**Files:**

- `three_populations.py` — three-population GA contention scheduler. One-line opt-in to PTX engine via `MAMBA_ENGINE=ptxd` (commit `752cc48`).
- `worker.py`, `worker_tinygrad.py` — workers spawn specialists and push to Firebase.
- `coordinator.py`, `orchestrator.py` — orchestrate cycles, mutations, distillation rounds.
- `specialist_trainer.py` — daily-driver MPS specialist trainer (~57 KB, the production path).
- `state_db.py`, `metrics_db.py` — local persistence with lineage tracking; SQLite is source of truth, Firebase does CDC.

**Diagnostic system** (April 22, commits `c31a163`, `c65ea87`, `8fa750e`, `d40ff1c`, `3b5e289`):

- `diagnostician.py` — signal detection + typed prescriptions.
- Schema migration: `diagnostic_history`, `error_analysis`, `provenance` tables.
- Plateau-triggered mutations + per-task lineage logging.
- Provenance: every mutation tagged with source.
- Model card with provenance + diagnostic stats.

**Stateless orchestrator** (April 22, commits `c7e15b4`, `0bda24e`, `87256ed`):

- Workers run diagnostician on themselves.
- Orchestrator reads DB, not memory — zero downtime if it dies.

---

## 4. Step-function MLPs (the "Lego library")

We started calling these models **"Legos"** because, like Lego bricks,
each one is a small specialist that snaps into a Python orchestrator and
composes with other Legos for new tasks without retraining. Each Lego
is a tiny MLP (200–1500 parameters, trains in 1–5 seconds) that learns
*one step* of a discrete rule — a Conway's-Life tick, one swap of bubble
sort, one Hanoi move, one GCD reduction. The orchestrator runs the
loop; the MLP only learns the per-step transition. This pattern
decouples **structural correctness** (orchestrator owns the loop and
the state machine) from **parametric fit** (MLP owns the rule's
parameters), which is how the same 1574-param Hanoi step Lego correctly
extrapolates from training on n=2..6 to a perfect 4095-move rollout
at n=12.

**Specialists (each is a 200–1500 param MLP, trains in seconds):**

- `conway_step_function.py` + `train_conway_step.py` — Conway's Life.
- `wireworld_step_function.py` + `train_wireworld_step.py` — WireWorld.
- `gcd_step_function.py` + `train_gcd_step.py` — Euclidean GCD step (331 params, 1.2 s train, commit `77d2ac1`).
- `bubble_step_function.py` + `train_bubble_step.py` — bubble-sort step.
- `maze_step_function.py` + `train_maze_step.py` — wave-front maze.
- `hanoi_step_function.py` + `train_hanoi_step.py` — Hanoi move predictor (1574 params, 1.9 s train, 100 % AR at n=12, commit `9d0d0a0`).
- `light_step_function.py` + `train_light_step.py` — 2D Cornell light-CA.
- `light_sh_step_function.py` + `train_light_sh_step.py` — spherical-harmonic light variant.

**Discovery variants (search the rule itself):**

- `discover_conway_step.py`
- `discover_gcd_step.py`
- `discover_bubble_step.py`
- `discover_hanoi_step.py`
- `discover_hanoi_setattn.py`

**Orchestration & composition:**

- `compose_logic_gate.py` — composition demo over frozen specialists.
- `sort_suite.py` — 4 sorting algorithms on a single bubble-step Lego, n=3000 benchmark (commits `5860a3a`, `b2d3f12`).
- `hanoi_composite_demo.py` — composite tasks (GCDHANOI, CHAIN) using the discovered Hanoi ensemble.

**Where it works:**

- Lego library: 5 step specialists totaling ~2.2k params train in ~5 s combined; composite tasks (Hanoi + GCD chained) work via Python orchestrator with **zero retraining**. Committed and documented `691cb75`, `c89fd14`.
- Hanoi step: 1574-param MLP trained on n=2..6 in 1.9 s, 100 % AR rollout at n=12 (4095 moves).
- 3D Cornell box: 406-param hard-gated Lego, 5.5 s training, byte-perfect (max diff 0.0000) over 32k voxels × 96 propagation steps. The hard-gate-by-category variant fixed the soft-gate compound drift that plagued the 2D version.
- Long render iteration arc — `fca5e97` → `d187a96` → `7adc4e8` → `2ffa9fa` → `ef459f4` → `1c5b18a`, fixing silhouette outlines, wall seams, and SH aliasing.

**Where it doesn't:**

- Speed: on simple CAs (Conway, WireWorld, 1000²×100 steps), NumPy vectorized rules are 3–5× faster than the neural-batched MPS Lego. The Lego wins when the rule has branchiness NumPy can't trivially vectorize (Light-CA, Hanoi).
- Iterated soft gates compound errors over rollouts; hard gates by category in the orchestrator are required for byte-perfect long-horizon rollouts.

---

## 5. Hanoi role-encoded MLPs (the recursive-task family)

A long thread of architectures for predicting the optimal next Hanoi
move from a state, all trained on canonical traces n=2..15 and tested on
held-out n=16..23. The arc went index-based → role-based → ensemble →
mixed-K → fingerprint-novelty diagnosis → GRU.

**The chain of attempts:**

- `discover_hanoi_aggregates.py`, `discover_hanoi_aggregates_plain.py` — first MLP, embedding-based features. 99.96 % held-out (87 errors / 196k). Commit `41a2b0f` exposed the inductive-bias ceiling.
- `discover_hanoi_holdout.py` — variant with no-N feature.
- `discover_hanoi_sinusoidal.py` — sinusoidal positional encoding attempt. Failed at 57 %.
- `discover_hanoi_setattn.py` — DeepSets / set-attention attempt. Training instability (commit `addd168`).
- `discover_hanoi_mamba.py` — Mamba block over disk-peg sequence. Slow on MPS; abandoned for the GRU below.
- `discover_hanoi_roles.py` — first role-encoded version (smallest-K + largest-K + parity + cmp_*). 99.81–99.99 %.
- `discover_hanoi_roles_ensemble.py` — 5–7 seeds at K=10, single ensemble vote. 99.9995 % (1 systematic error, commit `9395310`).
- `discover_hanoi_roles_mixed.py` — mixed K=8, 10, 12 seeds. The published "🎯🎯🎯 TRUE 100.000000 %" mixed-K ensemble (commit `0b3fe64`). 100 % on n=16, 17 held-out.
- `discover_hanoi_rl.py`, `discover_hanoi_rl_hybrid.py` — REINFORCE with imitation pretrain. Stalled at n=4 with entropy collapse (commit `feceb69`).

**Diagnostics:**

- `find_the_one_error.py` — pinpoints the single systematic error in the 99.9995 % ensemble.
- `analyze_hanoi_errors.py`
- `analyze_role_errors.py`

**Where it works:**

- 100 % prediction on n=16, 17 held-out (mixed-K ensemble).

**Where it doesn't (the key finding — see `findings.md` Hanoi entry):**

| n | states | accuracy | errors |
|---|---|---|---|
| 18 | 262,143 | 99.9989 % | 3 |
| 19 | 524,287 | 99.9424 % | 302 |
| 20 | 1,048,575 | 99.7671 % | 2442 |

- Diagnosis (`probe_invariance.py`, `probe_novel_fingerprints.py`): every error at n≥18 lands on a role-fingerprint never seen during n=2..15 training. The role *features* are invariant in form, but the MLP's *learned function* is a lookup over the fingerprint set it saw, and that set grows combinatorially with n.
- Off-trace augmentation (`discover_hanoi_offtrace.py`) cannot reach the missing fingerprints — at low n, the disk-overlap constraints make many high-n fingerprints impossible to construct. Empirical: a single K=12 MLP with 200 k off-trace samples scored **84 %** at n=18 (worse than canonical-only).

---

## 6. Order-invariant GRU (the Hanoi fix)

Replaces the role-MLP with a 45,318-param GRU that processes the
disk-peg sequence largest→smallest. Weights are shared per position, so
the function it learns is defined for any sequence length. Committed as
`9ceefa4` and `dc0a45a`.

**Files:**

- `discover_hanoi_invariant.py` — `HanoiInvariantGRU` + train + probe. `--offtrace-per-n N` adds N random reachable starts per n, labeled by the recursive `optimal_move_from_state` oracle.
- `hanoi_solve_gru.py` — GRU solver (sequential, one forward per step).
- `hanoi_parallel_solve.py` — batched-lockstep parallel solver; up to 50 independent runs share one GRU forward per tick. Recursive O(n) `optimal_count_from_state` for arbitrary starts.
- `discover_hanoi_offtrace.py` — `optimal_move_from_state` oracle (recursive O(n)) used by both off-trace augmentation and parallel solver.
- `_test_optimal_move.py` — sanity check that the oracle agrees with the canonical Hanoi trace at n=2..15.

**Where it works:**

- **Length invariance**: trained on n=2..15 canonical traces only, scores 100 % on canonical traces from n=15 to n=23 (8,388,607 states). 3,000 training steps, ~100 s on CPU. Hits 100 % at n=18 after just 500 training steps.
- **Length + start invariance** (with `--offtrace-per-n 15000`): 50/50 OPTIMAL on random off-canonical reachable starts at n=5..18 in 90 s wall-clock; 28/30 OPTIMAL at n=18..22 random starts (zero failures, 2 budget-cap timeouts at the largest n). Total: 78/80 OPTIMAL across the random-start probes, **zero non-optimal solutions**.

**Where it doesn't:**

- Off-trace augmentation slightly regresses canonical n=23 prediction (100 % → 99.63 %). Two checkpoints serve different needs:
  - `checkpoints/hanoi_invariant_gru.pt` — canonical-only, pure length invariance.
  - `checkpoints/hanoi_invariant_gru_offtrace.pt` — length + start invariance.
- We haven't pushed past n=23 (limited by `n_max_pad=24`); raising the pad and retraining is mechanical.

---

## 7. Synapse / AttendBridge (the ecology primitive)

A **synapse** in this codebase is the small bridge module that lets one
model query the *hidden state* of a frozen specialist — like an
inter-neuron link in biology, but between models. The implementation
(`AttendBridge`) is a tiny attention head: a router model produces a
query, attends to the specialist's hidden state, and reads back a
context vector. The router stays trainable; the specialist stays
frozen. This is the cheapest way to extend an existing organism with a
new behavior — about 1.1k parameters per synapse on our scale, vs.
re-training a whole specialist.

**Files:**

- `synapse.py` — `AttendBridge`, `Bridge`, `RouterModel`, `PlaceholderSpecialist`.
- `dual_task.py` — many-to-one composition demo (limited not by the primitive but by specialist input-distribution sensitivity, per `VISION.md`).
- `compose_logic_gate.py` — composition demo over Legos.
- `train_router.py` — router-training driver.

**Where it works:**

- At router d=16, the synapse gives +30 points; this is the regime where the cheap extension is genuinely cheap.

**Where it doesn't:**

- At router d=32+, the router alone solves the task and the synapse is near-redundant. The right interpretation: **synapses are the cheap way to extend small organisms, not a magic bullet for any size.**

---

## 8. Bootstrap / progressive (curriculum-driven full models)

The big training surface. Combines a Mamba-3 backbone with the
augmentations (`ExplicitRegisters`, `LoopCounter`, etc.) and a
curriculum-stage advancement system.

**Files:**

- `train_bootstrap.py` — `BootstrapModel`. Six-level curriculum (parity → counting → bubble → bubble-sort → multi-task → Hanoi token stream). See `BOOTSTRAP.md`.
- `progressive_model.py` — `ProgressiveModel` with all augmentations.
- `train_progressive.py` — main training driver.
- `specialist_trainer.py` — daily-driver MPS specialist trainer.
- `train_router.py` — router-side training for the synapse pattern.
- `train_hanoi_exec.py`, `train_hanoi_tool.py`, `train_hanoi_step.py` — Hanoi-task-specific training drivers.

**Curriculum** (`CURRICULUM.md`) supports:

- Adaptive teacher (commits `2f6a0d4`, `c4bb71f`) — observes performance, adjusts difficulty.
- Sequential task unlock (commit `65d72a2`) — 15 progressive tasks + 18 boss tasks.
- Cycle-based learning with gap throttle (commit `80d2f93`).
- Immutable checkpoints + carry-forward best_fresh (commit `e46b18e`).

**Where it works:**

- The 30+ task families it covers train end-to-end on MPS via `specialist_trainer.py`. The streaming batch protocol (BTCH format) means new tasks plug in without engine changes.

**Where it doesn't (or used to):**

- vast.ai instability pushed the cluster to local M4 Mac minis (see `MAC_LAYOUT.md`). The CUDA/PTX engine path is archived on `pod-archive` for now.

---

## 9. Tool-use over neural memory

For state-machine tasks where the state is large and discrete, a Python
*state-tool* + a feedback embedding into the model beats a learned
register bank.

**Hanoi-as-tool family:**

- `hanoi_tool.py` — Python tool owns the peg state, model proposes the move (commit `6afbb32`).
- `train_hanoi_tool.py` — trains the model on optimal traces with tool feedback in the loop.
- `hanoi_tool_validate.py` — byte-perfect AR validation.

**Hanoi-as-register-bank family (earlier approach for comparison):**

- `hanoi_oracle.py` — per-step register-execution trace generator.
- `hanoi_exec_oracle.py` — registers loaded with n at the SEP token for parity dispatch (commit `c8cd74c`).
- `train_hanoi_exec.py` — `ProgressiveModel`-integrated trainer (commit `7a3bf8f`).
- `hanoi_exec_validate.py` — AR validation. Small-curriculum model passes n=2..4 byte-perfect (commit `c5ffceb`).

**Where it works:**

- Hanoi via tool: 0 params per slot, runtime-variable slot count. Out-performs the 27k-param register-bank approach on the same task family.
- Hanoi-exec dual primitive (`RegisterBank` + `LoopCounter`): byte-perfect AR for n=2,3,4 (commits `c1ea6cf`, `8b90169`).

**Where it doesn't:**

- Tool-use needs the state to be cleanly externalizable. Tasks where the *internal* state of the SSM is the load-bearing thing (parity, modular counting, language) don't gain from the tool pattern.
- Register-bank approach hit a 27k-param ceiling and didn't extend past n=5 OOD without curriculum surgery — superseded by tool-use.

---

## 10. Validators / probes / orchestration

Not architectures, but the harness that lets us declare success
honestly.

**Per-task byte-perfect rollout checks:**

- `all_validate.py` — runs every task's validator.
- `fib_validate.py`, `fib_decimal_validate.py` — Fibonacci.
- `fish_validate.py` — fish-counting.
- `hanoi_step_validate.py` — Hanoi step-function rollout.
- `hanoi_tool_validate.py` — tool-use Hanoi.
- `hanoi_exec_validate.py` — register-bank Hanoi.

**Extrapolation probes:**

- `length_gen_general.py`
- `length_gen_hanoi.py`
- `length_gen_hanoi_binary.py`
- `length_generalization.py`

**Diagnostic probes (used in the GRU work):**

- `probe_invariance.py` — stream-eval canonical traces in 64k chunks at every n.
- `probe_novel_fingerprints.py` — for each error at n≥18, classify its role-fingerprint as seen-in-training or novel.
- `probe_phase.py`

**Distributed training plumbing:**

- `cluster_dispatch.py` — manifest-driven fan-out to M4 Pro + minis via SSH.
- `cluster_sync.py` — rsync logs centrally.
- `worker.py`, `worker_tinygrad.py`, `coordinator.py`.

**Connective tissue:**

- `state_db.py`, `metrics_db.py` — local persistence with lineage tracking.
- `firebase_sync.py`, `firebase_push.py` — telemetry to Firebase.
- `audit_db.py` — inspect state database.
- `check_realtime.py` — real-time `active_runs` + per-cycle history.

---

## What's the score?

| Track | Architecture | Best result | Status |
|---|---|---|---|
| Token-stream Hanoi | Mamba-3 + EOS gate + LoopCounter | n=100,000 byte-perfect (5000× extrapolation) | ✓ shipped |
| FIB-unary | Mamba-3 + EOS gate | F=6765 (123× extrapolation) | ✓ shipped |
| FIB-decimal | Mamba-3 + per-position iter_token | F(40) 9-digit perfect | ✓ shipped |
| HANOIBIN | Mamba-3 + EOS gate | n=256 byte-perfect | ✓ shipped |
| Parity | Mamba-3 minimal | 99.9 % at L=16 (RoPE + trapezoidal both load-bearing) | ✓ shipped |
| Modular counting | Mamba-3 + curriculum | mod 3, 5 solved | ✓ shipped |
| Selective copy | Mamba-3 | gates state writes correctly | ✓ shipped |
| Bilingual char-LM | Mamba-3 | EN+ES coherent | ✓ shipped |
| 3D Cornell light | 406-param Lego MLP, hard-gated | byte-perfect 32k voxels × 96 steps | ✓ shipped |
| Hanoi step (predictor) | 1574-param Lego MLP | 100 % AR at n=12 (4095 moves) | ✓ shipped |
| Hanoi step (predictor) | Mixed-K role-MLP ensemble | 100 % at n=16,17 held-out; decays past | partial — superseded by GRU |
| Hanoi step (predictor) | 45k-param order-invariant GRU | 100 % at n=15..23 canonical (8.4M states) | ✓ shipped |
| Hanoi solver (any state) | GRU + off-trace aug | 78/80 OPTIMAL random starts n=5..22 | ✓ shipped |
| Hanoi register-execution | RegisterBank + LoopCounter | byte-perfect n=2..4 | partial — superseded by tool-use |
| Tool-use Hanoi | Python tool + feedback embedding | 0 params per slot, runtime-variable | ✓ shipped |
| Sort suite (4 algorithms) | bubble-step Lego + orchestrator | n=3000, all correct | ✓ shipped |
| Lego speed (Conway, WireWorld) | step-function MLP | 3–5× slower than NumPy | known regime |
| Synapse extension | `AttendBridge` | +30 pts at router d=16, near-redundant at d=32+ | ✓ regime-bounded |
| RL on Hanoi | REINFORCE+imitation hybrid | stuck at n=4 (entropy collapse) | abandoned |
| PTX engine forward | Rust + CUDA Mamba-3 | bit-parity to PyTorch reference | archived (pod-archive) |
| PTX engine training | Layer 1 backward + AdamW | parity 61 % → 72 % stable; 2.1× CPU | archived (pod-archive) |
| Slot scheduler | Rust + CUDA stream multiplex | concurrent jobs, 8–17 % throughput gain | archived (pod-archive) |
| Knowledge distillation | Loss::CeKd kernel + Hinton math | real-teacher KD end-to-end | archived (pod-archive) |
| Three Populations | Workers/Teachers/Students GA | full lineage + Firebase telemetry | active on MPS |
| Diagnostician | typed prescriptions + plateau triggers | per-task targeted mutations | active |

The throughline: **Mamba-3 for sequence tasks where the state is part of
the model, step-function MLPs for tasks where the state is part of the
orchestrator, and a small GRU when the task is recursive and we need
true length invariance.** The composition layer (synapses,
orchestrators, tool-use) is what lets these primitives be reused across
tasks without retraining. The PTX engine is the production-grade
training substrate for the model class, kept warm on `pod-archive` for
the day a local Mac mini cluster re-justifies it.

The thesis the work is building toward: **language is a tool the model
uses, not the medium the model thinks in.** Reasoning happens in
small, precise specialists; language is the translation layer that
renders an answer for humans (or for another model). The harness that
composes specialists at runtime — finding existing ones, eventually
spawning new ones — is the first-class abstraction, not a wrapper
around a monolith. We expect to keep publishing tiny models in this
shape and let them constellate.
