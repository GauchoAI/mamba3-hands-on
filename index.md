# Index — what ML lives in this repo, where, and how well it works

This repo started April 20, 2026 as a Mamba-3 implementation study
(Lahoti et al., ICLR 2026) and grew, in 9 days and 496 commits, into an
ecology of small specialists for algorithmic tasks. Below is a snapshot
of what architectures are in here, what they're used for, and the
empirical ceiling each one has hit.

For deeper narrative on any line item, the lab notebook is `findings.md`
(54+ entries from the engine era plus the recent Lego/GRU entries); the
long-form vision is `VISION.md`; per-file architectural notes are in
`ARCHITECTURE.md`, `BOOTSTRAP.md`, and `CURRICULUM.md`.

---

## Project arc — 9 days, 496 commits

| Date | Commits | Focus |
|---|---|---|
| 2026-04-20 | 19 | Mamba-3 minimal SISO block, parity solved, phase probe, modular counting fail logged |
| 2026-04-21 | 104 | AugmentedMamba3 (registers, spike gates, persistent memory); Triton SSM scan; H100 deployment; Three Populations architecture; Firebase telemetry |
| 2026-04-22 | 43 | Stateless orchestrator; diagnostician with typed prescriptions; provenance tracking; plateau-triggered mutations |
| 2026-04-23 | 54 | PTX engine v3 — Layer 1 forward/backward kernels; cooperative multi-block persistent forward |
| 2026-04-24 | 106 | PTX engine Layer 1 complete + L2 MVP; ptxd daemon (JSON stdin/stdout trainer); bit-parity to PyTorch |
| 2026-04-25 | 67 | PTX scheduler (Tetris slot scheduler); streaming BTCH batch protocol; pluggable Loss/Optimizer/Schedule; real-teacher KD; hot-plug daemon |
| 2026-04-26 | 29 | REPAIRS R-1 through R-4 (parity from scratch in 26s); Mac-only split — vast.ai instability pushed to MPS daily driver |
| 2026-04-27 | 33 | Parameter-free LoopCounter (HANOIBIN n=100k byte-perfect); RegisterBank primitive; Hanoi-exec dual primitive; EOS-bias gating; FIB-decimal F(40) |
| 2026-04-28 | 41 | Hanoi step function (perfect extension); Lego library (5 specialists, ~2.2k params); sort suite; 3D Cornell light-CA byte-perfect; Hanoi role-MLP arc → fingerprint diagnosis → order-invariant GRU |

---

## 1. Mamba-3 SSM (sequence model)

The original target of the repo. Pure-PyTorch implementation of the
three innovations from Lahoti et al. (ICLR 2026):

- Trapezoidal (2nd-order) discretization via a data-dependent gate `trap`.
- Complex-valued dynamics via data-dependent RoPE applied to B and C.
- (MIMO omitted; SISO rank-1 version for clarity.)

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

Lives on the `pod-archive` branch, but its scaffolding stayed on `main`
for the day a local cluster makes it useful again. Was the
production-parity training engine before the Mac-only split. Built over
~3 days (April 23–25).

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

The architecture that orchestrates all the small models. Built on April
21 (commit `f518e0d`). Reduces a "specialist zoo" to one general
distillation pipeline.

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

Tiny MLPs that learn one step of a discrete rule (a CA tick, one swap of
bubble sort, one Hanoi move, one GCD reduction). The orchestrator runs
them in a Python loop. This pattern decouples *structural correctness*
(orchestrator owns the loop / state machine) from *parametric fit* (MLP
owns the rule's parameters).

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

Tiny routers that reach into a frozen specialist and harvest its hidden
state. The cheapest way to extend an existing organism with a new
behavior — ~1.1k params per synapse on our scale.

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
