# Index — what ML lives in this repo, where, and how well it works

This repo started as a Mamba-3 implementation study and grew into an
ecology of small specialists for algorithmic tasks. Below is a snapshot
of what architectures are in here, what they're used for, and the
empirical ceiling each one has hit. Recent results are at the top of
each section. Files cited are top-level Python in this directory.

For deeper narrative on any line item, the lab notebook is
`findings.md`; the long-form vision is `VISION.md`; per-file
architectural notes are `ARCHITECTURE.md`, `BOOTSTRAP.md`,
`CURRICULUM.md`.

---

## 1. Mamba-3 SSM (sequence model)

The original target of the repo. Pure-PyTorch implementation of the
three innovations from Lahoti et al. (ICLR 2026): trapezoidal
discretization, complex dynamics via data-dependent RoPE on B/C, and
the data-dependent gate `trap`. Sequential scan; no fused kernels in
the daily-driver path.

**Files:**
- `mamba3_minimal.py` — `Mamba3Block`, `Mamba2LikeBlock`, `Mamba3Config`. The reference SSM. Runs CPU/MPS/CUDA without custom kernels.
- `mamba3_lm.py` — `Mamba3LM` language-model wrapper.
- `mamba3_augmented.py` — `AugmentedMamba3` with `RegisterBank`, `LoopCounter`, `PersistentMemory`, `SpikeGate`. Hardware additions for unbounded computation.
- `progressive_model.py` — `ProgressiveModel` (the main training surface), with `ExplicitRegisters`, `OutputHistoryAttention`, `LoopCounter`, `MultiChannelStateFeedback`, `RegisterBank`.
- `ssm_scan_native.py`, `ssm_triton.py`, `ssm_cuda_kernel.py` — engine variants. The CUDA/PTX path lives on the `pod-archive` branch; daily driver is PyTorch + MPS.

**Where it works:**
- Parity, modular counting, selective copy, mod-3, FIB-unary out to F=6765 (123× length extrapolation), FIB-decimal F(40) 9-digit perfect via per-position iter_token.
- HANOIBIN to n=256 byte-perfect with EOS-bias gating; n=100,000 byte-perfect once `LoopCounter` was made parameter-free (5000× extrapolation).

**Where it doesn't:**
- Tower-of-Hanoi as a token-stream task hits a *bounded-counter* ceiling — at this scale Mamba-3 cannot extract n as an unbounded counter from the input, even with full attention auxiliaries. Output decoder and last-digit attention shortcuts were ruled out as the cause.
- The fix that *did* work (EOS-bias gating, parameter-free LoopCounter via `torch.where(c >= 0)`) is structural — the SSM dominates over additive injections, so new pathways need direct logit override, not addition. Documented in `findings.md` Entries on the EOS gate breakthrough.

---

## 2. Step-function MLPs (the "Lego library")

Tiny MLPs that learn one step of a discrete rule (a CA tick, one
swap of bubble sort, one Hanoi move, one GCD reduction). The
orchestrator runs them in a Python loop. This pattern decouples
*structural correctness* (orchestrator owns the loop / state machine)
from *parametric fit* (MLP owns the rule's parameters).

**Files (specialist + trainer pairs):**
- `conway_step_function.py` + `train_conway_step.py` — Conway's Life.
- `wireworld_step_function.py` + `train_wireworld_step.py` — WireWorld.
- `gcd_step_function.py` + `train_gcd_step.py` — Euclidean GCD step.
- `bubble_step_function.py` + `train_bubble_step.py` — bubble-sort step.
- `maze_step_function.py` + `train_maze_step.py` — wave-front maze.
- `hanoi_step_function.py` + `train_hanoi_step.py` — Hanoi move predictor.
- `light_step_function.py` + `train_light_step.py` — 2D Cornell light-CA.
- `light_sh_step_function.py` + `train_light_sh_step.py` — spherical-harmonic light variant.

**Discoveries (for the same tasks but searching the rule itself):**
- `discover_conway_step.py`, `discover_gcd_step.py`, `discover_bubble_step.py`, `discover_hanoi_step.py`, `discover_hanoi_setattn.py`.

**Where it works:**
- Lego library: 5 step specialists totaling ~2.2k params train in ~5 s combined; composite tasks (Hanoi+GCD chained) work via Python orchestrator with **zero retraining**.
- Hanoi step: 1574-param MLP trained on n=2..6 in 1.9 s, 100% AR rollout at n=12 (4095 moves).
- 3D Cornell box: 406-param Lego, 5.5 s training, byte-perfect (max diff 0.0000) over 32k voxels × 96 propagation steps. The hard-gate-by-category variant fixed the soft-gate compound drift that plagued the 2D version.

**Where it doesn't:**
- Speed: on simple CAs (Conway, WireWorld, 1000²×100 steps), NumPy vectorized rules are 3–5× faster than the neural-batched MPS Lego. The Lego wins when the rule has branchiness NumPy can't trivially vectorize (Light-CA, Hanoi).
- Iterated soft gates compound errors over rollouts; hard gates by category in the orchestrator are required for byte-perfect long-horizon rollouts.

---

## 3. Hanoi role-encoded MLPs (the recursive-task family)

A long thread of architectures for predicting the optimal next Hanoi
move from a state, all trained on canonical traces n=2..15 and tested
on held-out n=16..23. The arc went index-based → role-based → ensemble
→ mixed-K → fingerprint-novelty diagnosis → GRU.

**Files:**
- `discover_hanoi_aggregates.py`, `discover_hanoi_aggregates_plain.py` — first MLP, embedding-based features, 99.96 % held-out (87 errors / 196k).
- `discover_hanoi_holdout.py` — variant with no-N feature.
- `discover_hanoi_roles.py` — first role-encoded version (smallest-K + largest-K + parity + cmp_*). 99.81–99.99 %.
- `discover_hanoi_roles_ensemble.py` — 5–7 seeds at K=10, single ensemble vote. 99.9995 % (1 systematic error).
- `discover_hanoi_roles_mixed.py` — mixed K=8, 10, 12 seeds (the published "100 %" mixed-K ensemble). 100 % on n=16, 17 held-out — but see §3.5 for the catch.
- `discover_hanoi_setattn.py` — DeepSets / set-attention attempt.
- `discover_hanoi_sinusoidal.py` — sinusoidal positional encoding attempt (failed, 57 %).
- `discover_hanoi_mamba.py` — Mamba block over disk-peg sequence (slow on MPS; abandoned for the GRU below).
- `discover_hanoi_rl.py`, `discover_hanoi_rl_hybrid.py` — REINFORCE with imitation pretrain, partial; entropy collapse stuck it at n=4.
- `find_the_one_error.py`, `analyze_hanoi_errors.py`, `analyze_role_errors.py` — diagnostics.

**Where it works:**
- 100 % prediction on n=16, 17 held-out (mixed-K ensemble).

**Where it doesn't (the key finding — see `findings.md` Hanoi entry):**
- Prediction accuracy decays with n past the held-out range:
  ```
  n=18: 99.9989 %  (   3 errors)
  n=19: 99.9424 %  ( 302 errors)
  n=20: 99.7671 %  (2442 errors)
  ```
- Diagnosis (`probe_invariance.py`, `probe_novel_fingerprints.py`): every error at n≥18 lands on a role-fingerprint never seen during n=2..15 training. The role *features* are invariant in form, but the MLP's *learned function* is a lookup over the fingerprint set it saw, and that set grows combinatorially with n.
- Off-trace augmentation (`discover_hanoi_offtrace.py`) cannot reach the missing fingerprints — at low n, the disk-overlap constraints make many high-n fingerprints impossible to construct. Empirical: a single K=12 MLP with 200 k off-trace samples scored **84 %** at n=18 (worse than canonical-only).

---

## 4. Order-invariant GRU (the Hanoi fix)

Replaces the role-MLP with a 45,318-param GRU that processes the
disk-peg sequence largest→smallest. Weights are shared per position,
so the function it learns is defined for any sequence length.

**Files:**
- `discover_hanoi_invariant.py` — `HanoiInvariantGRU` + train + probe. `--offtrace-per-n N` adds N random reachable starts per n, labeled by the recursive `optimal_move_from_state` oracle.
- `hanoi_solve_gru.py` — GRU solver (sequential).
- `hanoi_parallel_solve.py` — batched-lockstep parallel solver. Fast O(n) `optimal_count_from_state` for arbitrary starts.
- `discover_hanoi_offtrace.py` — `optimal_move_from_state` oracle (recursive O(n)) used by both off-trace augmentation and parallel solver.

**Where it works:**
- **Length invariance**: trained on n=2..15 canonical traces only, scores 100 % on canonical traces from n=15 to n=23 (8,388,607 states). 3,000 training steps, ~100 s on CPU.
- **Length + start invariance** (with `--offtrace-per-n 15000`): 50/50 OPTIMAL on random off-canonical reachable starts at n=5..18 in 90 s wall-clock; 28/30 OPTIMAL at n=18..22 random starts (zero failures, 2 budget-cap timeouts at the largest n). Total: 78/80 OPTIMAL across the random-start probes, **zero non-optimal solutions**.

**Where it doesn't:**
- Off-trace augmentation slightly regresses canonical n=23 prediction (100 % → 99.63 %). Two checkpoints serve different needs: `checkpoints/hanoi_invariant_gru.pt` (canonical-only, pure length) vs `hanoi_invariant_gru_offtrace.pt` (length + start).
- We haven't pushed past n=23 (limited by `n_max_pad=24`); raising the pad and retraining is mechanical.

---

## 5. Synapse / AttendBridge (the ecology primitive)

Tiny routers that reach into a frozen specialist and harvest its hidden
state. The cheapest way to extend an existing organism with a new
behavior — ~1.1k params per synapse on our scale.

**Files:**
- `synapse.py` — `AttendBridge`, `Bridge`, `RouterModel`, `PlaceholderSpecialist`.
- `dual_task.py`, `compose_logic_gate.py` — composition demos.
- `train_router.py` — router-training driver.

**Where it works:**
- At router d=16, the synapse gives +30 points; this is the regime where the cheap extension is genuinely cheap.

**Where it doesn't:**
- At router d=32+, the router alone solves the task and the synapse is near-redundant. The right interpretation: **synapses are the cheap way to extend small organisms, not a magic bullet for any size.**

---

## 6. Bootstrap / progressive (curriculum-driven full models)

The big training surface. Combines a Mamba-3 backbone with the
augmentations (`ExplicitRegisters`, `LoopCounter`, etc.) and a
curriculum-stage advancement system.

**Files:**
- `train_bootstrap.py` — `BootstrapModel`. Six-level curriculum (parity → counting → bubble → bubble-sort → multi-task → Hanoi token stream). See `BOOTSTRAP.md`.
- `progressive_model.py` — `ProgressiveModel` with all augmentations.
- `train_progressive.py` — main training driver.
- `specialist_trainer.py` — daily-driver MPS specialist trainer (~57 KB, the production path).
- `three_populations.py` — three-population GA contention scheduler.

**Where it works:**
- The 30+ task families it covers train end-to-end on MPS via `specialist_trainer.py`. The streaming batch protocol (BTCH format) means new tasks plug in without engine changes.

**Where it doesn't (or used to):**
- vast.ai instability pushed the cluster to local M4 Mac minis (see `MAC_LAYOUT.md`). The CUDA/PTX engine path is archived on `pod-archive` for now.

---

## 7. Tool-use over neural memory

For state-machine tasks where the state is large and discrete, a Python
*state-tool* + a feedback embedding into the model beats a learned
register bank.

**Files:**
- `hanoi_tool.py`, `train_hanoi_tool.py`, `hanoi_tool_validate.py` — Hanoi-as-tool: Python owns the peg state, model proposes the move, optimal traces train it.
- `hanoi_oracle.py`, `hanoi_exec_oracle.py`, `hanoi_exec_validate.py` — register-bank version (the earlier approach) for comparison.

**Where it works:**
- Hanoi via tool: 0 params per slot, runtime-variable slot count. Out-perfoms the 27k-param register-bank approach on the same task family.

**Where it doesn't:**
- Tool-use needs the state to be cleanly externalizable. Tasks where the *internal* state of the SSM is the load-bearing thing (parity, modular counting, language) don't gain from the tool pattern.

---

## 8. Validators / probes / orchestration

Not architectures, but the harness that lets us declare success
honestly:

- `all_validate.py`, `fib_validate.py`, `fib_decimal_validate.py`, `fish_validate.py`, `hanoi_step_validate.py`, `hanoi_tool_validate.py` — per-task byte-perfect rollout checks.
- `length_gen_general.py`, `length_gen_hanoi.py`, `length_gen_hanoi_binary.py` — extrapolation probes.
- `probe_invariance.py`, `probe_novel_fingerprints.py`, `probe_phase.py` — diagnostic probes used in the GRU work above.
- `cluster_dispatch.py`, `cluster_sync.py` — fan-out to M4 Pro + minis; logs centralised.
- `orchestrator.py`, `coordinator.py`, `worker.py` — distributed training plumbing.
- `state_db.py`, `metrics_db.py`, `firebase_sync.py`, `firebase_push.py` — connective tissue between nodes.

---

## What's the score?

| Track | Architecture | Best result | Status |
|---|---|---|---|
| Token-stream Hanoi | Mamba-3 + EOS gate + LoopCounter | n=100,000 byte-perfect (5000× extrapolation) | ✓ shipped |
| FIB-unary | Mamba-3 + EOS gate | F=6765 (123×) | ✓ shipped |
| FIB-decimal | Mamba-3 + per-position iter_token | F(40) 9-digit perfect | ✓ shipped |
| HANOIBIN | Mamba-3 + EOS gate | n=256 byte-perfect | ✓ shipped |
| 3D Cornell light | 406-param Lego MLP, hard-gated | byte-perfect 32k voxels × 96 steps | ✓ shipped |
| Hanoi step (predictor) | Mixed-K role-MLP ensemble | 100 % at n=16,17; decays past | partial — superseded by GRU |
| Hanoi step (predictor) | 45k-param order-invariant GRU | 100 % at n=15..23 canonical | ✓ shipped |
| Hanoi solver (any state) | GRU + off-trace aug | 78/80 OPTIMAL random starts n=5..22 | ✓ shipped |
| Lego speed (Conway, WireWorld) | step-function MLP | 3–5× slower than NumPy | known regime |
| Synapse extension | `AttendBridge` | +30 pts at router d=16 | ✓ regime-bounded |
| RL on Hanoi | REINFORCE+imitation hybrid | stuck at n=4 (entropy collapse) | abandoned |

The throughline: **Mamba-3 for sequence tasks where the state is part
of the model, step-function MLPs for tasks where the state is part of
the orchestrator, and a small GRU when the task is recursive and we
need true length invariance.** The composition layer (synapses,
orchestrators, tool-use) is what lets these primitives be reused
across tasks without retraining.
