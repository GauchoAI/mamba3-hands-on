# `MAP.md` — Top-level repository catalog

The repository has grown to ~145 Python files at the root, plus several
subfolder experiments (`engine/ptx`, `three_pop`, `jepa/`, `rlf_cortex/`,
`registry/`, `mcp/`, `cli/`, `server/`, `generators/`).

This file catalogs the top-level `.py` files by category so future-you
can tell at a glance what's *active*, what's *historical research that
shipped its result*, and what's *one-off scratch*. New experiments
should go into a dedicated subfolder following the immutability
principle (see `feedback_immutability_principle.md` in the auto-memory)
— not at the top level.

## Active core (do not delete)

The daily-driver pipeline. Most live experiments depend on these.

| File | Role |
|---|---|
| `mamba3_minimal.py` | Canonical Mamba-3 block + config. The reference implementation. |
| `mamba3_lm.py` | Byte-level language-model wrapper around Mamba-3 |
| `cortex_counting.py` | The original Cortex experiment — `CortexLM` + `Primitive` registry + `CounterPrimitive`. Existing experiments load this directly. |
| `progressive_model.py` | Progressive curriculum scaffolding |
| `specialist_trainer.py` | The MPS-era specialist trainer that GA mutations call into |
| `three_populations.py` | The GA orchestrator. Workers → Teachers → Students. |
| `coordinator.py` | Inter-process coordination for `three_populations` |
| `state_db.py` | SQLite state machine for `three_populations` |
| `metrics_db.py` | Metrics writer used by the trainer + dashboard |
| `seed_db.py` | DB seeding for fresh runs |
| `audit_db.py` | DB integrity audits |

## SSM scan kernels (low-level)

The CUDA / Triton / native variants of the selective-scan kernel. All
imported by `mamba3_minimal.py` based on device.

| File | Role |
|---|---|
| `ssm_triton.py` | Triton kernel entry + native fallback dispatch |
| `ssm_triton_kernel.py` | Triton kernel definitions (CUDA only) |
| `ssm_scan_native.py` | torch.jit.script fallback for MPS/CPU |
| `ssm_cuda_kernel.py` | Pre-Triton CUDA path (kept for reference) |

## Telemetry + dashboard + Firebase

| File | Role |
|---|---|
| `dashboard.py` | Static-HTML dashboard generator from tuner state |
| `firebase_push.py` | The original GA event push API (`/mamba3/*` paths) — *do not break this; current UI depends on it*. New cross-experiment schema is in `docs/EXPERIMENT_FIREBASE_SCHEMA.md`. |
| `firebase_sync.py` | Firebase sync helper |
| `metrics_db.py` | Local metrics DB (parallel to Firebase) |

## Cluster + remote execution

| File | Role |
|---|---|
| `cluster_dispatch.py` | Manifest dispatcher across multiple machines |
| `cluster_sync.py` | rsync-based artifact sync |
| `worker.py` | Worker process |
| `worker_tinygrad.py` | Worker variant on tinygrad |
| `mac_sweep.py` | Apple Silicon sweep helper |

## Diagnostics + status

| File | Role |
|---|---|
| `diagnostician.py` | Top-level diagnostic runner |
| `status.py` | Current run status reporter |
| `analyze.py` | Generic analysis harness |
| `auto_tuner.py` | Hyperparameter sweeper |

## Data corpus generation

| File | Role |
|---|---|
| `make_bilingual_corpus.py` | Tatoeba en-es bilingual corpus (Path A baseline) |
| `make_opensubtitles_corpus.py` | OpenSubtitles bigger corpus |
| `make_teacher_corpus.py` | MLX-era Path A pseudo-label distillation |
| `external_teacher.py` | External teacher API plumbing |

## Distillation / training pipelines

| File | Role |
|---|---|
| `distill.py` | General distillation harness |
| `distill_from_router.py` | Router → student distillation |
| `train_progressive.py` | Progressive curriculum trainer |
| `train_bootstrap.py` | Bootstrap Level-0 trainer |
| `train_router.py` | Router training |
| `train_step_function.py` | Generic step-function trainer |

## Step-function specialists (Lego library)

Each file is a self-contained byte-perfect specialist in the Lego library.
They share the same shape: a small MLP that learns the step function for
one task, byte-perfect at training-distribution N, used as a primitive in
composite tasks.

| File | Task |
|---|---|
| `bubble_step_function.py` + `train_bubble_step.py` + `discover_bubble_step.py` | Bubble sort |
| `conway_step_function.py` + `train_conway_step.py` + `discover_conway_step.py` | Conway's Game of Life |
| `gcd_step_function.py` + `train_gcd_step.py` + `discover_gcd_step.py` | GCD |
| `hanoi_step_function.py` + `train_hanoi_step.py` + `discover_hanoi_step.py` | Tower of Hanoi |
| `hanoi_step_validate.py` | Validation for the Hanoi step specialist |
| `light_step_function.py` + `train_light_step.py` | Light propagation |
| `light_sh_step_function.py` + `train_light_sh_step.py` | Light + spherical harmonics |
| `maze_step_function.py` + `train_maze_step.py` | Maze step |
| `wireworld_step_function.py` + `train_wireworld_step.py` | Wireworld CA |
| `cornell_3d.py` + `cornell_3d_sh.py` + `cornell_lightca.py` + `cornell_pathtrace_showdown.py` | Cornell box renderer specialists |
| `conway_speed_showdown.py` + `wireworld_speed_showdown.py` | Speed comparisons |

## Hanoi reasoning arc

A long sub-arc covering the Tower of Hanoi reasoning research. Many
files because the cliff was repeatedly diagnosed and re-diagnosed.

| File | Role |
|---|---|
| `discover_hanoi_aggregates.py` + `discover_hanoi_aggregates_plain.py` | Aggregate-feature classifiers |
| `discover_hanoi_holdout.py` | Holdout-N accuracy probe |
| `discover_hanoi_invariant.py` | Order-invariant GRU (the eventual fix at n=23) |
| `discover_hanoi_mamba.py` | Pure Mamba probe |
| `discover_hanoi_offtrace.py` | Off-trace generalization probe |
| `discover_hanoi_rl.py` + `discover_hanoi_rl_hybrid.py` | RL-augmented Hanoi |
| `discover_hanoi_roles.py` + `discover_hanoi_roles_ensemble.py` + `discover_hanoi_roles_mixed.py` | Role-encoded MLPs |
| `discover_hanoi_setattn.py` | Set-attention attempt |
| `discover_hanoi_sinusoidal.py` | Sinusoidal positional encoding probe |
| `hanoi_solve.py` + `hanoi_solve_gru.py` | Direct AR solvers |
| `hanoi_oracle.py` + `hanoi_exec_oracle.py` | Oracle injection variants |
| `hanoi_exec_validate.py` + `hanoi_step_validate.py` | Validators for AR + step variants |
| `hanoi_parallel_solve.py` | Multi-disk parallel solver |
| `hanoi_tool.py` + `hanoi_tool_validate.py` | Hanoi-as-tool variant |
| `hanoi_trace.py` | Trace generator |
| `hanoi_composite_demo.py` | End-to-end demo |
| `train_hanoi_exec.py` + `train_hanoi_tool.py` | Trainers for the Hanoi variants |
| `analyze_hanoi_errors.py` | Error analyzer |
| `length_gen_hanoi.py` + `length_gen_hanoi_binary.py` | Length-generalization probes |
| `finetune_hanoi_to_gcd.py` | Cross-task transfer probe (Hanoi → GCD) |

## Cortex / Counter / EOS-bias / LoopCounter arc

| File | Role |
|---|---|
| `cortex_counting.py` (also under "Active core") | The 151k-LM Cortex existence proof |
| `compose_logic_gate.py` | Logic-gate composition |
| `register_inspector.py` | Register-bank inspection |
| `selective_copy.py` | Selective-copy gating experiment |
| `parity_experiment.py` | The original parity-via-RoPE experiment |
| `crack_mod3.py` + `crack_mod3_v2.py` | Mod-3 cracking |
| `modular_counting.py` | Modular counting |
| `step_decode.py` | Step-by-step decoder |
| `analyze_repeat_count.py` + `analyze_role_errors.py` | Counter / role analyzers |
| `timeline_repeat_count.py` | Repeat-count timeline plotter |
| `formal_language.py` | Formal-language probe |
| `fib_validate.py` + `fib_decimal_validate.py` | Fibonacci-decimal byte-perfect validators |

## Validators / probes / one-shot scripts

| File | Role |
|---|---|
| `all_validate.py` | Cross-suite validator |
| `ablation_parity.py` | Parity ablation harness |
| `bench_scan.py` | SSM-scan benchmarker |
| `check_eos.py` + `check_flicker.py` + `check_gpu_data.py` + `check_realtime.py` + `check_task_history.py` + `check_winning_configs.py` | One-shot diagnostic probes |
| `debug_mamba3.py` | Single-step debugger |
| `find_the_one_error.py` | Single-error isolator |
| `fish_validate.py` | "Fish" task validator |
| `grokking.py` | Grokking-curve experiment |
| `length_gen_general.py` + `length_generalization.py` | Length-generalization sweeps |
| `multi_trace.py` | Multi-task trace plotter |
| `probe_invariance.py` + `probe_novel_fingerprints.py` + `probe_phase.py` | Probes |
| `round_trip_test.py` | Round-trip serialization test |
| `sort_suite.py` | Sort-task suite |
| `test_register_sizes.py` | Register-size test |
| `_test_optimal_move.py` | Hanoi-step optimal-move smoke (underscore prefix = private) |

## Synapse / orchestration / harness

| File | Role |
|---|---|
| `synapse.py` | The Synapse v2 / AttendBridge implementation |
| `orchestrator.py` | Cross-task orchestrator |
| `dual_task.py` | Two-task interleaving |
| `exp_augmented_survey.py` + `exp_augmented_vs_plain.py` + `exp_multitask.py` + `exp_universal_step.py` | Experiment runners |
| `mamba3_augmented.py` | Mamba-3 + augmentation |
| `play.py` + `play_server.py` + `serve.py` + `render.py` | Playable demo + serving |
| `assistant.py` | Top-level assistant entry |
| `amplify.py` | Some kind of amplification probe |
| `strategies.py` | Strategy library |

## Three-populations support

| File | Role |
|---|---|
| `three_populations.py` (also under "Active core") | The GA orchestrator |
| `hot_plug_test.py` | Hot-plug worker test |
| `train_progressive.py` | Progressive trainer |

## Where new experiments go (don't add to top level)

- Each new self-contained experiment ships in its own subfolder, with
  local copies of any dependencies (per the immutability principle).
- See `engine/ptx/`, `jepa/`, `rlf_cortex/` for examples.
- Cross-experiment shared code stays at the top level (only `mamba3_minimal.py`
  / `ssm_*.py` / `state_db.py` qualify; *don't add others*).

## How to retire something

If a top-level file is superseded by code in a subfolder:
1. Don't delete it (archived state has reference value).
2. Add it to the archive entry below with a one-line note pointing to
   the replacement.

### Archive

*(empty for now — no top-level files have been formally retired)*
