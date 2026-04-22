# Experiment Report: Resource Pool Run — 2026-04-22

## Summary

10+ hours of training on H100 with the subprocess-based resource pool.
15 tasks attempted, 2 genuine masteries, 3 false graduations (inherited metrics bug).
Key finding: parallel workers on shared GPU are **slower per task** than sequential
despite 99% GPU utilization.

---

## Two Genuine Masteries

| Task | Exp | Real Peak | Cycles | Time | Notes |
|------|-----|-----------|--------|------|-------|
| **same_different** | exp_0002 | **96%** | 62 | ~30 min | First teacher. Genuine mastery. |
| **run_length_next** | exp_0006 | **100%** | 63 (1 real) | instant | Inherited same_different weights. Hit 100% on first cycle. Genuine transfer learning success. |

run_length_next is remarkable: it loaded same_different's weights and immediately scored 100%.
This suggests these two tasks share underlying representations.

## Three False Graduations (inherited best_acc bug)

| Task | Exp | Real Peak | Reported Best | What Happened |
|------|-----|-----------|---------------|---------------|
| sequence_completion | exp_0004 | **50%** | 96% | Inherited same_different's best_acc. Never actually hit 96%. |
| pattern_period | exp_0005 | **65%** | 96% | Same bug. Real peak was 65%. |
| mirror_detection | exp_0007 | **70%** | 96% | Same bug. Real peak was 70%. |

These "graduated" because `_handle_finished` read the specialist checkpoint's `accuracy`
field, which was the inherited 96% from the parent task — not their own performance.

## Tasks That Didn't Graduate

### Batch 1 (original 4 workers, cycles 1-500, ~31s/cycle)

| Task | Exp | Real Peak | Final Acc | Cycles | Behavior |
|------|-----|-----------|-----------|--------|----------|
| **parity** | exp_0000 | 67% | 47% | 500 | Peaked at cycle 57, then oscillated 43-60% for 443 cycles. Loss flat at 0.347. |
| **binary_pattern_next** | exp_0001 | 93% | 73% | 500 | Peaked at cycle 21, oscillated 55-92% for 479 cycles. Never hit 95%. |
| **odd_one_out** | exp_0003 | 48% | 9% | 500 | Peaked at cycle 77. Hardest task in batch. Loss still high (1.1). |

### Batch 2 (inherited weights from same_different, cycles 63-562)

| Task | Exp | Real Peak | Final Acc | Cycles | Behavior |
|------|-----|-----------|-----------|--------|----------|
| **sequence_completion** | exp_0004 | 50% | 41% | 501 | Started from same_different weights. Oscillated 22-50% entire run. |
| **pattern_period** | exp_0005 | 65% | 31% | 501 | Best effort 65%. Loss oscillated 0.45-0.67. |
| **mirror_detection** | exp_0007 | 70% | 43% | 501 | Peaked early at cycle 78, then degraded. Loss stuck at 0.347 (same as parity). |

### Batch 3 (inherited weights, currently running, cycles 561-688+)

| Task | Exp | Real Peak | Current Acc | Cycles | Behavior |
|------|-----|-----------|-------------|--------|----------|
| **repeat_count** | exp_0008 | 60% | 37% | 130+ | Oscillating 37-60%. Loss ~0.59. |
| **geometric_next** | exp_0010 | 78% | 62% | 130+ | Best performer in batch. Loss dropping (0.43-0.77). |
| **alternating_next** | exp_0011 | 61% | 61% | 130+ | Steady ~40-61%. Not improving. |
| **arithmetic_next** | exp_0009 | 14% | 10% | 130+ | Hardest task. Loss still >1.2. Barely learning. |

## Comparison: Previous Sequential Run

The old sequential `three_populations.py` (one task at a time, full GPU, ~6s/cycle)
produced these results in approximately 3 rounds (~25 minutes per round):

| Task | Acc | Cycles | Notes |
|------|-----|--------|-------|
| **modus_ponens** | 100% | 4 | Instant mastery |
| **run_length_next** | 100% | 63 | Fast |
| **geometric_next** | 98% | 7 | Near-instant |
| **logic_gate** | 95% | 8 | Near-instant |
| **logic_chain** | 95% | 10 | Near-instant |
| same_different | 96% | 62 | Fast |
| sequence_completion | 96% | 562 | Slow but converged |
| pattern_period | 96% | 562 | Slow but converged |
| mirror_detection | 96% | 562 | Slow but converged |

**5 tasks mastered in ~75 minutes** with the old sequential approach.
The new parallel system produced **2 genuine masteries in ~10 hours**.

## Key Findings

### 1. GPU Utilization ≠ Throughput (for small models)

| Metric | Sequential | Parallel (4 workers) |
|--------|-----------|---------------------|
| GPU % | 25% | 99% |
| Cycle time | ~6s | ~31s |
| Cycles/hour/task | 600 | 116 |
| Total cycles/hour | 600 | 464 |
| Graduations in 1h | ~5 | ~1 |

4 workers sharing an H100 produce LESS total throughput than 1 worker alone.
The 99% utilization is from CUDA context-switching overhead, not productive compute.
For 100K-param models, GPU memory bandwidth is the bottleneck, not compute — adding
more models doesn't help, it creates contention.

### 2. Cross-Task Weight Transfer: Mixed Results

- **run_length_next from same_different: SUCCESS** — 100% on first cycle. These tasks
  share binary comparison primitives.
- **All other transfers: FAILED** — inherited weights produced worse starting accuracy
  than random init in most cases. The tasks are too different for weight transfer
  to help (e.g., same_different → arithmetic_next).

### 3. Training Instability with Shared GPU

Workers oscillate wildly — binary_pattern_next swings between 55% and 93% across
cycles. The 100-example evaluation has high variance, and shared GPU scheduling
adds noise. The old sequential system showed the same oscillation but converged
because it had more cycles/hour (600 vs 116) — more samples from the noisy
distribution.

### 4. The inherited best_acc Bug

When a child loads a parent checkpoint from a DIFFERENT task, it inherits:
- Model weights (potentially useful for transfer learning)
- Optimizer state (meaningless for different task)
- `best_acc` from parent task (WRONG — inflates reported accuracy)
- `cycle` count from parent (WRONG — starts mid-count instead of cycle 1)

This caused false graduations: the pool thought tasks had 96% accuracy when
they actually peaked at 50-70%.

### 5. GA Mutations: Never Executed

All 12 experiments used identical config: `d=64 L=3 lr=0.001 wd=0.1 adamw ce`.
The GA queued thousands of mutated configs into `pending_configs`, but the
admission controller blocked them all because GPU was permanently at 99%.
The smart mutation system (MutationHistory, cost-aware exploration) was never
exercised.

## Recommendations

1. **Go back to sequential training** — one task at a time, full GPU. It's
   strictly faster for 100K-param models on H100.

2. **Fix the inherited best_acc bug** — when loading a checkpoint from a
   different task, reset `best_acc=0` and `cycle=0`.

3. **Use GA for config exploration only when stuck** — run quick 10-cycle
   probes with mutated configs, pick the best, continue sequentially.

4. **Student takes turns on GPU** — after each task graduates, the student
   gets a few distillation cycles before the next task starts.

5. **Keep weight inheritance for related tasks** — run_length_next benefiting
   from same_different is real. Group similar tasks and transfer weights
   within groups.
