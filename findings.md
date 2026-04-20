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

## Open threads

- **Length generalization.** Trained on L=16, holds at L=32 (90%), degrades
  beyond. The small residual decay (`A ≤ -1e-4`) erodes parity over long
  sequences. `A_floor = 0` or a gated `A` might fix this.
- **Ablation.** Is the parity win from the RoPE, the trapezoidal gate, or
  both? Worth stripping each one out individually.
- **MIMO.** Implementing rank-r lift would be the natural next step for
  harder state tracking.
- **Associative recall.** Another benchmark from the paper, closer to
  language-modeling value than parity.
