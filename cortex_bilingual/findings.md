# `cortex_bilingual/` findings

Folder-local research notes. Standing references first, then dated
entries. The cross-project arc lives in
[`docs/findings/cortex.md`](../docs/findings/cortex.md); this file is
specific to the bilingual-LM line of work and the MI experiments
that ride on it.

---

## Standing reference — claim hierarchy

These are the discrete claims the cortex / primitive-attached-LM
research could make. Weakest to strongest. Future entries should
state which claim they're updating.

### 1 — Plug-in architecture is viable

> A frozen LM + a small adapter that reads/writes the residual stream
> at specific layers can be trained end-to-end against task
> supervision.

Engineering claim. **Status: verified** (existence proof on a synthetic
counting-only LM, 2026-04-28; reproduced against a language-trained LM
2026-04-29 and a wider-N variant 2026-04-30).

Falsifier: adapter training never converges, or LM destabilizes when
adapter is attached.

### 2 — Coupled composition

> If the host LM's training corpus contains the primitive's domain
> (e.g. a 5% unary mixin), a primitive attached after the LM is frozen
> can drive task-specific behavior, including past the LM's training
> distribution.

**Status: verified.** Yesterday's N≤30 LM gave a ceiling around N=30;
today's N≤60 LM extends to N=500 (off-by-4 calibration, not a
counting failure).

Falsifier: adapter trains but produces no OOD reach beyond the LM's
trained range.

### 3 — OOD reach is LM-bounded, not primitive-bounded

> For a fixed primitive, the OOD ceiling is set by the host LM's
> training-distribution width, not by the primitive's parameter count
> or architecture.

**Status: supported, not locked in.** 2× widening of unary support
produced ~17× shift in OOD ceiling — one data point past the line.

Falsifier: a 4× widening (N≤120) does not further shift the ceiling,
OR a bigger primitive on a narrower LM beats a smaller primitive on
a wider one.

### 4 — Cold composition ("the strong claim")

> A primitive composes onto a host LM that has *never* seen the
> primitive's domain in training. Drop a counter onto a pure-language
> LM, train only the adapter, get counting behavior.

**Status: untested in this folder.** Probed in another experiment
folder with mixed results.

This is the claim that delivers the "Lego library" vision — pretrained
LM + plug-in primitive collection, no LM retraining per task family.
Without claim 4, every new task-family requires a new LM trained on
that family's domain, which is roughly what fine-tuning already
provides; a different recipe, not a different paradigm.

Falsifier: adapter trains to convergence on aux-loss but the gate
output ignores the LM's hidden state — composition fails to emerge.

### 5 — Library composition

> Multiple primitives can be attached to the same frozen LM
> simultaneously without retraining the LM and without interfering
> through the shared residual stream.

**Status: untested.** Plausible (different primitives gate on
disjoint byte patterns) but unproven; residual-stream interference
or gradient-time collisions could break it.

### 6 — Host-architecture-universal

> The same primitive composes onto any sufficiently capable host LM
> (Mamba, Phi, Llama, …) with the same recipe.

**Status: untested.** Plausible if claim 4 holds, but tokenizer
mismatch (BPE vs byte-level) and hidden-state structure (attention
KV cache vs SSM scan) introduce real engineering work.

### What today's evidence narrows

Claims 1, 2 are verified. Claim 3 is one data point past the line.
Today's result is **consistent with** claim 4 but does not test it
— the host LM had the unary mixin baked into its training corpus,
so the primitive was reading a pre-conditioned hidden state.

Claim 4 is the scientifically interesting one. Claims 5 and 6 are
speculative until 4 is settled.

---

## Iteration speed strategy

Wall-clock from kicking off the wider-N training to getting a demo
result on 2026-04-30 was **6h 22min**. Of that, 5h 50min was the
LM training. Most of the MI questions queued in
[`docs/findings/cortex.md`](../docs/findings/cortex.md)
do not need retraining — they need *forward passes through the
existing checkpoint plus a tiny adapter or probe*.

**Tiered iteration ladder**, fastest to slowest:

| Tier | Action | Cost |
|---|---|---|
| **A** | Probe / adapter fine-tune on existing checkpoint (e.g. bias-only fix on `read_proj.bias` to close +4; per-layer linear probes; SVD of layer-0 residuals at unary positions) | **seconds to minutes** |
| **B** | Retrain just the counter primitive on existing LM with a different recipe | **~30 min** (counter-attach is 1k steps, 1k params) |
| **C** | Retrain a *mini* LM (2L d=64, seq_len=64, 2k steps) for fast hypothesis testing on architectural questions | **~30 min** (proposed; not yet committed) |
| **D** | Retrain the full LM (4L d=128, seq_len=128, 10k steps) | **~6 h** |

Default to tier A. Climb only when the probe shows the question
genuinely requires new weights. The wider-N retrain is in tier D —
worth doing when something earns it, not by default.

---

## Entries

### 2026-04-30 — wider-N MI probe (N≤60 LM, ~17× OOD shift)

Full entry in
[`docs/findings/cortex.md`](../docs/findings/cortex.md#entry--wider-n-bilingual-lm-17x-ood-shift-on-the-counter-primitive-2026-04-30).
Updates claim 2 (verified, stronger). Updates claim 3 (supported,
one data point). Does **not** test claim 4.

Three small follow-ups queued, all tier A:
1. Bias-only fine-tune of `read_proj.bias` to close the +4 offset.
2. Per-layer linear probe — which layer carries the unary-mode feature.
3. SVD on layer-0 residual at unary positions — does the counter
   ride a single subspace.

Plus one tier-D probe to nail down claim 3:
4. Train an N≤120 LM (4× widening). Linear-vs-multiplicative
   falsifies between two clean hypotheses.
