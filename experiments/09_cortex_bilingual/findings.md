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

### 2026-04-30 — tier-A MI probes (per-layer, count-regression)

Two fast probes against the wider-N checkpoint, no retraining:

- **`probe_layers.py`**: per-layer linear classifier on the binary
  feature "is this byte mid-unary-run". All 4 layers reach 98%+
  test accuracy in-distribution (recall_unary ~92%) and the feature
  *generalizes perfectly* to N up to 200 (recall_unary ~97%
  OOD). The binary signal is **layer-redundant** and **OOD-robust**.
- **`probe_count.py`**: per-layer linear regressor on the integer
  count. L3 wins (in-dist MAE 4.6) but **the count saturates at
  ~58 past training distribution**: at true N=150, the residual
  still predicts ~58 regardless. Bucketed bias on L3:
  `(60,100] → -21.7`, `(100,150] → -46.3`. The count signal is
  sharply training-distribution-bounded.

**Mechanism for today's counter +4:** the gate fires on the right
bytes (binary signal works to N=500) but the count signal it would
need to emit the right run-length isn't in the residual past
N≈60. The counter falls back to a learned average run-length from
N≤60 training, hence the constant +4 across N=10..500.

---

## Closure — 2026-04-30

This experiment line is closed.

**State at closure.**
- Best LM: 472,960-param byte-level Mamba-3, 4L d=128, trained 10k
  steps in MLX bf16, ce 0.987 bpc 1.424 — undertrained for natural
  language. Bilingual probes degenerate.
- Best primitive: `CounterPrimitive` (1,028 trainable params),
  attached to the wider-N LM. Counter fires correctly to N=500
  (~17× past N≤30 baseline) with a uniform +4 calibration offset.
- Tooling: tier-A probe scripts (`probe_layers.py`, `probe_count.py`)
  + a working pipeline (Kappa streams, auto-pack, transparent
  reader, session archiver, schema registry) that is genuinely
  reusable for any future experiment in this repo.

**What was actually validated** (claims map):
- Claim 1 (plug-in architecture viable): ✅
- Claim 2 (coupled composition, learned primitive): ✅
- Claim 3 (OOD reach is LM-bounded): one data point past the line
- Claim 4 (cold composition): not tested here
- Claims 5-6: not tested here

**Why closure.** The decision criterion was the cost-to-result
curve and the existence of a dominating alternative:

1. **Cost.** The smallest meaningful experiment is a 6 h MLX
   training run, the resulting LM is a poor language model
   regardless, and each follow-up that requires retraining is
   another 6 h. Tier-A probes are minutes, but they only diagnose;
   they don't extend capability.
2. **Result.** The strongest claim today's evidence supports is
   "training-distribution width × C → OOD reach." That's
   memorization-shelf-widening, not new computation in the forward
   pass. The primitive is reading what the LM already memorized.
3. **The original cortex thesis.** The bet was: a primitive with
   *its own computation* (not learned via gradient descent against
   a coupled host) injects results into a frozen pretrained LM's
   forward pass and adds *new capability*. Today's setup tests a
   weaker, easier version of that — and even there, results are
   modest.
4. **Dominating alternative.** Standard LoRA fine-tuning on a
   pretrained model (e.g. Phi-3.5-mini) for tool calling is
   mature, well-tooled, fast to iterate, and produces a usable
   system this week. The forward-pass-tool-invocation idea would
   need to clear that bar to justify continued effort, and right
   now it cannot.

**What would re-open it.** A clean falsifier-passing positive on
claim 4 — a primitive with its own computation cold-attached to a
frozen LM that has never seen the primitive's domain, producing
genuine OOD capability the LM lacks alone. That experiment is
running in another folder; if it lands, the work here resumes
under a different setup (likely Phi or a pure-language Mamba host,
hardcoded computational state in the primitive, ~minutes of
training). Until then, the bilingual-LM-from-scratch + learned
primitive direction is closed.

**Reusable artifacts.**
- `cortex_bilingual/probe_layers.py`, `cortex_bilingual/probe_count.py`
  — generic per-layer MI probes; trivially adaptable to any model
  with a residual stream.
- The Kappa pipeline (`experiment_pusher.py`, `kappa_packer.py`,
  `cloud_archive.py`, `stream_reader.py`, `kappa_schemas.py`,
  `session_archiver.py`) — used by every future experiment.

**Outcome.** First experiment in this repo closed by decision
rather than by follow-up. The ambitious thesis remains
interesting; this particular path to it does not justify
further compute given the alternative.
