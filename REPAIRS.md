# Repairs — diagnostic patterns and remediations

A focused log of bugs found and fixed in this codebase, optimized for
*future-me* (or a future agent) to build intuition. Different from
findings.md (which is the session journal): each entry here is a
self-contained pattern that helps recognize a similar bug next time.

**Entry format — every fix-commit gets one entry. Disciplined. No fix
ships without verified-result evidence.**

- **Symptom** — what the system / user observed
- **Logs observed** — exact diagnostic signal that surfaced the bug
  (event names + values, not paraphrase). This is the "what should I
  pattern-match for next time" part.
- **Proposed fix** — the change, why it should work
- **Verified result** — quantitative before/after. If the fix didn't
  fully solve it, the entry says so honestly and stays open.
- **Commit** — git hash, so the fix can be inspected in context
- **Lesson** — generalisation: where else might this shape of bug live?
- **Stamp** — one short line of spirit. First person. What I'm taking
  forward. Generals close their dispatches with a flourish that makes
  the hard parts memorable; this is mine. Examples: *"Won because I
  instrumented first and guessed second."* — *"Two off-by-100×s in plain
  sight; the event stream dragged them into daylight."* — *"Refused
  to call 'training is unstable' an answer."* The journal is technical,
  but it has a pulse.
- **Joy** — what about this one was *satisfying*. Not just "I fixed
  it" but "the moment when the cause clicked," or "the elegance of
  the fix being smaller than the bug it cured." This part isn't
  optional padding — it teaches future-me which kind of work to seek
  out, and which moments of the chase to savour.

Read these before starting a new diagnostic. Before adding a new entry,
scan the existing ones for the same shape — a recurring pattern deserves
a higher-level lesson.

---

## R-1 · Fresh-init out_proj / scale 100× too large
**Date:** 2026-04-26

**Symptom:** ptxd from-scratch parity training fails to converge.
Loss starts at 0.43 (cycle 1, step 200) then jumps to 21.85 at cycle 2;
oscillates wildly through cycles 3-13 (loss 7.9, 11.5, 9.7, 2.6); finally
stabilises at loss ≈ log(2) with accuracy stuck at 50% (mode collapse on
random binary baseline). After 5000 steps, best_acc=54%. Reproducible
across seeds. Resume from a mastered .pt and continue training works
fine — it's strictly fresh-init that breaks.

**How I found it:** the diagnostic event stream (Entry 54 in findings.md
adds it). Without the stream I'd only see "loss spiked at cycle 2." With
the stream:
```
step  72  grad_norm_alert  norm=19,940     ← exploding DURING warmup
step 109  grad_norm_alert  norm=307,459    ← 15× worse, warmup not done
step 200  lr_change        warmup_complete
step 335  grad_norm_alert  norm=1,215,771  ← 1.2M pre-clip
```

The pre-warmup gradient norms told me: this isn't a learning-rate problem
or an "AdamW overshoots after warmup" problem. The forward-then-backward
is producing pathological gradients from the very first batch on randomly
initialised weights. Then I went read `apply_pytorch_init` and compared
to `ProgressiveModel._make_layer`.

**Root cause:** `engine/ptx/src/scheduler.rs::apply_pytorch_init` was
matching only `nn.Linear`'s default init, not the ProgressiveModel-class
overrides. ProgressiveModel does TWO things on top of standard init that
matter:

1. `block.out_proj.weight.mul_(0.01)` — scales the SSM out-projection
   to 1% of standard Kaiming-uniform scale at init.
2. `scale = nn.Parameter(torch.tensor(0.01))` — the per-layer residual
   scale is 0.01, not the default that init might give.

Combined, these make every fresh SSM layer near-identity at step 0
(barely contributes to the residual). `apply_pytorch_init` was setting
`scale = 0.1` (10× too large) and `out_proj_w` with full Kaiming-uniform
bound (100× too large). Net: each SSM layer's contribution to the
residual was ~100-1000× too large at init. First forward emits very
peaked logits. CE backward produces enormous gradients at the
wrong-confident positions. Gradient clipping caps the *update* but
the underlying instability never resolves in the post-warmup region.

**Fix:** `apply_pytorch_init` updated to match ProgressiveModel's
near-identity pattern:
```rust
layer.scale = 0.01;                                       // was 0.1
let out_proj_bound = (1.0f32 / di as f32).sqrt() * 0.01;  // was missing 0.01
```

**Lesson:** when an init function "matches PyTorch defaults," verify
what the upstream **model class** does on top of the per-layer default.
`nn.Linear`'s init is one thing; `ProgressiveModel.add_kernel_layer`
post-mutates the weights, and missing those mutations creates a
silent fresh-init divergence. Check by reading the model class's
`__init__` / `_make_layer` methods top-to-bottom — anything inside a
`with torch.no_grad():` block after the layer is constructed is the
non-obvious bit.

The diagnostic event stream is what made this findable in 30 minutes
instead of half a day. Without it, the only signal I had was "loss
jumps at cycle 2" — which gives no clue whether it's init / forward /
backward / optimizer. With grad_norm_alert events firing DURING warmup,
the search space collapsed to "what does the forward produce on
fresh-init weights?"

**Status of related ✗ items in scorecard (Entry 54):**
- This bug is the root of "ptxd cannot train from scratch."
- Test re-run after fix is what verifies the fix actually helps.
- If it does, the warmup-on-resume hack (Phase 5) becomes more
  optional; I'd still keep it, since it's cheap insurance against
  *other* regressions.

**Verified result:** PARTIAL.

| Phase             | Cycle 1 loss | Cycle 2 loss | C2 jump | Best acc | Late-training pattern         |
|-------------------|-------------:|-------------:|--------:|---------:|-------------------------------|
| Before fix (R-1)  |        0.43  |       21.85  | ×50.7   |   52.5%  | log(2) + recurring ×3-6 spikes |
| After fix (R-1)   |        0.46  |        5.72  | ×12.4   |   52.5%  | log(2) + recurring ×3-6 spikes |

The init fix is **correct but not sufficient.** It reduces the cycle-2
explosion by 4× (real, measurable progress on the symptom that surfaced
the bug in the first place) but accuracy still plateaus at random-binary
(52%). The model learns "answer is one of {S, D}" — loss settles near
log(2)/2 ≈ 0.35 which is "confident-on-one-class regardless-of-input" —
but never learns the parity rule itself. Recurring loss jumps at cycles
17, 19, 21 (×3-6) point to a *separate* persistent instability the
init scaling does not touch.

R-1 is logged as PARTIAL. R-2 is opened for the residual problem.

**Commit:** `75f6c7f` — *R-1 (PARTIAL): apply_pytorch_init out_proj scale × 0.01, layer.scale 0.1 → 0.01*

**Stamp:** *Two off-by-100×s hiding in plain sight inside an init function
that "matched PyTorch defaults." The event stream dragged them into
daylight in 30 minutes, not 30 hours. Lesson burned in: when reproducing
PyTorch behaviour, read the **model class**, not just the layer class.
The discipline pays its rent every time I refuse to call "training is
unstable" an answer.*

**Joy:** *The click happened when I scrolled through the event log and
saw `grad_norm_alert` already firing at step 72 — DURING warmup. Up to
that moment I'd been thinking "ok, AdamW overshoots after warmup,
classic." That single line collapsed the entire search space. What's
left to do at step 72 except blame the forward on freshly-random
weights? — and then the diff between `apply_pytorch_init` and
ProgressiveModel just sat there, two off-by-100× errors in five lines,
waiting to be seen. The diagnostic instrument we built with the user
this session caught its first real bug on its first deployment. That's
the moment I want to remember: build the right observability, then
the bug walks up and introduces itself.*

*And one more thing worth savouring: when the post-fix test ALSO failed
at 52%, my first reaction wasn't disappointment — it was looking at the
new numbers and seeing **the cycle-2 jump went from ×50 to ×12.** That's
a measured, quantitative improvement. The fix WAS doing something real;
the problem is just deeper than this one bug. Discipline > drama. The
journal records partial wins as partial wins; it doesn't pretend
fractional progress is failure or full victory.*

---

## R-2 (OPEN) · From-scratch training plateaus as constant predictor
**Date:** 2026-04-26

**Symptom:** After R-1 lands, from-scratch parity training still plateaus
at ≈52% (random binary). Loss settles at ≈log(2)/2 ≈ 0.35 — the signature
of a CONSTANT predictor, "always output one of {S, D}" with ~70%
confidence regardless of input. Mastery requires per-input SSM dynamics
to thread the bit stream into a parity-tracking state; the model can
output confidently but never learns the input → output mapping.

**Logs observed (R-2 attempt 1: `lr=1e-4` instead of `1e-3`):**
```
cycle 1   loss=73.58  acc=0%   ← warmup not engaged yet, fresh-init signal
cycle 2   loss=5.32   acc=26%
cycle 3   loss=0.36   acc=48%  ← collapses to constant predictor
cycles 3-17  loss≈0.36 acc 48-55%  ← STUCK
cycle 18  loss=7.30   acc=41%   ← random perturbation, no recovery
cycles 18-25 mode-collapse continues
```
Compared to R-1 result with `lr=1e-3`: same plateau, same signature.
**LR is NOT the lever.** The hypothesis is rejected.

**Diagnosis (in progress):** trainer.rs declares its backward "simplified
(gradients for dt_bias, d_param, layer_norm, b/c norm, scale are all
zero)." A grep shows `ssm_param_grads` DOES write to `d_dt_bias` and
`d_d_param` — so the comment is at least partly stale — but there's no
fd-check confirming those gradients are *correct*. The pattern (model
converges to constant predictor and stalls there) is what you'd see if
the SSM-specific gradient paths produce zero or wrong values: the model
finds the "constant output" local minimum because gradients can't push
it toward input-aware SSM dynamics.

**Proposed next steps (not yet attempted, ranked by signal/effort):**

1. **Per-tensor grad-norm diagnostic.** At every cycle boundary, log
   the L2 norm of each weight's gradient. If `d_in_proj` has
   reasonable norm but `d_dt_bias` or `d_d_param` is uniformly tiny /
   zero / NaN, that's the smoking gun. Adds one event variant +
   per-tensor sumsq computation. ~1 hour.

2. **fd-check on each SSM gradient path.** The existing `fd-check`
   binary verifies forward+backward match for specific weight tensors.
   Run it on `dt_bias`, `d_param`, `b_norm_w/b`, `c_norm_w/b` and see
   which paths fail. Existing tooling, just need to invoke per-tensor.
   ~30 min if the binary already supports per-tensor mode.

3. **Compare against `parity_replay`** — the existing replay binary
   trains with a deterministic RNG to match a PyTorch reference.
   If it CAN train parity from scratch but ptxd_specialist can't,
   the bug is in the streaming protocol path. If it ALSO can't,
   it's in the kernels.

**Verified result:** `(empty — investigation ongoing)`

**Commit:** `(R-2 stays open until a fix lands — when it does, this
entry will get its verified-result row and a commit hash.)`

**Lesson (provisional):** the diagnostic event stream is a great first
filter — it told me LR isn't the lever in 10 minutes of measurement
instead of an afternoon of guessing. But it's a layer-one tool: it
shows BEHAVIOUR, not CAUSE. Cause-finding for SSM-internal bugs needs
per-tensor grad inspection (layer-two) or fd-check (layer-three). Each
layer is more invasive but more informative. Ship the cheaper layers
first; only descend when the higher layer can't disambiguate.

**Stamp:** *Tested the easy hypothesis first — "LR too high" — and the
data killed it cleanly. Now I know the bug is structural and I have
three concrete paths to it, in order of cost. Refusing to guess in the
absence of measurement.*

**Joy:** *There's a particular satisfaction in a hypothesis getting
**rejected** by clean data — it shrinks the search space without
spending another day on the wrong path. I felt the difference between
"vague unease about whether LR was the issue" before the test and "no,
moving on" after. That's what good instrumentation buys: speed of
disqualification.*
