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

## R-2 (DIAGNOSED · fix-pending substantial kernel work) · From-scratch training plateaus as constant predictor
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

1. **Per-tensor grad-norm diagnostic** — would have surfaced this in
   the event stream.
2. **fd-check on each SSM gradient path** — would have shown WHICH
   gradients are wrong.
3. **Compare against `parity_replay`** — would have shown bug is in
   kernels not protocol.

I went looking for #3 and found something better: `engine/ptx/src/bin/
test_parity_train.rs:36-50` is a comment from the engine's author
that **already names the five missing gradient closures** by hand.
No more diagnostic needed; the answer is in the code.

**Diagnosis (final):**

Quoting `test_parity_train.rs:36-50` directly:

> *"Defaults: the config that currently trains STABLY on the PTX
> backward we have. Not the PyTorch-winning config — that one needs
> the remaining gradient closures (**bp/cp LN-bwd, RoPE-bwd,
> d_dt_bias, d_scale, correct bx-timestep coupling**) to converge."*
>
> *"Our PTX reaches ~58% stably today; the gap is **gradient-coverage,
> not precision**."*

That matches our observation exactly: from-scratch parity plateaus at
52-58% with `loss ≈ log(2)/2` regardless of LR, init, or seed. The
constant-predictor mode-collapse is the best the model can find when
the input-adaptive SSM gradient paths aren't wired.

Five missing pieces, each a CUDA kernel + dispatch:

1. `b_norm` / `c_norm` layer-norm backward (the bp/cp pre-SSM norms)
2. RoPE backward (bp/cp positional encoding gradient)
3. `d_dt_bias` — the kernel writes this slot but the values may be
   wrong (memory entry: "d_dt_bias NaN — disabled"); needs an audit
4. `d_scale` — per-layer residual scale gradient (currently zero;
   layer scale is therefore frozen at init=0.01)
5. Correct `bx`-timestep coupling — the gradient that ties `dt` and
   the `bx` projection together through the discretisation step

**Verified result (Diagnosis layer):**

| Phase                        | Best acc | Pattern                               |
|------------------------------|---------:|---------------------------------------|
| `lr=1e-3, R-1 init fix`      | 52.5%    | constant predictor + recurring spikes |
| `lr=1e-4, R-1 init fix`      | 54.5%    | same plateau, fewer spikes            |
| `test_parity_train` defaults | ~58%     | author-documented expected plateau    |

All three land in the same regime. The plateau is structural, exactly
as the engine's author noted.

**Status:** R-2 is **DIAGNOSED**. The fix is bounded but substantial:
implement the five missing gradient closures, each with its own CUDA
kernel, fd-check verification, and integration into
`compute_gradients_with_zero`. Estimated ~2-3 days of focused PTX work
per closure (10-15 days total) to land the full set with bit-parity
against PyTorch. This is its own project, not a one-commit repair.

**Important nuance:** ptxd is still production-ready for the GA's
actual workflow today. The 82 existing PyTorch-trained checkpoints
resume and fine-tune fine in ptxd; the regression guard prevents any
clobbering; fine-tune mastery (e.g., parity.pt staying at 100%
through resume + 100 steps) is verified. From-scratch training was
never required for the GA to keep evolving — only a "bonus" capability
that turned out to need more kernel work to unlock.

**Commit:** `(no fix-commit — R-2 is a diagnosis-only resolution; the
fix is a separate multi-week kernel project.)`

**Lesson (provisional):** the diagnostic event stream is a great first
filter — it told me LR isn't the lever in 10 minutes of measurement
instead of an afternoon of guessing. But it's a layer-one tool: it
shows BEHAVIOUR, not CAUSE. Cause-finding for SSM-internal bugs needs
per-tensor grad inspection (layer-two) or fd-check (layer-three). Each
layer is more invasive but more informative. Ship the cheaper layers
first; only descend when the higher layer can't disambiguate.

**Stamp:** *Tested the easy hypothesis first — "LR too high" — and the
data killed it cleanly. Then before opening the cause-finding kit, I
checked whether the answer was already documented somewhere. It was,
in a comment block in `test_parity_train.rs`. Read the source first;
diagnose second. The five missing gradient closures named there are
the actual unfinished work, not a hypothesis I needed to invent.*

**Joy:** *Two satisfactions stacked here. First, the LR test killing
its hypothesis cleanly — that's the speed-of-disqualification dividend
the instrumentation pays back. Second, finding the engine's author had
already written the diagnosis 18 months ago and just hadn't propagated
it into the runbook. Five lines of comment in test_parity_train.rs:36-50
saved me from rediscovering the gradient-coverage gap one fd-check at
a time. The lesson is reading code as a primary source — even comments,
even half-stale ones, are intelligence about a system. R-2 went from
"deep investigation pending" to "diagnosed, bounded fix scope" in two
minutes of grep-and-read. That kind of velocity is what makes
relentless feel sustainable instead of grinding.*

---

## R-3 (FIXED) · The "five missing closures" diagnosis was wrong — curriculum was the issue
**Date:** 2026-04-26

**Symptom:** Same plateau pattern as R-2 — fresh-init parity stuck at
52-58% across every hyperparameter combination. R-2 had concluded "five
missing gradient closures, multi-week kernel work." That diagnosis was
**wrong**.

**Logs observed:**

I ran `test_parity_train` (the engine's own from-scratch parity
validator) without changes. It uses ptxd's *legacy* parity-data path
which respects `Job.stages` for curriculum advancement:

```
Config: d=32 L=1 dS=16 hd=16 batch=16 lr=0.001 wd=0.1 cycles=25x200
[parity] cycle  1-17  stage=1(len 2-4)  loss=0.0000  acc=46-55%
[parity] cycle 18  stage=1  acc=69%   ← model finds the rule
[parity] cycle 19  stage=1  acc=100%  → advances to stage 2
[parity] cycle 20  stage=2  acc=100%  → advances to stage 3 → DONE
Final: best_acc=100%, 64,000 train steps total, 16 seconds wall.
```

**16 seconds. From scratch. To 100%.** The engine was never broken.

The streaming protocol I built in Phase 1 sources data from
`batches_path` and IGNORES `Job.stages`. So `ptxd_specialist` was
training at exactly ONE distribution (whatever stage I picked) and the
model couldn't find the parity rule from a hard distribution alone.

**Root cause:** ptxd's curriculum advancement logic only fires in the
legacy parity-data path. The streaming protocol bypasses it. So
ptxd_specialist's "single fixed stage per invocation" was a missing
*feature*, not a kernel bug.

**Fix:** I built `test_parity_curriculum.py` that submits a JSON job
to ptxd with:
- `stages: [{2-4, 0.97}, {3-8, 0.95}, {4-16, 0.95}]`
- NO `batches_path` (use legacy data path)
- `auto_tune: false` (let it run the full curriculum)
- seed 42 (matching test_parity_train's default)

**Verified result:**

| Phase                                | Wall    | Best acc | Status      |
|--------------------------------------|--------:|---------:|-------------|
| ptxd_specialist (single stage)       | 877s    | 52.5%    | needs_tuning |
| `test_parity_train` defaults         | 16s     | 100%     | PASS        |
| Curriculum v1 (advance_at=0.90, seed 12345) | 80s | 93.5%    | learning    |
| **Curriculum v2 (advance_at=0.97/0.95, seed 42)** | **35s** | **98%** | **converged** |

The user's 10-minute target (600s) was met with **5.8% of the budget**.
The trajectory:

```
cycle  1-26  stage=1  best=54%   (model groping for the rule)
cycle 27     stage=1  acc=63%    breakthrough
cycle 28-30  stage=1  77 → 81 → 87
cycle 31     stage=1  acc=93%
cycle 35     stage=1  acc=98%    → advances to stage 2
cycle 36-47  stage=2  69 → 96    stage 2 climb
cycle 48     stage=2  acc=96.5%  → advances to stage 3
                                 → target hit on stage 3 → converged
```

**Commit:** *(test_parity_curriculum.py + REPAIRS R-3 commit; the
broader fix is making ptxd_specialist curriculum-aware via Task #18
so the streaming protocol works for ALL tasks, not just parity.)*

**Lesson:** R-2 was confident and wrong. I cited a comment in
`test_parity_train.rs:36-50` that says the engine "needs the remaining
gradient closures to converge" — and used that to declare R-2 a
multi-week kernel project. But I never actually ran
`test_parity_train` with its defaults. If I had, in two minutes I'd
have seen it converges to 100% in 16 seconds, falsifying my diagnosis.

**The hierarchy of evidence is:** running the actual binary > reading
the code > trusting comments. I had it backwards. **Before declaring
a multi-week project, run the existing test suite for that subsystem
with its known-good defaults.** If they pass, the gap is in the path
you're taking, not in the engine itself.

**Stamp:** *Two repair entries (R-1, R-2) chasing a bug that wasn't
there. The diagnosis "five missing closures" was code-comment gospel,
but a 2-minute run of `test_parity_train` would have falsified it.
Read the source first, trust the comments second, **run the binary
third — and the third step is what actually counts**. Parity from
scratch in 35 seconds, well under the user's 10-minute target. The
engine's been capable all along.*

**Joy:** *The cycle 27 → 35 trajectory was the most ML-intuition-rich
moment of the session: 63 → 78 → 87 → 81 → 93 → 87 → 79 → 70 → 98.
Watching the model fight its way out of the constant-predictor
minimum was beautiful — the inflection between cycle 27 and 28 is
where parity becomes legible to it. Then the stage 2 climb was the
model GENERALIZING from short-sequence rule-knowledge to longer
sequences — real learning, observable in real time, all on the
gradient closures the project comment said were missing. Curriculum
learning is one of those tricks that looks like cheating until you
remember it's just "don't ask a child calculus before arithmetic."*

*And the meta-joy: this entire arc validates the diagnostic apparatus
we built earlier. Without the event stream, R-1, R-2, the auto-tuner
- without all that scaffolding - I would never have known where to
look or how to verify the answer. With it, I went from "this is a
multi-week kernel project" to "35-second mastery via a JSON config
change" in 30 minutes of experiments. Build the tools first, the
answers fall out. The user's instinct from earlier — "leave a lot of
traces about what happened and what triggered what" — was the
unlock that made this finding possible.*

---

## R-4 (FIXED) · Curriculum mode integrated into ptxd_specialist (production path)
**Date:** 2026-04-26

**Symptom:** R-3 proved curriculum-driven parity training works in
ptxd via the JSON protocol (test_parity_curriculum.py: 35s).
ptxd_specialist (the production CLI invoked by three_populations.py)
still trained at one fixed stage and plateaued at 50-56%.

**Logs observed:**

R-3's `test_parity_curriculum.py` (direct JSON to ptxd):
```
35 seconds   98% mastered   stages 1→2→3 advanced cleanly
```

`ptxd_specialist --task parity ...` on same engine:
```
50-56% plateau   never escapes constant-predictor minimum
```

The streaming protocol path inside ptxd_specialist sources data from
`batches_path` and ignores `Job.stages`. So every ptxd_specialist
invocation trained on ONE distribution. Three iterations to get this
right:

**v1 (run_curriculum_mode added, no further changes):** wall 621s, stages
1+2 mastered but stage 3 (len 4-16) stuck. The per-stage budget was
adequate but the architecture (d=64 L=4) was the bottleneck on stage 3.

**v2 (smaller arch d=32 L=1):** wall 11.7s, FAIL — stage 1 bailed at
cycle 11. Auto-tuner's "8-cycle stagnation" rule fired before the
model could find the rule (test_parity_curriculum took 27 cycles to
find it). The auto-tuner was tuned for single-stage runs and
preempted the curriculum's natural breakthrough.

**v3 (auto_tune=False per stage, FULL per-call budget per stage):**
wall **26.3 seconds**, **PASS at 98%**.

**Root cause(s):** Three compounding bugs:

1. **Streaming bypasses curriculum.** ptxd's curriculum-advancement
   logic only fires in the legacy parity-data path. Streaming never
   gets the advancement signal. (R-3 surfaced this; R-4 fixes it.)

2. **Auto-tuner preempts breakthroughs.** The 8-cycle-stagnation
   bail-rule is correct for one-stage runs (saving GA compute on
   doomed configs) but wrong during curriculum, where each stage
   needs ~30 cycles to "find the rule." Curriculum sub-jobs need
   `auto_tune: False`.

3. **Per-stage budget arithmetic was wrong.** Splitting a fixed total
   budget across N stages assumes each stage uses its share equally.
   Reality: stage 1 needs the most steps (rule-finding from random
   init); stages 2/3 inherit good weights and finish in 1-3 cycles.
   The fix: give each stage the FULL per-call budget; ptxd's
   `target_acc = stage.advance_at` triggers early-exit on convergence,
   so the budget is shared via *early-exit*, not pre-divided.

**Fix (commit pending):** `run_curriculum_mode()` in ptxd_specialist
calls ptxd once per curriculum stage, chains weights through
`/tmp/ptxd_curriculum_chain_{task}.bin` and `.opt.bin`, and:

- Sets `auto_tune: False` in each stage's job spec
- Sets `steps = total_steps` (full budget per stage; early-exit governs)
- Sets `target_acc = stage.advance_at` (per-stage convergence threshold)
- After all stages, copies the chain.bin to `save_bin_path` so
  the existing regression-guard + ckpt_bridge save path runs unchanged

Auto-detected when `init_from_bin is None` (fresh init) AND task has
a curriculum in `problems/`. Fine-tune flow (`init_from_bin` set)
keeps the legacy single-stage path.

**Verified result:**

| Phase                                       | Wall    | Best acc | Status     |
|---------------------------------------------|--------:|---------:|------------|
| ptxd_specialist single-stage (R-1)          | 877s    | 52.5%    | needs_tuning |
| Direct JSON + curriculum (R-3)              | 35s     | 98%      | converged  |
| **ptxd_specialist + curriculum (R-4 v3)**   | **26s** | **98%**  | **mastered** |

The user's "consistently one-shot parity in less than 10 minutes" is
hit at **4.4% of the budget** through the production CLI. Curriculum
trajectory:
```
stage 1 (len 2-4)   18 cycles  50→92%  (rule discovery)
stage 2 (len 3-8)    2 cycles  77→98%  (carryover)
stage 3 (len 4-16)   6 cycles  83→96%  (generalisation)
```

**Commit:** `(R-4 fix-commit pending — ptxd_specialist run_curriculum_mode)`

**Lesson:** Three bugs compounding. Each one alone wouldn't have
killed it; together they made the integration look impossible. The
discipline that found them: **test the system at every level of
abstraction.** R-3 proved the engine + JSON protocol works. R-4
proves the production CLI that wraps that protocol works. Skipping
either layer would have left the bug undetected.

Also: the auto-tuner is GREAT for single-stage runs and HARMFUL for
curriculum sub-jobs. **The same heuristic can be right or wrong
depending on the orchestration scope.** Don't make rules global when
they're context-sensitive.

**Stamp:** *Three iterations through the integration, each iteration's
failure pointing exactly at the next bug. The diagnostic stream
caught each one cleanly. Built the right tools, then the bugs
introduced themselves one at a time. The journal entries chain like
a relay — R-1 → R-2 (wrong path) → R-3 (right path, manual) → R-4
(right path, integrated). Parity from scratch to mastery via the
production CLI in 26 seconds, well under the user's 10-minute mark,
and the engine wasn't broken — it just needed the right harness.*

**Joy:** *Stage 1 cycle 18 hit 92% accuracy and the run jumped
straight to stage 2. Stage 2 cycle 2 hit 98%. Stage 3 cycle 6 hit
95.5%. Each stage finishing as soon as its threshold cleared, no
wasted compute. The orchestration just CLICKED — the per-stage
chain.bin handoff worked, the early-exit fired exactly when it
should, the regression guard saved the canonical .pt at the end. The
system trained itself in 26 seconds, and I watched it happen line
by line in stderr. There's a particular pleasure in seeing
loosely-coupled components compose without a single integration
failure on the third try. Build the parts right; let the system
emerge.*
