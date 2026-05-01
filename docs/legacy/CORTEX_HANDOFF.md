# Cortex experiment — session handoff

**Session id (predecessor):** `94146998-e0b9-4586-8984-1a1eefe3122d`
**Date:** 2026-04-28
**Status at handoff:** Phase 1 done. Counter primitive HURT vs baseline
at first OOD point (cortex FAIL(13) at N=50, baseline OK at N=50).
Phase 2 (auxiliary supervision) is **CODED** in `cortex_counting.py`
but NOT YET TRAINED. Just run `python cortex_counting.py train_phase2`.

---

## The thesis being tested

The user has built byte-perfect "Lego" specialists (Hanoi GRU, GCD, Conway,
LoopCounter, etc.) and a byte-level Mamba-3 LM that learned Spanish from
zero. The question that opened this thread:

> How do I go from a library of small puzzle-experts to a small LM that
> *reasons internally*, in one forward pass — not by emitting tool-call
> tokens that a Python parser dispatches?

Initial proposal (rejected by user): LM as router/dispatcher into specialist
tools. Their objection: "tool use is just Python with extra steps; any LM
does it. I want a small LM that **actually reasons** — counting 1 to 100
internally, not by writing software."

The architectural answer that survived: **specialists are forward-pass
modules in the residual stream, not external tools.** The LM emits gate
signals from its hidden state, the specialist does its byte-perfect
arithmetic on those gates, and the result flows back into the residual
stream as a learned embedding. No tokens, no parser, no loop — reasoning
*is* the forward pass.

The first concrete experiment is counting (the simplest reasoning failure
in current LMs, e.g. the Apple "Illusion of Thinking" paper).

---

## What was built

**File:** [cortex_counting.py](cortex_counting.py)

Two classes:
- `CounterPrimitive` — a stateful integer counter that scans across the
  sequence. Per position, the LM emits `inc_gate` and `reset_gate` from
  its hidden state via two `Linear`s. Recurrence: `c[t] = (1-reset[t]) *
  c[t-1] + inc[t]`. Counter value is read back via sinusoidal embedding
  (log-spaced periods 1..4096), projected to `d_model`, added to residual.
  ~6.6k params at d_model=96.
- `CortexLM` — `Mamba3LM` with the counter injected after the first SSM
  layer. With `use_counter=False` it's identical to the baseline LM —
  same params, same architecture. This makes the comparison clean.

**Task format (unary in/out, isolates counting from digit composition):**
```
N=5  ->  "*****:aaaaa\n"
N=12 ->  "************:aaaaaaaaaaaa\n"
```
Train on N ∈ [1, 30]. Eval at N ∈ {3, 15, 30, 50, 100, 200, 500} — last
three are far OOD by length.

**Key design decisions:**
- Unary in *and* out: a decimal format would conflate counting with
  digit-composition (model never sees "200" during training).
- Counter `read_proj` initialized with small random weights, NOT zero.
  Zeroing kills gradients flowing back to the gates — gates would never
  train. Verified in early smoke test.
- Counter biases initialized to "rarely increment, never reset" so gates
  start near no-op and have to learn when to fire.
- Counter is a sequential Python for-loop scan over positions. Slow but
  correct. Could be parallelized as a Mamba-style scan later if needed.

---

## Phase 1 results (4000 steps each, eval after)

```
    N      baseline        cortex
    3            OK            OK
   15            OK            OK
   30            OK            OK     ← end of training distribution
   50            OK      FAIL(13)     ← OOD: baseline extends, cortex worse than baseline!
  100      FAIL(72)       FAIL(3)
  200       FAIL(3)      FAIL(12)
  500      FAIL(67)    FAIL(None)
```

**The counter primitive HURT performance, not just sat unused.** Cortex
got worse than baseline at the first OOD step (N=50) — baseline
extends 67% past training, cortex collapses to 13.

This is a stronger negative signal than the smoke-test predicted. Read:
the counter isn't passive noise; it's actively destabilizing the LM's
representations. Some hypotheses for why:

- `read_proj` initialized with std=0.02 (non-zero, so gates can train)
  injects small random vectors at every position. In-distribution,
  the LM compensates. OOD, the counter's behavior extrapolates differently
  (sinusoidal wrap on larger counter values) so the compensation
  doesn't transfer.
- Reset gate biased to -5.0 (almost never reset); over 4000 steps the
  small drift means counters accumulate monotonically across the full
  sequence with no semantic meaning. The injected embeddings are
  position-correlated noise.
- The cortex output also has trailing garbage characters (`G`, `�`) on
  short outputs — the residual injection is spilling into the head's
  decoding even when the SSM has emitted its newline.

**This validates Phase 2.** "Make the option available" doesn't yield
emergent counter use; it yields capacity that gets co-opted as noise.
The architectural claim — primitive-in-residual-stream as a reasoning
substrate — needs a forcing function (auxiliary supervision) to test
whether the wiring itself is load-bearing.

Checkpoints in `checkpoints/cortex/{baseline,cortex}.pt`.
Training log at `/tmp/cortex_phase1.log`.

---

## Phase 2 — implemented, ready to run

The Phase 1 result doesn't disprove the thesis — it shows that
**giving the model an option doesn't force the model to use it**, and
worse, the option becomes additive noise that destabilizes OOD. The
cleaner question Phase 2 answers: *if the counter is properly engaged,
does it deliver OOD generalization the baseline lacks?*

**Implementation (already in cortex_counting.py):**

1. `CounterPrimitive.forward` now returns `(injection, inc_logits)`.
   The pre-sigmoid `inc_logits` are exposed for direct supervision.
   ([cortex_counting.py:113](cortex_counting.py:113))
2. `CortexLM.forward(tokens, return_aux=True)` returns
   `(logits, {"inc_logits": ...})`. Default `return_aux=False` keeps
   the existing call sites and `generate_greedy` working unchanged.
   ([cortex_counting.py:213](cortex_counting.py:213))
3. `counter_targets(tokens)` derives per-position binary targets from
   the byte sequence:
   - Counter A target = 1 where byte == `*`, else 0  → counts input N
   - Counter B target = 1 where byte == `a`, else 0  → counts output a's
   ([cortex_counting.py:310](cortex_counting.py:310))
4. `TrainConfig.lambda_aux` (float, default 0.0) — when >0 and the
   model has a counter, the loss becomes
   `main_loss + lambda_aux * BCE_with_logits(inc_logits, targets)`,
   masked by the same per-position mask used for the LM loss.
   ([cortex_counting.py:386](cortex_counting.py:386))
5. `cmd_train_phase2()` — entry point that trains a fresh cortex model
   with `lambda_aux=0.5`, saves to `checkpoints/cortex/cortex_aux.pt`.
   ([cortex_counting.py:492](cortex_counting.py:492))
6. `cmd_eval()` extended — auto-detects `cortex_aux.pt` and adds it to
   the comparison table. ([cortex_counting.py:515](cortex_counting.py:515))

**To run from a fresh session:**
```bash
.venv/bin/python cortex_counting.py train_phase2   # ~12 min on MPS
.venv/bin/python cortex_counting.py eval           # 3-way comparison table
```

The baseline.pt and cortex.pt from Phase 1 are preserved; only
cortex_aux.pt is new. The eval will print baseline vs cortex vs
cortex+aux side by side.

**Smoke-test before the long training run** (recommended):
```bash
.venv/bin/python -c "
import torch, torch.nn.functional as F
from cortex_counting import CortexLM, CortexLMConfig, CountingDataset, counter_targets
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
cfg = CortexLMConfig(use_counter=True, n_layers=2, d_model=64, max_seq_len=80)
m = CortexLM(cfg).to(device)
ds = CountingDataset(1, 10, 80, device=device, seed=0)
x, y, mask = ds.get_batch(4)
logits, aux = m(x, return_aux=True)
target = counter_targets(x)
bce = F.binary_cross_entropy_with_logits(aux['inc_logits'], target, reduction='none').mean(-1)
loss = (bce * mask.float()).sum() / mask.float().sum().clamp_min(1.0)
loss.backward()
g = m.counter.inc_proj.weight.grad.norm().item()
assert g > 1e-6
print(f'OK: aux loss={loss.item():.4f}, inc_proj grad norm={g:.4f}')
"
```

**Decision points after Phase 2 trains:**
- If `cortex+aux` extends OOD where baseline doesn't → architectural
  claim validated. The next question is *emergent* engagement: how to
  get the counter recruited without explicit supervision (curriculum
  that starves the SSM, larger N range, etc.).
- If `cortex+aux` also fails OOD → the wiring or readout is wrong.
  Suspects in priority order: (a) sinusoidal frequencies don't
  extrapolate well past training counter range; try linear/learned
  positional encoding instead; (b) counter injected at wrong layer
  (try after layer 1 or per-layer); (c) reset gate needs aux too.
- If `cortex+aux` matches `cortex` (Phase 1, hurts OOD) → aux loss
  isn't strong enough to outweigh the destabilizing effect; bump
  `lambda_aux` to 1.0 or 2.0.

---

## Commands

```bash
# from /Users/miguel_lemos/Desktop/mamba3/mamba3-hands-on  (NOT the worktree)
.venv/bin/python cortex_counting.py train         # Phase 1: baseline+cortex (already done)
.venv/bin/python cortex_counting.py train_phase2  # Phase 2: cortex+aux (~12 min on MPS)
.venv/bin/python cortex_counting.py eval          # 3-way length-gen comparison
.venv/bin/python cortex_counting.py demo          # generation samples
```

Phase 1 log: `/tmp/cortex_phase1.log` (can be deleted)
Checkpoints: `checkpoints/cortex/{baseline,cortex,cortex_aux}.pt`
(cortex_aux.pt only exists after `train_phase2` runs)

**Path note:** all files live in the **main repo**, not the worktree.
A `cd` early in the predecessor session shifted cwd to main repo,
so all `Path("checkpoints/cortex")` and Write tool calls landed in
`/Users/miguel_lemos/Desktop/mamba3/mamba3-hands-on/`. The worktree
at `.claude/worktrees/eager-volhard-be86cc/` is essentially empty of
this work. Run all commands from main repo.

---

## What NOT to do

- **Don't switch to decimal counting format** without thinking carefully.
  It conflates counting with digit composition and muddies the result.
  Keep unary in/out.
- **Don't make the SSM weaker to "force" counter use.** That's a fake
  win — it doesn't show the architecture works, just that the SSM was
  starved. Auxiliary supervision is the honest path.
- **Don't conclude the thesis is wrong if Phase 1 fails.** Phase 1 only
  tests the *emergent* version. Phase 2 tests the *architectural* claim.
  Both matter, but they answer different questions.
- **Don't add a Phase 2 abstraction that touches the existing
  specialists** (HanoiGRU, etc.) before the counting demo separates.
  Get the counting existence proof first; generalize after.

---

## Repo orientation for the next session

- LM core: [mamba3_lm.py](mamba3_lm.py), [mamba3_minimal.py](mamba3_minimal.py)
- Existing primitives:
  - LoopCounter (counter-value embedder, NOT a counter): [progressive_model.py:211](progressive_model.py)
  - HanoiInvariantGRU (byte-perfect on n=2..23): [discover_hanoi_invariant.py](discover_hanoi_invariant.py),
    checkpoint at `checkpoints/hanoi_invariant_gru.pt`
  - GCD/Bubble/Conway/Maze step functions: `discover_*_step.py` and
    `train_*_step.py`
- Memory files (load on demand):
  - `feedback_tool_use_over_neural_memory.md` — for state machines,
    Python tool > learned register. The cortex thesis is the *opposite*
    direction: differentiable in-forward-pass, no tool call. Worth
    re-reading to keep the distinction sharp.
  - `project_lego_library.md` — the 5-specialist library
  - `feedback_unbounded_not_extrapolation.md` — LoopCounter refactor to
    `torch.where` on sign of c. Same spirit as the new
    `CounterPrimitive` (no max_count anywhere).

---

## Open questions for the next session (in priority order)

1. **Did Phase 1 separate baseline from cortex at OOD N?** If yes (unlikely
   based on smoke test), document it and write up findings. If no, proceed
   to Phase 2.
2. **Phase 2 implementation:** the auxiliary supervision design above.
3. **If Phase 2 works:** how to make engagement emerge without aux
   supervision? Curriculum that gradually starves the SSM? Larger N range
   so SSM capacity is exceeded? This is the actually-hard research
   question.
4. **Second primitive:** wire `HanoiInvariantGRU` into the residual stream
   the same way (frozen specialist this time, not learned-from-scratch).
   Train an LM on natural-language descriptions of small Hanoi instances
   and test internal solving. Validates the pattern beyond counting.
5. **Generalization to language:** the longer arc — train the cortex LM on
   bilingual.txt (the Spanish corpus) interleaved with counting, see if
   counting capability survives mixed-task training without crowding out
   language. The user's actual goal is an LM that has both.

---

## How to resume

```sh
claude --resume 94146998-e0b9-4586-8984-1a1eefe3122d
```

The new global default `permissions.defaultMode = "bypassPermissions"`
is in effect for the resumed session — no more permission prompts.

`~/.claude/settings.json` was edited this session to add that default
along with the existing `skipDangerousModePermissionPrompt: true`.
