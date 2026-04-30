# Cortex / Counter / LoopCounter arc — findings journal

The Cortex arc: building byte-perfect algorithmic primitives that ride
the LM's residual stream so a small model can do unbounded computation
(parity, counting, Fibonacci, etc.) without growing parameters.

Spans:
- The original Cortex existence proof (772-param counter on a 151k LM,
  byte-perfect to N=500)
- LoopCounter primitive — additive injection vs gated injection
- EOS-bias gating — train n≤20, generalize to n=230 on Hanoi-binary
- FIB-decimal: per-position iter_token, perfect to F(40)
- Parameter-free LoopCounter via torch.where on sign of c — truly
  unbounded
- Counter primitive on a frozen bilingual LM (the cross-experiment
  result that motivated jepa/)

The current ongoing work in `jepa/` and `rlf_cortex/` builds on top of
this arc. Future Cortex work that ships in its own subfolder gets its
own findings.md there; cross-cutting takeaways come back here as they
emerge.

All entries below are verbatim relocations from the root file (2026-04-30).

---

## Entry — Counter primitive on a frozen bilingual LM: partial composition (2026-04-29)

The cortex thesis stress-tested on a real, language-trained host LM
rather than the synthetic counting-only LM that produced the
byte-perfect existence proof on 2026-04-28.

**Setup.** A 472,960-param byte-level Mamba-3 LM (4 layers, d_model=128)
trained for 10,000 steps on the 17.7 MB Tatoeba en-es corpus + 5%
unary cortex mixin. Loss settled at ~1.0 bpc 1.45 bpc-by-the-end.
Then froze every weight, attached a fresh `CounterPrimitive` sized
to d_model=128 (1,028 trainable params), and fine-tuned only those
adapters for 1000 steps on a 50/50 mix of bilingual and unary
batches. λ_aux=0.5 BCE on inc_logits, lr=3e-3. Aux loss converged
crisply (0.354 → 0.0001) — the gates fire correctly on `*` and
`a` bytes through the bilingual LM's hidden state.

Eval at hard_gates_inference=True. Side-by-side vs the same LM
without the counter:

```
N      baseline                cortex+counter
3      FAIL → 7                OK ✓                    ← cortex helps at small N
10     FAIL → 14               FAIL → 9                ← off by 1
30     FAIL → 31               FAIL → 29               ← off by 1, IN-DISTRIBUTION
50     FAIL → 54               FAIL → 48
100    FAIL → None             FAIL → None             ← both drop out of unary
200    FAIL → 65               FAIL → 34               ← cortex worse here
500    FAIL → 33               FAIL → 33
```

Counter-attached samples at training-distribution prompts:
- `*****:` → 4 a's (should be 5) ← off by one
- `**********:` → 9 a's (should be 10) ← off by one
- `***************:` → 16 a's (should be 15) ← off by one in the other direction

**The honest read.**

This is *not* the byte-perfect demo the synthetic-LM cortex experiment
produced. It is also *not* a null result — three concrete things
are validated and three are not.

What the experiment validated:
1. **The plugin interface holds.** A 1,028-param adapter (0.22% of LM
   params) attached to a fully frozen 472k-param language-trained LM,
   trained against the same aux-supervision recipe, learns to fire on
   the right bytes. The pluggable `Primitive` class works as designed
   on a non-toy host.
2. **The bilingual LM is not destabilised.** Bilingual probes
   (`'The cat '`, `'¿Dónde '`, etc.) produce the same family of
   outputs whether the counter is attached or not. The counter
   contributes additively without trampling the LM's language
   ability.
3. **Counter helps at small N.** At N=3 the baseline overshoots to
   7 a's (it pattern-matches "in unary mode emit ~7 things"); the
   cortex version emits exactly 3. The counter's signal *is* getting
   through and is affecting the right decision.

What the experiment *did not* validate:
1. **Byte-perfect counting at training-distribution N.** Even at N=30
   the cortex version is off by one, emitting 29 a's. The synthetic-LM
   experiment was byte-perfect at every N up to 500.
2. **Strong OOD extension.** At N=200 the cortex version produces 34
   a's (not 200); at N=500 it produces 33. The synthetic-LM cortex
   produced 500 of 500 byte-perfect at the same N.
3. **Train-free composition in the strong sense.** The plugin needed
   1000 fine-tune steps; that's "small adapter fine-tune", not
   "frozen plugin attaches and works". A pre-trained CounterPrimitive
   from the synthetic experiment did not transfer (different d_model,
   different hidden-state distribution at unary positions).

**Diagnosis: distribution-shift on the counter readout.**

The systematic off-by-one pattern is the smoking gun. The bilingual
LM's hidden state at `*` and `a` byte positions encodes "I am
reading the unary form" — that's the signal `inc_proj` learned
to read, and aux loss confirms gates fire correctly. But the
read_proj has to convert the counter state into a residual injection
that the *frozen, pre-trained* head will read as "emit `a`" vs
"emit `\n`". On the synthetic LM this was learnable end-to-end. On
this frozen bilingual LM, the head's natural disposition (trained
on `*N:aN\n` lines where N≤30, average ~15) carries a strong
bias toward emitting `\n` near the end of unary runs. The counter's
"still counting" signal isn't dominant enough to overcome it; the
LM emits `\n` one position early.

Things that would likely fix the off-by-one:
- Higher `injection_scale` (currently 10) — loud the counter more.
- Aux supervision on `\n` emission too (predict newline target),
  not just `inc` gates.
- Train at more diverse N values so the LM doesn't have such a
  strong "newline by ~15-30" prior.
- Or: the right architectural change is to give the counter a path
  that bypasses the head's tied-embedding bottleneck — but Phase 4
  of the synthetic experiment showed direct head-bias paths regress
  via capacity-splitting.

**Diagnostic: re-ran with `--injection-scale 30`** (3× louder).
Aux convergence identical (it's gate-only supervision). Result:

```
            scale=10 (original)    scale=30 (diagnostic)
N=3         OK ✓ → 3              OK ✓ → 3
N=30        FAIL → 29             OK ✓ → 30        ← off-by-one fixed
N=50        FAIL → 48             FAIL → 51        ← oscillates by 1
N=100..500  drops out of unary    drops out of unary  (same)
```

The in-distribution off-by-one **was** purely signal-magnitude-bound;
louder counter resolves it. That confirms the mechanism diagnosed
above. But OOD failure is unchanged: at N ≳ 80 stars the LM exits
unary mode entirely and emits dominant Spanish phrases (`'ér es el
problema...'`). This is **not** a counter-readout problem and not a
magnitude problem — it's the LM's implicit `stars→short→switch out`
prior, learned because training only saw `*N:aN` lines with N≤30.

The fix shape is therefore *upstream* of the counter:
- Train the bilingual LM with a wider N distribution (e.g., 1..200)
  so it doesn't acquire a "stars are short" prior, OR
- Distill from a stronger teacher (the JEPA direction in `jepa/`)
  so the student LM's hidden state at long unary runs is clean
  and the counter's signal can dominate it.

That's the actual next experiment to validate the strong cortex
composition claim.

**The actual position this leaves us in.**

The cortex thesis survives but with a sharper statement: forward-pass
primitives extend a language-trained LM's reasoning capability *with
fine-tuning of small adapters*, but the strong claim ("primitive
attaches and works without LM-side adaptation") is unproven and the
off-by-one suggests the more honest framing is "small-adapter
fine-tune, not adapter-free plug-in."

This points at the JEPA-Cortex direction (jepa/) the user was
preparing in parallel: rigorous hidden-state distillation gives
the student LM a more *informationally rich* hidden state than
plain byte-CE training, which in turn should give residual-stream
primitives a cleaner attachment surface. That is the experiment
that would actually validate the strong plug-and-play claim.

Reproduction:
```bash
python train_counter_attach.py \
    --lm-ckpt checkpoints/lm/step_FINAL.pt \
    --steps 1000 --unary-p 0.5 --lambda-aux 0.5 --lr 3e-3
python demo_cortex.py
```

Both checkpoints under `checkpoints/lm/` (frozen bilingual LM) and
`checkpoints/lm_counter/` (LM + trained counter adapter) — ~5.5 MB
each. ~26 min total on M4 Pro MPS (training + eval).

---

## Entry — Cortex: a 772-param counter primitive lets a 151k Mamba LM count byte-perfect to N=500 (2026-04-28)

Tested the **cortex thesis** end-to-end: rather than a small LM emitting
tool-call tokens that an external Python loop dispatches, treat
algorithmic primitives as **forward-pass modules in the residual
stream**. The LM emits gates from its hidden state; the primitive runs
its own arithmetic; its output re-enters the residual as a learned
embedding. No tokens, no parser, no Python loop outside the forward pass.

Single file: `cortex_counting.py`. Task: unary copy-the-length
(`*N:aN\n`). Train on N ∈ [1, 30], eval at N ∈ {3, 15, 30, 50, 100, 200,
500}. Six phases, all on the same byte-level Mamba-3 LM (2 layers,
d_model=96, ~150k params).

**Headline:**

```
              baseline      Phase 1     Phase 2     Phase 3.2    Phase 6+hard
N=3..30           OK            OK          OK            OK           OK
N=50              OK      FAIL(13)    FAIL(33)      FAIL(48)           OK
N=100       FAIL(72)       FAIL(3)    FAIL(29)      FAIL(88)           OK
N=200        FAIL(3)      FAIL(12)    FAIL(31)     FAIL(128)           OK
N=500       FAIL(67)    FAIL(None)    FAIL(29)     FAIL(183)           OK
```

A 772-param counter (`CounterPrimitive`) wired into the residual stream
of an otherwise unchanged 150,704-param Mamba-3 LM extends counting to
N = 500 — **16.7× past the longest training example** — byte-perfect.
Same architecture without the counter caps at N = 72.

**The four ingredients that mattered**, in order of impact:

1. **Aux BCE supervision on `inc_logits`** from byte-conditional targets
   (`inc[A]` should fire on `*`, `inc[B]` on `a`). Without this the
   counter sits unused — Phase 1 was *worse* than baseline at OOD.
2. **Tanh-saturated readout** with temperature k=8: `[tanh(c/k),
   tanh(diff/k)]`, no raw features. Bounded in (-1, 1) regardless of N.
   The `tanh(diff/k)` channel is the decisive "is c[A] ahead of c[B]"
   signal — same value at N=30 and N=500.
3. **Injection scale ≥ 3** on the `read_proj` output. Below 3, the
   counter contribution can't outvote the SSM's positional-OOD logits
   at long sequence positions, and the model under-counts proportionally.
4. **Hard-gate threshold at inference** — replace `sigmoid(inc_logit)`
   with `(inc_logit > 0).float()` when not training. Applied as a flag
   on the trained checkpoint, no retraining required. Diagnosis: at OOD
   positions the SSM hidden state has drifted, so sigmoid gates that
   fire at 0.999 in-distribution settle near 0.95; over hundreds of
   increments the slippage compounds to tens of missing counts.

**Cautionary tale (Phase 4):** adding a direct `Linear(read_in →
vocab)` from counter readout straight to LM logits — a "side channel" —
*regressed* OOD scaling from 88 to 40 at N=100. The model offloaded the
emit-a-vs-newline decision onto the cheap shortcut and trained the
residual injection less, leaving smaller total override authority.
Don't add side channels to a path that already works.

**Plugin/adapter refactor:** after the existence proof landed, the
architecture was generalised from hard-coded counter wiring to a
`Primitive` base class. `CortexLM(cfg, primitives=[...])` accepts any
list of `Primitive` subclasses; `CortexLM.forward` and
`CortexLM.aux_loss` are primitive-agnostic. Adding a new primitive is
now: subclass `Primitive`, implement `forward(x, tokens)` and
`aux_loss(...)`, append. Pre-refactor checkpoints (v3..v6) load via
state-dict key migration (`counter.*` → `primitives.0.*`).

**What this does NOT yet provide:** train-free composition. Each
primitive's adapter (gate Linears + read_proj) still co-trains with the
LM. The plug *interface* is plug-and-play; the *training story* is
still co-training. The next architectural gate is either a frozen-LM +
per-primitive adapter fine-tune, or a pre-trained "plugin port" — a
generic socket the LM is shaped to drive through a fixed protocol so
any conforming primitive slots in.

**What this does NOT cover:** language. The LM was trained on a single
synthetic task. Whether the same pattern works when the LM also has to
do English/Spanish next-byte prediction is the next experiment, and the
natural setup for testing whether primitives can attach to a
*language-trained* LM (the actual goal: a small LM that both speaks and
reasons).

Reproduction:
```bash
python cortex_counting.py train          # baseline + cortex (Phase 1)
python cortex_counting.py train_phase2   # + aux supervision
python cortex_counting.py train_phase3   # + tanh-only readout
python cortex_counting.py train_phase4   # + head-bias side channel  (regression)
python cortex_counting.py train_phase5   # + 3× injection scale
python cortex_counting.py train_phase6   # + 10× injection scale
python cortex_counting.py eval           # full 7-way comparison table
python cortex_counting.py demo           # baseline vs cortex side-by-side
```

Each phase ~12 min on M4 Pro MPS. Checkpoints under `checkpoints/cortex/`
(~150-160 KB each). The byte-perfect existence proof is on the v5 / v6
checkpoint with `counter.hard_gates_inference = True` (default in `eval`
and `demo`).

---

## Entry — LoopCounter primitive: additive injection isn't enough (2026-04-27)

**Setup.** Following the bidir result (rules out rightmost-byte
attention; the model learns a bounded periodic counter), I added
a `LoopCounter` module to `progressive_model.py`: an oracle-driven
embedding-table pathway. External code computes the per-position
counter trajectory — n at SEP, decrementing one per output
position, 0 at the EOS slot, sentinel elsewhere — and feeds it as
a second input to the model. The module looks up `c_emb[c_t]`,
projects, and adds a gated contribution to the model stream
before the LM head. Training uses HANOIBIN, n≤20.

**The pathway works structurally.** Teacher-forced EOS probe at
cycle 40 (loss 0.38, byte acc 100%, mix=0.1005, c_emb[0] norm
0.39, every other c_emb row near zero):

```
n=21..50 (OOD!) → predict EOS at counter==0 ✓
n=1..20 (in-dist) → predict '1' at counter==0 ✗
n=100 → fail
```

Autoregressive: 13/20 in-distribution (vs 0/20 baseline + bidir),
0/10 OOD past 20.

**The shape of the failure.** The model has learned `c_emb[0]` =
"strong EOS push" (this is the only counter row with meaningful
norm). At positions where the SSM is uncertain (n=21-50, just past
training range), the counter wins and EOS fires. At positions
where the SSM has memorized "still in answer cycle" (n=1..20), the
counter pathway is too weak to override. At very-OOD (n=100+) the
SSM's bounded-counter cycle wins again because its position-driven
"still in answer" signal accumulates faster than the counter's
single-position EOS push.

**Why the pathway stays weak.** `mix` barely moved from its 0.1
init across 40 cycles. The gradient on `mix` comes mostly from
the EOS slot (one position per example), while the rest of the
positions agree between SSM and counter (both want '1') and
provide no differentiating signal. With grad clip 1.0 and
loss averaged across many positions, the counter-pathway
gradient is tiny.

**The aux-loss attempt didn't help.** Added an EOS-weighted
auxiliary CE loss at counter==0 positions to concentrate gradient.
Weight=5: byte accuracy collapsed (100% → 16-83%, oscillating).
Weight=1.0: ~3 cycles of gentler turbulence then NaN at cycle 47.
The aux loss creates conflicts with the existing CE loss without
actually growing the counter pathway — `mix` stayed at 0.1005.

**What this proves.**

- ✓ Oracle-supervised primitive can predict EOS correctly at OOD
  ranges (counter pathway is structurally correct).
- ✗ Additive injection of an oracle primitive is insufficient
  when the SSM has competing learned patterns.
- ✗ Concentrating gradient via aux loss destabilises training
  without growing the new pathway.

**The architectural lesson.** The model needs a *gating* mechanism,
not addition. The counter pathway should be able to *override*
the SSM's prediction at counter==0 rather than competing with it.
Concretely: a direct counter→EOS logit bias

```python
logits[..., EOS] += eos_bias_table[c_t]
```

This bypasses both `mix` and the LM head's weight tying for the
specific case of "stop now" — the counter doesn't have to fight
the SSM, it just adds a hard preference.

Saved diagnostic checkpoints:
- `tower_of_hanoi_binary_loopctr.pt` (cycle 40, pre-aux, 13/20)

---

## Entry — EOS-bias gating: train n≤20, generalize to n=230 (2026-04-27)

**Setup.** After the LoopCounter additive injection failed
("additive isn't enough" entry), I added a second pathway: a
direct EOS-logit bias keyed off the same counter trajectory.
Implementation in `LoopCounter`:

```python
self.eos_bias = nn.Embedding(max_count + 2, 1)
# Hot init:
#   counter = 0  -> +30  (force EOS)
#   counter > 0  -> -15  (suppress EOS)
#   sentinel     -> 0    (input span / past-EOS, no bias)
```

In `ProgressiveModel.forward`, after `head(x)` produces logits, we
add `eos_bias[counter_values]` directly to the EOS logit column.
This bypasses the LM head's weight tying — the counter doesn't
have to fight the SSM's `'1'`-aligned hidden state through a
shared output embedding, it gets its own dedicated channel.

Also bumped `mix` init 0.1 → 1.0 so the c_emb pathway has full
presence in the hidden state from step 1.

**Why the gap from +5/+15 to +30 mattered.** The SSM trained on
HANOIBIN puts logit on `'1'` of ~44 at the EOS-target position
(after seeing several `'1'` tokens, weight-tied LM head's `'1'`
weight aligns with the hidden state). Bias of +5 → EOS logit ~20,
loses. Bias of +15 → EOS logit ~30, still loses (logit `'1'` even
after gradient pressure stays at ~44). Bias of +30 → EOS logit
~57, wins decisively.

**Training.** Same configuration as bidir (lr=1e-4, no aux loss,
no bidir input), HANOIBIN n≤20. 8 cycles to clear all 6 stages at
100% byte accuracy, loss=0.61. NaN at cycle 9 (same instability as
all previous Hanoi variants — gradient grows when sequence length
jumps to stage 6). Cycle 8 saved cleanly.

**Result. Trained on n≤20, evaluated autoregressively to n=256:**

```
n =   1..20  : 20/20  ✓  (in distribution)
n =  21..100 : 80/80  ✓  (4–5x training length)
n = 101..230 : 130/130 ✓  (11x training length)
n = 231      : off-by-one (230 ones for n=231)
n = 232..256 : pred_len drifts down to ~225-230
```

**The model trained on n≤20 emits exactly n ones for every n in
[1, 230].** The cliff at 231 is graceful — the model produces
~230 ones for any n thereafter, suggesting the SSM's residual
`'1'`-attractor wins once the hidden state has been integrating
`'1'` tokens for ~230 timesteps.

**What this proves architecturally.** The previous diagnostics
(HANOIBIN, bidir, additive LoopCounter) all showed that the
model could not extract n from its input *and* count to it. The
EOS-gate experiment splits that:

- **n is provided externally** (oracle/parser reads input bytes)
- **The neural recurrence handles the loop** (read counter →
  decrement → check zero → emit / stop)

With this split, a 124k-param Mamba-3 trained on n≤20 runs the
loop correctly for n up to 230. The architectural ceiling we
identified was specifically about *fusing* parse + count in the
SSM's hidden state. Given a clean primitive interface, the SSM
is a competent loop executor.

**The lesson generalizes.** This is the "tool use" pattern: the
neural model orchestrates a primitive, doesn't replace it. For
unbounded computation tasks at this scale, the architectural
move is to *factor* the program — externalize parts the SSM
can't represent (unbounded counter, exact arithmetic), keep the
SSM in the loop body (decision, output token).

Saved checkpoint: `tower_of_hanoi_binary_eosgate.pt` (124k params,
loss 0.61, byte-acc 100%, length-gen 230/230 to n=230, gentle
falloff to n=256).

**Iron-solid v2: bidirectional gating.** The cliff at n=231 wasn't
about EOS — it was the SSM's logit on **SEP=258** growing faster
than `'1'`=49 as the answer span deepened (~0.05 per position).
At k≈232 they crossed; the model emitted SEP (filtered from the
printable answer string, hence the off-by-one).

Fix: mirror the EOS bias on the iteration token. Added
`iter_bias` to `LoopCounter`: per-counter scalar bias on
`logit[iteration_token=49]`. Hot init: `c=0 → −30`, `c>0 → +15`,
sentinel → 0. Now the LoopCounter explicitly encodes loop-body
semantics: "while counter>0: push iteration_token; at counter=0:
push EOS." Both biases bypass weight tying.

Re-trained from scratch (same config), cycle 11 gave loss 0.51
without NaN (improvement over v1's NaN at cycle 9).
Length-gen n=1..256: **256/256 ✓**. Counterfactual (feed input
n, counter m) 13/13 — emits exactly m ones regardless of input.
Edge cases (n=0, n=1, n=256) 3/3.

The recurrence is iron-solid out to the counter table's max.

Saved: `tower_of_hanoi_binary_eosgate_v2.pt`.

---

## Entry — FIB-decimal: per-position iter_token, train n<=20, perfect to F(40) (2026-04-27)

After HANOIBIN/FIB-unary established the LoopCounter pattern with a
single iteration token, decimal Fibonacci is the first task where the
iteration token *varies per position* — at output position k of FIBD
n, the model emits the k-th digit of str(F(n)). Counter at SEP =
digit_count(F(n)), decrementing per position; eos_bias dominates at
counter=0; iter_bias dominates the per-position digit at counter>0.

**Architectural extension.** `LoopCounter` now exposes an
`iter_token_per_pos` channel: instead of a fixed scalar
`iteration_token`, the oracle passes a (B, L) tensor specifying
which token to bias UP at each position. `forward` /
`forward_step` use scatter_add to route the iter_bias amount to
the position-specific token column. HANOIBIN / FIB-unary still
work with the scalar fallback; FIB-decimal uses the new path.

**iter_bias init had to grow.** With variable iter_token, the LM
head's weight tying creates a per-position adversary: at position
sep+k+1 the model has just consumed the k-th answer digit, so
`logit[digit_k]` reaches ~50 (weight-tied alignment with
embed(digit_k)). +15 bias on the *next* digit (digit_{k+1}) was
overwhelmed. Bumped iter_bias hot init from 15 to 50 — the model
locks onto the correct digit immediately and convergence is
dramatically faster (acc 23% at cycle 1 vs 0% with +15).

**Result.** Train fib_decimal n<=20 (max digits = 4, F(20)=6765),
30 cycles, lr=1e-4, no NaN, final loss 0.26. Eval n=1..40 with
step-decoder:

```
n=1..20:  20/20 ✓ (in distribution)
n=21..40: 20/20 ✓ (out of distribution, max 9 digits)
```

n=40 → "102334155" (9 digits, 2.25x training-max digit count).
Each digit emitted is the *correct* digit of F(n) — the model
isn't just emitting any digit of the right length, it's emitting
the *exact* digits the oracle specified.

**fib_decimal_validate.py.** 57 tests across length-gen,
counterfactual (input vs target divorced), and edge cases (n=0
"0", far-OOD). All pass byte-for-byte vs Python reference in
1.2s.

**Extrapolation depth.** Pushed validate further:
  - n=1..100: 100/100 ✓ (5x training length, max digits 21)
  - n=1..200: 196/200 (4 failures at n=144, 187, 191, 199)

The failures all share a shape: model emits `'0'` instead of the
correct digit at deep answer-span positions (k=30+). For example
n=191 emits "...833800000000" (8 zeros tail) instead of
"...833808526209". Same shape as the HANOIBIN SEP-drift cliff —
deep answer-span positions accumulate an SSM hidden-state bias
that overwhelms the +50 iter_bias.

**v3: extended curriculum closes the cliff.** Trained with stages
up to n_max=60 (13-digit answers; max stages 1-6 cleared, NaN at
stage 7 transition to n_max=100). Saved cycle-9 checkpoint
(`fib_decimal_deep_n60.pt`):

  - n=1..200: **200/200 ✓** (3.2x depth extrapolation: 13-digit
    training -> 42-digit eval) — iron solid in the supported range.
  - n=1..500: 5 failures starting at n=268 (~56 digits). Cliff is
    pushed out from n=144 (depth 30) to n=268 (depth 56) by
    exposing the SSM to deeper answer spans during training.

Training cap (n_max in curriculum) ↔ extrapolation depth scales
roughly linearly: 4-digit cap ↔ depth-30 cliff; 13-digit cap ↔
depth-56 cliff. The architecture works; the limitation is that
the SSM's hidden-state at position k+answer_offset has only been
supervised up to k_train, so deep positions fall back to the
weight-tied LM head's "predict the previous token" or "0" mode.

The fix that stayed *inside* the iron-solid bar: train deeper
curriculum, not bigger model.

The per-position iter_token primitive generalizes cleanly. The
loop body is now: "while counter>0: emit iter_token_per_pos[t];
at counter=0: stop." Both the WHEN-to-stop and WHAT-to-emit
signals come from the oracle; the SSM provides the recurrence.

Saved: `tower_of_hanoi_binary_eosgate_v2.pt` (HANOIBIN, 256/256),
`fib_unary.pt` (FIB-unary, 20/20), `fib_decimal.pt` (FIBD, 40/40).

---

## Entry — Parameter-free LoopCounter: truly unbounded (2026-04-27)

User's challenge: "I still see mentions of extrapolation. But a true
computation wouldn't have just a set limit however big of extrapolation
it would be perfect."

The challenge was right. Earlier "256/256" results hid a real
limit: `LoopCounter`'s `c_emb`, `iter_bias`, `eos_bias` were all
embeddings of shape `(max_count+2, ...)`. Beyond max_count we
clamped to sentinel. Counter values 21..max_count generalised
only because the hot init was uniform — but the architecture
genuinely had a hard cap.

Inspecting trained biases revealed the cap was a fiction:

```
iter_bias[1]:  +50.10
iter_bias[5]:  +50.00
iter_bias[20]: +49.83
iter_bias[100]: +49.83  (init, never seen)
iter_bias[256]: +49.83  (init, never seen)
```

For c>0 the bias is essentially constant. The model only ever uses
the **sign** of c (sentinel/zero/positive), not the integer value.
A `(max_count+2, ...)` table is a wasteful encoding of a 3-valued
flag.

**Refactor.** Replaced the embedding tables with `torch.where`
dispatch on sign, plus 2 d_model embeddings (`stop_emb` /
`iter_emb`) and 4 scalar bias parameters. No `max_count`. Sentinel
is now -1 (any negative value).

```python
def get_eos_bias(self, c):
    is_zero, is_pos = (c == 0), (c > 0)
    return torch.where(is_zero, self.eos_bias_zero,    # +70 init
           torch.where(is_pos, self.eos_bias_pos,      # -30
                       torch.zeros_like(c, dtype=...)))   # 0 sentinel
```

LoopCounter parameters drop from ~70k (max_count=256) or ~270k
(max_count=1024) to **4,293**. Total ProgressiveModel: 124k -> 108k.

**Demonstration of unboundedness.** Trained HANOIBIN with the
parameter-free arch on n≤20 (lr=5e-5, 30 cycles, no NaN — first
clean run of the week). Step-decoder eval at increasing n:

```
n =     20:  20/20 ✓ (in distribution)
n =    100: 100/100 ✓
n =    256: 256/256 ✓ (the old "ceiling" is now nothing special)
n =   1000: 1000/1000 ✓
n =   5000: 5000/5000 ✓
n =  10000: 10000/10000 ✓ (500x extrapolation)
n =  50000: 50000/50000 ✓ (2500x)
n = 100000: 100000/100000 ✓ (5000x)
```

**The computation is genuinely unbounded.** The model trained on
n≤20 emits exactly n ones for any n we hand it. Compute scales
linearly via step decode: n=100k generates in 704s @ ~142 tok/s
on M4 Pro. There is no numerical or architectural cliff in the
range tested.

**What this proves.** The "extrapolation factor" framing was a
self-imposed limit. With the table cap removed:

- HANOIBIN: arbitrary n. The recurrence is the program.
- The remaining limit is *training-curriculum depth* for tasks
  where the SSM hidden state at deep answer-positions has to
  predict different content per position (FIB-decimal). For
  tasks where content is constant (HANOIBIN) the SSM never
  needs deep-position supervision and the architecture is
  truly unbounded.

Saved: `tower_of_hanoi_binary_paramfree.pt` (124,472 storage,
107,832 actual params via weight tying).

---

