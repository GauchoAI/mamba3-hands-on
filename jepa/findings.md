# JEPA-Cortex — experiment journal

A living document for the JEPA-Cortex run. Three sections:
1. **Hypothesis** — what we set out to test, written before launch.
2. **Live observations** — timestamped notes as the runs unfold.
3. **Conclusion** — the final read once training stops.

Cross-project learnings (lessons that generalize beyond this experiment)
get folded back into the root `findings.md`. This file holds the full
detail of the run.

---

## 1. Hypothesis (ex ante, 2026-04-29 morning)

**Claim under test.** JEPA-style latent distillation (predict the teacher's
*next thought-trajectory hidden* in latent space) is a strictly better
training signal for a small bilingual byte-level Mamba-3 LM than plain
pseudo-label distillation (predict the teacher's *next byte*). At the
same param count, on the same teacher corpus, a JEPA-trained student
should:

- generalize better on held-out novel text (we use `data/bilingual.txt`
  as the OOD probe — different distribution from the Qwen-generated
  teacher corpus),
- maintain prompt-conditioning under small-corpus pressure (i.e. not
  collapse onto a single high-confidence response mode),
- preserve the cortex residual-stream primitives (CounterPrimitive)
  through training without interference.

The JEPA term is gated by `jepa_warmup` (default 2000 steps) so the LM
head locks in basic byte fluency before the latent objective fires. The
isotropy regularizer `sigreg` (Cramér-Wold KS-distance plus a per-dim
magnitude term) prevents the intent embedding from collapsing to zero —
known failure mode of joint-embedding objectives.

**Why four parallel variants instead of one.**

The four 4070 Ti cards on the rented box make 1 vs 4 experiments cost
the same wall-clock — so we run a small loss-weight sweep instead of a
single big run. The variants are designed to *triangulate* the loss
landscape, not just to find the best point:

| Run | λ_jepa | λ_sigreg | λ_aux | batch | Reasoning |
|---|---|---|---|---|---|
| gpu0-ref       | 1.0 | 0.1 | 0.5 | 64 | Canonical full-strength JEPA. The "does this even help?" datapoint. |
| gpu1-lowjepa   | 0.3 | 0.1 | 0.5 | 64 | Maybe λ=1.0 is too aggressive on a small corpus (~310 records). Lower pressure may keep the byte head intact while still getting *some* regularization. |
| gpu2-highsig   | 1.0 | 0.5 | 0.5 | 64 | SIGreg's job is anti-collapse. Higher λ_sigreg should produce visibly more isotropic intent — a stronger geometry-of-thought claim. |
| gpu3-zerojepa  | 0.0 | 0.1 | 0.5 | **32** | Control. Without it we can't claim anything. **Note the batch-size mismatch** — gpu3 shares GPU 3 with the teacher generator so we kept it smaller. This is a confound we'll have to discount in the comparison. |

What I expect, in advance of any data:

- **Most likely (60%):** JEPA-on runs improve on bilingual held-out CE
  while gpu3-zerojepa overfits and degrades. Not by a huge margin (~0.1
  nats) but visible.
- **Plausible (25%):** JEPA loss saturates around 0.7-0.9 (predictor
  latches onto easy invariants in the teacher hiddens — markdown
  structure, sentence boundaries — and stops). Bilingual CE flattens.
- **Less likely (10%):** JEPA over-regularizes at λ=1.0; gpu0 fluency is
  visibly worse than gpu1. Would tell us λ_jepa sweet spot is ~0.3-0.5.
- **Long shot (5%):** JEPA's residual-shape effect indirectly trains the
  counter primitive's stop-gate; count_acc rises spontaneously.

Counter primitive (`count_acc_30`, `count_acc_100`) is the secondary
research interest; we don't expect it to land in this run, since
`λ_aux=0.5` plus byte-position-only BCE supervision was insufficient
in the standalone counting experiment too.

---

## 2. Live observations

### 2026-04-29 — bring-up

- vast.ai 4× RTX 4070 Ti, 12 GB each, ~40 TF BF16, ~504 GB/s.
- First trainer launch at `batch_size=64` OOM'd at 11.12 GiB (the SSM
  scan keeps activations for backward across 4 layers × seq 512). Cut
  to `batch=16, seq=256` → 2 GB used. Way over-corrected. Settled on
  `batch=64, seq=256` → 6.7 GB used / 100% util on each card. **The
  default config in the trainer was wrong for 4070 Ti**; will fix in
  `train.py` defaults once the run finishes.
- Teacher generator (Qwen-2.5-1.5B BF16) ran on GPU 3 alongside, ~14%
  GPU util (sample-bound on the per-token tokenizer.decode loop in
  `thought_positions`); no contention with the gpu3-zerojepa control.
- Teacher corpus reached the 80 MB target in ~2h. ~310 records.

### Step 0–1000 (pre-JEPA, pure byte CE + aux + sigreg)

All four runs identical until step 2000 except gpu3 (different batch).
gpu0/gpu1 run identically to 4 decimals — same seed, JEPA loss
multiplied by zero, no divergence.

byte CE drops cleanly: 5.6 → 5.0 → 4.1 → 2.5 → 1.6 over the first 1000
steps. By step 1000 the model produces real bilingual sentences:

- *"I just discovered an ancient manuscript hidden behind."*
- *"Tengo que cuidar de mi tiempo y establecer horarios"*
- Number-word pairs: `4 :: cuatro`, `5 :: cinco` (correct).

intent_var is small (0.003-0.005) but *growing* under SIGreg pressure.

### Step 1000–2000 (still pre-JEPA, model gets more verbose)

byte CE drops further to ~1.0-1.2. Real fluency emerging:

- *"She leaned out the window to catch a glimpse of"*
- *"What time are we meeting."*
- *"Un viento suave flota a través de las hojas."*
- *"It's been raining non-stop all day."*

The model also picked up Qwen's distinctive refusal voice
(*"I cannot complete this task as requested..."*) — pseudo-label
distillation transferred *style*, not just lexicon.

Counter primitive: emits the right *character* (`***:` → "aaaa…")
but doesn't stop. count_acc_30 = 0 across all runs. Aux loss is
dropping but the stop-signal hasn't materialized.

intent_var divergence by SIGreg config visible at step ~1000:
gpu2-highsig (λ_sigreg=0.5) at 0.039, others at 0.005. SIGreg working
as designed.

### Step 2000–2500 (JEPA fires)

First teacher batch with `jepa_w > 0` at step 2150 (`w=0.15`):

```
step=2150  byte=0.026  jepa=1.6204 (w=0.15)  sig=0.558  total=0.325
```

JEPA contribution = 0.24, 74% of total loss. Suddenly the dominant
gradient signal. Note: jepa_loss had already drifted from 1.95 → 1.62
during warmup — purely from the residual changing under byte-CE
training (the head itself never received gradient since `loss = 0 *
jepa_loss`, but the residual it reads moved on its own).

By step 2400, gpu0 and gpu1 metrics start to **diverge** (they were
identical for 2000 steps). Tiny drift (Δ byte_ce_teacher ≈ 0.026) but
real — the JEPA gradient is steering them apart.

### Step 2400–2800 (JEPA contributing meaningfully)

Bilingual byte CE trajectories:

| step | gpu0-ref | gpu1-lowjepa | gpu2-highsig | gpu3-zerojepa |
|---|---|---|---|---|
| 2200 | 1.443 | 1.441 | 1.488 | (s.2200 not aligned) |
| 2400 | 1.348 | 1.351 | 1.387 | s.2400 → 1.297 |
| 2600 | 1.307 | 1.306 | 1.341 | s.3600 → 1.416 (rising) |
| 2800 | 1.253 | **1.249** | 1.291 | s.4000 → 1.310 (recovered) |

JEPA-on runs improve monotonically. JEPA-off oscillates (climbed to
1.42 mid-overfit, then bounced back).

teacher byte CE picture is opposite: gpu3-zerojepa best (1.52 at step
4000) because it memorizes the small corpus harder. JEPA-on runs at
~1.94-2.04 — slightly worse, because the JEPA gradient pulls residual
toward thought-trajectory matching rather than exact byte memorization.

The trade is exactly what a regularizer should do: lose a little on the
training distribution, gain on the held-out *real text* distribution.

### Step 2800 — qualitative comparison (the headline finding)

Side-by-side talk.py at the same step on each run, same seed=42, same
six prompts, same temp=0.7. Counted distinct meaningful response openings
across the six prompts:

| Run | distinct outputs / 6 | best fluent fragment |
|---|---|---|
| gpu0-ref      | ~5 | *"Tom se habla cabeza de dormir a los niños"* |
| gpu1-lowjepa  | ~5 | *"10::Diez, 11::Once, 12::Doce"* (clean Spanish drill) |
| gpu2-highsig  | ~5 | *"Tengo que mucho alguien pasado con mí"* |
| **gpu3-zerojepa** | **~2** | most prompts return the same phrase: *"...about help you ever sick. :: Mucha a pasado por mí. I don't see you tellim a dinner..."* |

gpu3-zerojepa is **mode-collapsed** — four different prompts ("Hello,
how are you?", "Hola, como estas?", "I went to the market…", "The cat
sat on the") yield nearly the same response. On "Mi color favorito es"
it falls into unary mode (`***:aaaa…`), the simplest learned pattern.
JEPA-on runs all differentiate prompts.

This is the most striking observation of the run. Not a 0.06-nat
quantity — a *qualitative behavior change*. JEPA's gradient on the
residual is preserving prompt-conditioning under the small-corpus
pressure that mode-collapses byte-CE-only training.

### Step 2900 — gpu3-zerojepa retired, replaced with gpu3-bigaux

After documenting the mode-collapse finding, killed gpu3-zerojepa and
launched a new variant informed by what we'd learned:

```
gpu3-bigaux:
  λ_jepa = 0.3      ← qualitative sweet-spot (gpu1's clean number drill)
  λ_sigreg = 0.1    ← default; high made gpu2 prompt-blind
  λ_aux = 2.0       ← 4× the others, push counter-stop into existence
  batch_size = 64   ← matched, eliminate the confound
```

The headline scientific question for this variant: does λ_aux=2.0
surface a stop signal in the counter primitive that λ_aux=0.5 couldn't?

### Step 3000–4000 — full JEPA ramp

JEPA loss has dropped further on the mature runs:

| step | gpu0 jepa | gpu1 jepa | gpu2 jepa |
|---|---|---|---|
| 2150 (warm-end) | 1.62 | 1.68 | 1.59 |
| 2400 | 1.21 | 1.29 | 1.20 |
| 2700 | 1.06 | 1.12 | 1.00 |
| 3000+ | 0.75 | 0.85 | 0.74 |
| ~4000 | **0.28** | 0.34 | **0.25** |

The thought_head genuinely learned to predict teacher hiddens. JEPA
loss is still dropping at step 4000.

Bilingual held-out CE has continued to improve with JEPA on:

| step | gpu0 | gpu1 | gpu2 |
|---|---|---|---|
| 2800 | 1.253 | 1.249 | 1.291 |
| 3800 | 1.214 | **1.199** | 1.234 |

gpu1-lowjepa still leading on the headline metric. The picture
crystallizing: **moderate JEPA pressure beats both no-JEPA and
full-pressure** on this corpus size.

count_acc_30 / count_acc_100 still 0 across all mature runs. Counter
primitive's stop-gate did not materialize during the full JEPA ramp.

---

## 3. Conclusion — stopped early at step ~4000–4200, all four runs degrading

We stopped the run when the qualitative degradation became clear. The
quantitative metric continued to look fine (gpu1-lowjepa hit byte_ce_biling
= 1.158 at step 4200, the lowest of the entire experiment), but **side-by-
side talk.py samples at step 4000 looked worse than at step 2800**:

```
step 2800 (good window)
  >>> Hola, como estas?    "Un viento suave flota a través de las hojas."  ← grammatical
  >>> Tom said             "What time are we meeting."                      ← grammatical
  >>> ***:                 aaaaaaaaaaa Thank in your libro can hambrella?

step 4000 (degraded)
  >>> Hello                Tom are much here to be a strict... (gpu0)
  >>> Hola                 Tom are much here to be a strict... (gpu0, same opener!)
  >>> Hello                ***************:aaaaaaaa  (gpu1, fell into unary)
  >>> Tom said             ***************:aaaaaaaa  (gpu1, fell into unary)
```

By step 4000 **all three JEPA-on runs fall into the unary attractor**
(`***:aaaaa…`) on at least one prompt. They didn't at step 2800. The
qualitative gap with gpu3-zerojepa we saw at step 2800 has closed —
not because gpu3 got better but because the JEPA-on runs got worse.

### Best snapshots, by qualitative quality

The best output we got from this experiment was the **step 1800 sample
on gpu3-zerojepa** (before it mode-collapsed; we replaced it with
gpu3-bigaux at step 2900). The gpu0-ref step 2800 ckpt is a close
second. Those two checkpoints are what we'd ship as "the demo."

### The byte_ce-vs-qualitative disconnect

Held-out byte_ce_biling kept improving even as samples got worse. Why:

- byte_ce is the per-position average negative-log-likelihood of the
  next byte given context. The model's distribution over byte 5,000
  in a 1.16-nat-CE run is *just slightly* better than at 1.20 — but
  the next-byte distribution doesn't tell us anything about what
  happens 200 generated bytes later, which is where the unary
  attractor traps autoregressive generation.

- Mode collapse is a *trajectory* property, not a per-position one.
  It shows up in long-form generation, never in held-out byte CE on
  a corpus of short sentences.

This is a generalizable lesson: **byte CE is not a reliable stop
signal for small bilingual byte LMs.** We need a sample-quality
metric (response diversity over a fixed prompt set, repetition rate,
attractor-frequency) running alongside, or we'll keep stopping in
the wrong place.

### What worked

1. **Pseudo-label distillation lands quickly.** A 1M-param byte-level
   Mamba-3 produces fluent bilingual sentences in ~6 hours on 4×4070Ti
   with ~80 MB of Qwen-2.5-1.5B-generated paired bilingual data. The
   teacher's *style* (markdown structure, refusal voice, number-pair
   format) transferred. Number-word pairs (16=Dieciséis, 18=Dieciocho)
   emerged without explicit supervision.

2. **JEPA loss converges.** From 1.95 → 0.25 over 1800 post-warmup
   steps. The thought_head genuinely learned to predict teacher hiddens
   in latent space.

3. **SIGreg's magnitude term works.** intent_var grew from 0.005 → 0.72
   on gpu2-highsig (λ_sigreg=0.5) — a 144× increase. The Cramér-Wold
   isotropy + magnitude calibration is doing what was specified.

4. **Mid-training (step ~2000-2800) the JEPA gradient prevented mode
   collapse** that gpu3-zerojepa exhibited at the same window. This was
   real, observable, and what we hoped for.

5. **CounterPrimitive's gates fired correctly.** The right *character*
   ('a') for `***:` prompts. Plus we saw the unary task pattern emerge
   spontaneously on natural-language prompts (gpu3 at step 1800 invoked
   `***:aaaaa…` as one of its modes).

6. **Cortex residual injection didn't break language learning.** The
   model trained smoothly through the JEPA-thought-pressure phase
   without any of the destabilization we saw in earlier cortex
   experiments.

### What didn't work

1. **Late-training (step ~3000+) all JEPA-on runs mode-collapsed
   onto the unary attractor**, partly because the unary task is so
   cleanly predictable that the JEPA gradient (now matching teacher
   thoughts on the 5% counting-prompt records) reinforces it as a
   safe high-confidence mode.

2. **Counter primitive's stop signal never materialized.** count_acc_30
   = 0 across every run. λ_aux=0.5 + position-BCE on `*`/`a` bytes is
   insufficient. gpu3-bigaux (λ_aux=2.0) was meant to test if more aux
   pressure surfaces the stop signal, but only got to step 700 before
   we stopped.

3. **The clean JEPA-vs-no-JEPA comparison stayed confounded.**
   gpu3-zerojepa was at batch_size=32 (sharing GPU 3 with the teacher
   generator) while the others ran batch=64. We can claim "JEPA prevents
   mid-training mode collapse" qualitatively but not "JEPA-on lowers
   bilingual byte CE by X" quantitatively.

4. **The corpus is too small for 30k steps.** ~310 records × batch 64
   × 30k steps ≈ 6,000 epochs. Pure memorization regime by step 2400;
   late-training degradation by step 4000.

5. **byte_ce_biling failed as a stop signal.** It said "still
   improving" while samples said "actively getting worse." Auto-stop
   off this metric would have shipped the worst checkpoints.

### Generalizable lessons (folded back into root findings.md / memory)

- Byte CE is not a reliable stop signal for small bilingual byte LMs.
  Need a sample-quality metric (response diversity over fixed prompts,
  attractor frequency) running alongside.
- A 5% unary mixin in the teacher corpus is a too-attractive mode for
  a 1M-param byte-level LM trained to memorization. Either remove it
  from the corpus, or train against a much larger non-unary corpus so
  the unary fraction is materially smaller.
- JEPA-style latent regularization helps mid-training (prevents the
  mode collapse that pure byte-CE produces) but does not save you from
  late-training collapse on a tiny corpus.
- Pinning checkpoints by training byte loss (current AsyncCheckpointer
  behavior) is wrong for this regime; we'd want pinning by held-out
  metric *and* sample-quality.

---

> **Side experiment forked off (2026-04-30):**
> While round 2 is cooking, we noticed
> [`batteryphil/mamba2backbonerecursion`](https://github.com/batteryphil/mamba2backbonerecursion)
> on Reddit — Mamba SSM with recursive latent forcing (loop the hidden
> state through the same layers N times before emitting each token).
> Genuinely interesting and at least partially-validated (their 2.8B
> hits 75% on BIG-Bench Lite) so we forked the experiment off into
> [`rlf_cortex/`](../rlf_cortex/) — same baseline as this branch at the
> moment of the fork, plus an `--n-loops` flag that runs the SSM stack
> N times per token with a decayed lifeline of the original embedding
> re-injected each loop. The two experiments share the data directory
> but namespace their checkpoints + runs separately so they don't
> interfere with each other.

> **Round 2 headline result (2026-04-30):**
>
> The single architectural change that produced the most fluent
> bilingual output is **removing the CounterPrimitive entirely**, not
> any of the JEPA-loss-weight knobs we expected.
>
> At step 4600 / byte_ce_biling matched-step:
>
> | Variant | byte_ce_biling | diversity | best fluent fragment |
> |---|---|---|---|
> | gpu0-pure-bilingual (counter present, no aux) | **1.21** | 0.46 | "There are sold has been took in the filosopirent" |
> | **gpu1-no-cortex (no counter at all)** | 1.27 | **0.77** | *"¿Cuántos años tienes?"*, *"Los días lluviosos son ideales para pasar tiempo en casa leyendo"*, *"Me gustaría encontrarme"* |
> | gpu2-tinier (counter, d_model=128) | 1.30 | 0.38 | mode-collapsed onto unary attractor (`*****:aaaaa`) on 3+ prompts |
>
> The byte CE penalty for removing the counter is small (0.06 nats);
> the diversity gain is large (0.31 nats of pairwise Jaccard distance);
> the qualitative gain is enormous — *"¿Cuántos años tienes?"* and
> *"Los días lluviosos son ideales para pasar tiempo en casa leyendo"*
> are the most grammatical Spanish the project has ever produced from
> a from-scratch ~1M-param byte-level LM.
>
> **Why removing the counter helps:** with --mix-unary 0 (round 2) the
> CounterPrimitive's gates never get aux supervision. The primitive sits
> in the residual stream emitting whatever its untrained gates produce,
> which is statistically structured noise that the LM head learns to
> *predict* (it correlates with the byte stream because both flow from
> the same residual). The model spends capacity modeling its own untrained
> circuit. Even worse, when the corpus has *any* unary-shaped data, the
> counter becomes an attractor — gpu2-tinier mode-collapsed onto unary
> at step 6200 even though `--mix-unary 0` removed the synthetic batches.
>
> **Generalizable lesson:** the Cortex thesis (residual primitives extend
> small-LM reasoning) holds for *targeted* primitive tasks (the original
> 151k-LM byte-perfect counter to N=500 still works). It does *not* hold
> for "let the LM use the primitive opportunistically while training on
> a different task" — co-training a primitive with no aux supervision is
> net-negative. The primitive needs either explicit aux supervision or
> to be absent. Half-measures hurt.
>
> Pinned: `checkpoints/jepa_cortex_pinned_round2/gpu1-no-cortex/light_step_0004600.pt`.
> Cross-project takeaway folded into root `findings.md`.

## 4a. Round 3 — knob sweep around the gpu1-no-cortex winner (2026-04-30 PM)

After the round-2 headline result landed (gpu1-no-cortex wins on the
metric that matters: diversity at near-equal byte CE), the two
"failed-mode" variants from round 2 (gpu0-pure-bilingual and
gpu2-tinier) were retired and replaced with focused follow-ups around
the winning config. Each replacement varies exactly *one* knob from
gpu1's config — clean experimental design.

### Why retire the round-2 failures

`gpu0-pure-bilingual` and `gpu2-tinier` did exactly what we expected
them to do — they validated the half-measures-hurt hypothesis:

- gpu0 (counter present, no aux): byte_ce 1.21 (best on this metric)
  but diversity 0.46 → mode-collapses on greedy decoding ("There are
  sold has been took in the filosopirent" repeats across prompts)
- gpu2 (counter, d_model=128): mode-collapsed onto unary attractor
  `*****:aaaaa` on 3+ prompts despite `--mix-unary 0`

Their best checkpoints are pinned to
`checkpoints/jepa_cortex_pinned_round2/{gpu0-pure-bilingual,gpu2-tinier}/`
(12 MB total, including the per-run metrics + samples logs) so the
data isn't lost. They aren't going to teach us anything new from
here, so we replace them.

### Round 3 variants (running now)

The four GPUs now host:

| GPU | Run | What's varied vs gpu1's winning config |
|---|---|---|
| 0 | gpu0-no-cortex-highjepa | `λ_jepa = 1.0` (vs 0.3) — does more JEPA pressure help when the cortex isn't there to absorb it? |
| 1 | gpu1-no-cortex | (winner; continues running) |
| 2 | gpu2-no-cortex-highsig | `λ_sigreg = 1.0` (vs 0.3) — does pushing isotropy further widen the diversity gap? |
| 3 | gpu3-no-cortex-bigger | `d_model = 256` (vs 192) — does more capacity help or hurt without a counter? |

All four use:
- `--use-counter false` (the round-2 winning move)
- `--mix-unary 0` (no synthetic unary batches)
- `--mix-teacher 0.7 --mix-biling 0.3`
- `--steps 10000`
- `--batch-size 64 --seq-len 256` (gpu3 uses 32 to fit d_model=256 in 12 GB)

### What round 3 will tell us

- If gpu0-no-cortex-highjepa beats gpu1: λ_jepa=0.3 was too low for
  no-cortex; the JEPA term wants more pressure when not competing with
  a counter primitive's gradient pull.
- If gpu2-no-cortex-highsig beats gpu1: λ_sigreg=0.3 was leaving
  diversity on the table; pushing isotropy harder helps.
- If gpu3-no-cortex-bigger beats gpu1: 1M params was the bottleneck,
  and 1.6M / d_model=256 is a strict improvement on this corpus.
- If gpu1 stays best: we've found a local maximum; further gains need
  a structural change (Cerebras corpus, HaltingHead, etc.) not a knob
  twist.

The diversity metric is now in eval_daemon by default, so this round
is the first one where the dashboard will surface mode-collapse
detection live (rather than us discovering it after-the-fact).

---

## 5. Next round of experiments — proposal (kept from round 2 for reference)

The next round attacks the failure modes we observed:

| Variant | Idea | Why |
|---|---|---|
| **gpu0-pure-bilingual** | `mix-unary=0`, λ_jepa=0.3, λ_sigreg=0.1, no synthetic unary batches at all. CounterPrimitive still in the model architecturally, just untrained. | Tests whether the mode-collapse is corpus-level (the 5% counting prompts in the teacher corpus) or trainer-level (the synthetic unary batches). If gpu0 still mode-collapses, the corpus is the cause. |
| **gpu1-no-cortex** | `mix-unary=0`, λ_jepa=0.3, `use-counter=False`, no CounterPrimitive, no aux loss. Pure bilingual byte LM with JEPA. | Tests the language model alone, no cortex. If this is the fluent one, we know the cortex is causing the late-stage collapse and we re-attach it carefully later. |
| **gpu2-tinier** | `mix-unary=0`, λ_jepa=0.3, d_model=128 (down from 192), 4 layers. ~500k params. | Smaller model = less capacity to memorize the small corpus = less likely to overfit and collapse. The standalone Cortex existence proof was 151k params; we may have overshot. |
| **gpu3-largercorpus** | Same as gpu0 but **regenerate teacher corpus to 240 MB** (3× current) with `--counting-fraction=0`. Then train. | The honest fix: more data, no unary attractor in the corpus. Takes ~6 hours of teacher gen first. |

All variants:
- 10000 steps instead of 30000 (we saw that step 2000-3000 was the
  sweet spot; ditto here likely)
- Save canary samples every 50 steps (down from 100) so we have
  finer-grained qualitative snapshots
- Track *response diversity* in the eval daemon: compute Jaccard
  similarity across canary completions for the same step. When it
  collapses below threshold, alarm.

The clean comparison this time: same batch size on all four, same
seq_len, same total steps. The only differences are the variant axes.

### What we'd need *before* relaunching

1. **Add a `mix_unary=0.0` smoke test** to confirm the trainer handles
   it correctly (no division-by-zero, no zero-batch errors).
2. **Add `--use-counter False` flag handling** to make sure the model
   builds without a CounterPrimitive and the aux path skips cleanly.
3. **Add response-diversity metric to eval_daemon** — Jaccard or
   character-bigram-overlap across the 8 canary prompts' completions.
   Above ~0.7 = collapsed.
4. **For gpu3-largercorpus only:** kick off teacher generation with
   `--counting-fraction 0 --target-mb 240` first (~6 h on GPU 3).
   Other variants can train against existing corpus immediately.

These are the precondition tasks before the actual re-launch.

---

## How to reproduce

```bash
# 0. provision: 4× RTX 4070 Ti box (vast.ai), ~50 GB persistent /workspace
# 1. on the box:
cd /workspace && git clone https://github.com/<user>/mamba3-hands-on.git
cd mamba3-hands-on
uv venv .venv
uv pip install --python .venv/bin/python torch \
    --index-url https://download.pytorch.org/whl/cu121
uv pip install --python .venv/bin/python transformers accelerate \
    sentencepiece numpy pyyaml

# 2. data:
uv run python make_bilingual_corpus.py     # data/bilingual.txt
mkdir -p data
tmux new -d -s teacher \
  "CUDA_VISIBLE_DEVICES=3 .venv/bin/python jepa/make_teacher_thoughts.py \
     --target-mb 80 --out data/teacher_thoughts \
     --device cuda --dtype bfloat16"
# wait until data/teacher_thoughts.bin reaches 5+ MB (~10 minutes)

# 3. four parallel trainers (one per GPU). Full lines in DEPLOYMENT.md.
mkdir -p runs checkpoints
# (see DEPLOYMENT.md "Phase 2 — four parallel trainers")

# 4. dashboard at http://localhost:8090 (after SSH tunnel):
tmux new -d -s dashboard \
  "CUDA_VISIBLE_DEVICES=3 .venv/bin/python jepa/eval_daemon.py \
     --serve --port 8090 --device cuda:0 \
     --runs $PWD/runs/jepa_cortex/gpu0-ref,$PWD/runs/jepa_cortex/gpu1-lowjepa,..."
```
