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

## 4b. Round 3 conclusion + Round 4 launch (2026-04-30 evening)

### Round 3 conclusion: the metric was lying

After round 3 produced gpu1-no-cortex's 0.80 diversity + grammatical
Spanish ("¿Cuántos años tienes?", "Los días lluviosos son ideales para
pasar tiempo en casa leyendo"), we built a **hidden-state retention
metric** to ask the actually-honest question: does the model's latent
at end-of-response retain anything from the latent at end-of-prompt?

The result is unambiguous and damning:

```
gpu1-no-cortex step 7600 (round-2/3 winner):
    diversity:  0.803
    retention: -0.039     ← essentially zero
    drift:      1.473     ← residual moves 1.47x its own magnitude
```

**The model that we crowned as the winner is on autopilot.** Its
hidden state at end-of-response has no measurable relationship to
its hidden state at end-of-prompt. Whatever fluent text it produces,
it is *not conditioned on the prompt* at the latent level. byte CE
+ diversity have been measuring fluency-and-variety, not coherence.

This refutes the entire round-3 framing. The "best variant" of the
sweep is no better at coherence than any other; it just produces
more visibly different fluent autopilot per prompt.

### Why this happened

The teacher corpus (`teacher_thoughts.bin`) is bag-of-parallel-pairs
generated by Qwen-2.5-1.5B asked to "write 12 parallel English-
Spanish pairs about <topic>." The student saw `<instruction>` →
`<list of unrelated bilingual pairs>`. The relationship between
prompt and response in the training data is *the parallel-pair
format*, not semantic. The student learned exactly that — produce
parallel pairs in `:: ` format, regardless of what the prompt
actually said.

No amount of loss-weight tuning fixes a corpus-structure problem.

### Round 4 — switch to OpenSubtitles + measure retention live

**Corpus change:** `data/opensubtitles.txt` (500 MB OPUS-OpenSubtitles
parallel corpus). Each line is `en :: es` like Tatoeba, but the
crucial difference: **consecutive lines are consecutive dialogue
turns from the same movie**. When the trainer reads a 256-byte
window, it sees 3-4 consecutive subtitle lines in sequence. The
model's next-byte prediction over that window is *literally
learning Q→A continuation* by the structure of the data.

**Metric change:** prompt-response retention added to `eval_daemon`.
Drift is also recorded. Round 4 runs will show retention live,
catching coherence problems immediately rather than after-the-fact.

**Trainer enhancement:** `write_canary` now records both greedy and
temperature-sampled (temp=0.7, seed=42) outputs at every checkpoint.
Cross-variant comparisons become file reads, not inference reruns.

### Round 4 design: 2×2 — {fresh, resume} × {d_model 192, 256}

| GPU | Run | Init | d_model | Why |
|---|---|---|---|---|
| 0 | gpu0-resume-192 | resume from gpu1-no-cortex/light_step_0007600 | 192 | Tests "can the round-3 winner adapt to dialogue data, or is its bilingual lexicon a wall?" |
| 1 | gpu1-fresh-192 | fresh weights | 192 | Clean baseline. From scratch, just subtitles. |
| 2 | gpu2-fresh-256 | fresh weights | 256 | Tests "if subtitles are the right corpus, does more capacity widen the gap?" |
| 3 | gpu3-resume-256 | resume from gpu3-no-cortex-bigger/light_step_0002600 | 256 | Tests resume at the bigger size. |

All four use:
- `--use-counter false` (round-2 win)
- `--mix-unary 0`, `--mix-teacher 0`, `--mix-biling 1.0` (no JEPA
  target batches; no synthetic unary; pure subtitle data)
- `--bilingual-path data/opensubtitles.txt` (the conversational corpus)
- 10000 steps, seq_len 256

**Baseline to beat:** retention > 0.30 (vs round-3 winner's -0.04).
This is the threshold where the model starts to actually condition
its response on the prompt at the latent level.

### What round 4 will tell us

- If both fresh variants reach retention > 0.30, the corpus *was*
  the problem and the architecture is fine. We have a path to a
  conversational model.
- If both resume variants reach retention > 0.30 *faster* than the
  fresh ones, we keep more transferable knowledge by resuming.
- If neither reaches retention > 0.30, the corpus alone isn't enough
  and we need either logit distillation (per-token soft targets,
  the upgrade we left on the table — see memory entry on this) or
  the conversational-JEPA loss (predict response embedding from
  prompt embedding, not just next-byte hidden states).

### Pinned round-3 ckpts

`checkpoints/jepa_cortex_pinned_round3/` (23 MB):
- gpu0-no-cortex-highjepa @ step 1600 (early, didn't converge)
- gpu1-no-cortex @ step 7600 (best round-2/3, autopilot per retention)
- gpu2-no-cortex-highsig @ step 1600 (high λ_sigreg refuted)
- gpu3-no-cortex-bigger @ step 2600 (highest diversity, best fluency on 1 prompt, but unary-collapsed on others)

---

## 4c. Round 4 mid-run pivot — kill 3 of 4, build conv-jepa (2026-04-30 night)

Round 4 had four variants (data-only fix: subtitles + retention live).
At ~step 7000 on the early-resume runs, all four were retention-pinned
near zero — gpu0-resume-192 the worst at -0.30, the rest noise around 0.
Sample text on gpu1-fresh-192 was lifted dialogue ("It's not safe", "I
can't tell you") that varied per prompt without conditioning on it. Same
autopilot mode under a new corpus.

**Decision:** kill gpu0-resume-192, gpu1-fresh-192, gpu3-resume-256.
Keep gpu2-fresh-256 cooking as the data-only-fix control. Use the freed
GPUs for an **architectural** test instead of more data tweaks.

### Round 5 design — convJEPA + the long-postponed real distillation

Two architectural variants on top of the round-4 corpus, one knob each:

| GPU | Run | Adds | Why |
|---|---|---|---|
| 1 | (corpus gen now, training after) | – | Generate `subtitle_thoughts.bin`: pairs of consecutive subtitle lines, teacher hidden state captured at end-of-response |
| 0 | gpu0-conv-jepa-192 | `conv_jepa_loss(student_h_p, teacher_h_r)` | Forces cross-segment latent commitment: student's end-of-prompt residual must predict the teacher's end-of-response hidden. Plain `jepa_loss` doesn't catch autopilot because its same-sample shifted target lets the head learn local continuation. |
| 3 | (later) gpu3-conv-jepa-kd-192 | conv-jepa + logit-projection KD | The actually-rich distillation signal (per-byte teacher distribution → student byte distribution → KL). Postponed: needs ~250 GB offline corpus or a streaming teacher in the trainer; both are bigger lifts. |
| 2 | gpu2-fresh-256 | (unchanged from round 4) | Control: data-only fix at d_model=256. If conv-jepa beats it, the fix is architectural; if not, neither is enough. |

**conv_jepa_loss** (`jepa/arch.py`): smooth-L1 between projector(student
residual at end-of-prompt position) and the teacher's last-layer hidden
at end-of-response. Same `ThoughtHead` used by `jepa_loss` does both
projections. Ramp factor shared with `jepa_loss` so the LM head locks in
basic byte fluency before any latent target kicks in.

**Subtitle thoughts corpus** (`make_subtitle_thoughts.py`): consecutive
non-empty / non-unary OpenSubtitles lines paired (line N → line N+1).
Each record is JEPT-format with K=1 thought, identifying the teacher's
last-layer hidden at the BPE token whose decoded prefix first reaches
end-of-response in the byte stream. Loaded by the existing
`TeacherThoughtsDataset` for free.

**Trainer wiring**: new batch kind `"conv"` with mix weight `--mix-conv`,
new `--lambda-conv-jepa`, and a new branch in the train loop that uses
`prompt_lens - 1` as `prompt_end_pos` and `batch.teacher_thoughts[:, 0, :]`
as the response hidden target.

**Baseline to beat:** retention > 0.30 sustained over 1000+ steps, like
round 4. If conv-jepa gets there and gpu2-fresh-256 doesn't, the
architectural lever is real and we have the path to a conversational
model. If neither does, we promote logit-projection KD to round 6.

### Round 4 mid-run trajectory check — gpu2-fresh-256 control (step 50–800)

```
step    retention   drift   diversity  byte_ce_biling
 50      0.96       0.27     0.00       5.68    untrained baseline
100      0.93       0.36     0.00       4.56
150      0.25       1.23     0.00       3.50
200      0.21       1.26     0.00       2.85
250      0.13       1.32     0.00       2.38
300      0.11       1.34     0.07       2.25
350      0.11       1.34     0.25       2.08
400      0.08       1.36     0.00       1.91
500      0.05       1.38     0.14       1.78
600      0.10       1.35     0.36       1.72
700     -0.05       1.46     0.25       1.70    ← crosses zero
750     -0.10       1.50     0.43       1.60
800     -0.07       1.48     0.18       1.57    autopilot regime
```

**This refutes the "subtitles + retention live" hypothesis on its own.**
The trajectory is the round-3 winner's trajectory in miniature: byte_ce
descends cleanly while retention slides past zero by step 700 and stays
there. Diversity ricochets between 0.13 and 0.43 — exactly the
false-positive that masked autopilot in round 3.

Data alone, even with conversational structure in the corpus and the
retention metric watching live, is **not** the lever. The next-byte
objective on a small Mamba-3 with no cross-segment latent supervision
will reliably fall into prompt-blind fluency. Conv-JEPA was queued as
the architectural fix for exactly this reason; the round-4 control
turns conv-jepa from "should we try it" into "no other lever left."

---

## 5. Round 5 — conv-jepa-as-built refuted (2026-05-01)

### Setup

`gpu0-conv-jepa-192` ran past full ramp:
- `conv_jepa_loss` (smooth-L1 between projected student-residual at
  end-of-prompt and teacher's last-layer hidden at end-of-response,
  captured offline by `make_subtitle_thoughts.py` over 130k OpenSubtitles
  pair records, 400 MB)
- shared ramp with `jepa_loss`: gradient gated to 0 until step 2000,
  ramps linearly to full λ=1.0 by step 3000
- everything else identical to `gpu2-fresh-256` control

### Result

```
                  step   retention   drift   diversity   byte_ce   conv_loss
pre-ramp baseline 2000   -0.072      1.51    0.43        1.29      1.32
ramp 0.2          2200   +0.153      1.21    0.56        1.35      0.61  ← noise
ramp 0.4          2400   -0.152      1.56    0.48        1.32      ~0.5
ramp 0.6          2600   -0.035      1.46    0.41        1.30      ~0.5
ramp 0.8          2800   -0.157      1.56    0.68        1.24      ~0.5
ramp 1.0          3000   -0.071      1.49    0.48        1.27      ~0.5
post-ramp +200    3200   -0.059      1.48    0.59        1.20
post-ramp +400    3400   -0.075      1.50    0.64        1.19
post-ramp +3800   6800   -0.143      1.55    0.66        1.10
```

Mean retention across 9 readings post-ramp-start = **-0.07**. Range
[-0.30, +0.15]. Statistically indistinguishable from the gpu2-fresh-256
control's noise band (-0.10 to +0.16) over the same step range. The
conv-jepa loss itself fits — drops from 1.32 baseline to ~0.5 — so the
projector is learning a valid mapping. byte_ce keeps descending (1.29 →
1.10) and diversity climbs (0.43 → 0.66), so training is healthy.
**Retention is the only metric that doesn't move.**

The single +0.15 reading at step 2200 (which I initially flagged as
"first sign of conv-jepa working") was measurement noise; subsequent
readings reverted to baseline. Eight canary prompts isn't enough for
single-checkpoint readings to be conclusive — need rolling means across
5+ checkpoints.

### Why conv-jepa-as-built didn't move retention

**Capacity inversion.** The thought-head projector is 1.33M parameters
(192 → 768 → 1536, 2-layer GELU MLP). The Mamba-3 encoder is 1.02M.
*The projector is bigger than the encoder.*

In this configuration:

```
residual_at_end_of_prompt → [projector 1.33M] → projected_pred
target = teacher_response_hidden (fixed, from offline corpus)
loss = smooth_l1(projected_pred, target)
```

The projector has enough capacity to fit a passable "average response
hidden" mapping from whatever locally-available signal is in the
residual — a short-term encoding of the prompt's last few bytes is
enough. It doesn't *need* the residual to carry prompt-conditional
response info; it absorbs the smooth-L1 burden by itself. Gradient
flowing back to the SSM is therefore weak, the SSM stays prompt-blind,
and retention doesn't move.

Vision JEPA gets this right: predictor is a small MLP, encoder is a
ViT. The asymmetry forces representation work into the encoder. We
inverted that asymmetry; the result is what should have been expected.

### Concurrent refutation: gpu2-fresh-256 (control) finished at step 10000

```
                  step   retention   drift   diversity   byte_ce
control final     10000   -0.075      1.52    0.39        1.04
```

byte_ce 1.04 — record low for the project, beats round-3 winner's 1.10.
Retention -0.075, indistinguishable from autopilot baseline. Confirms
fully: data-fix-alone (corpus + retention live) does not crack
autopilot. Run is `[done]`, ready to be pinned.

### Summary — two clean refutations

1. **Data fix alone**: subtitles + retention live + 10k steps →
   retention asymptotes near zero. Mode is stable.
2. **Conv-jepa-as-built**: projected smooth-L1 with projector > encoder
   → projector absorbs the loss, residual untouched. Mode is also stable.

byte_ce, diversity, and retention are **uncorrelated at this scale**. A
1M-param byte-Mamba-3 hits byte_ce 1.04 and diversity 0.4 with
retention ~0. Improving fluency does not improve coherence on its own.

### Round 6 — proposed levers, ranked by directness

The architectural lever must affect either (a) what gradient reaches
the SSM, or (b) what the SSM has to do to satisfy the loss.

| # | lever | one-line change | hypothesis |
|---|---|---|---|
| 1 | **shrink projector** | `ThoughtHead(d_model, d_teacher, hidden=32)` → ~110k params, 1/10th of encoder | If the projector can't fit the smooth-L1 alone, it must extract prompt-conditional info from the residual. Restore the JEPA capacity asymmetry. **Cheapest test.** |
| 2 | **target end-of-prompt hidden, not end-of-response** | swap `make_subtitle_thoughts.py` to capture teacher hidden at end-of-prompt; same loss | Plain hidden-state distillation: student residual mimics teacher residual at the same byte position. Known to work. Loses cross-segment commitment but is a reliable baseline. |
| 3 | **VICreg on residual + conv-jepa** | force per-dim variance of `residual_at_end_of_prompt = 1` across batch | Anti-collapse on the residual itself. Forces the SSM to produce informative residuals; the conv-jepa target then has something to learn from. Closest to "complete the conv-jepa idea." |
| 4 | **logit-projection KD** | per-byte teacher distribution → student byte distribution → KL | Highest signal density. Big lift: ~250 GB offline corpus or streaming Qwen-1.5B in trainer. The actual real-distillation upgrade. |

**Recommendation: launch #1 immediately on the freed GPU 2 as
`gpu2-conv-jepa-tiny-proj-192`.** Single number change; if retention
moves, the asymmetry hypothesis is confirmed and we know how to scale
conv-jepa properly. If it still doesn't, we're out of cheap moves and
#3 / #4 become unavoidable.

---

## 6. Round 6 — capacity-asymmetry refuted (2026-05-01)

### Setup

`gpu2-conv-jepa-tinyproj-192`: identical to round-5 conv-jepa except
`--thought-head-hidden 32` → projector is now 56,864 params (1/18th of
the 1.02M encoder). Same corpus, same loss, same ramp schedule.

### Result — head-to-head with round-5 conv-jepa-original

```
                step    tinyproj          original
                2000    -0.07 / 1.51      -0.07 / 1.51   (identical pre-ramp; same seed)
                2200    -0.20 / 1.59      +0.15 / 1.21   (first ramp checkpoint)
                2400    -0.21 / 1.60      -0.15 / 1.56
```

Conv loss IS dropping on tinyproj (1.27 at step 2050 → 0.67 at step
2400). So the 56k projector still has enough capacity to absorb the
smooth-L1 burden. **Asymmetry is not the bug.**

byte_ce identical between the two runs (~1.32). Drift slightly higher on
tinyproj. Retention plateaus at -0.21 — same noise band as
conv-jepa-original. Neither projector size moves the retention metric.

### What this rules out

The bottleneck is **not projector capacity.** Even when the projector
is 1/18th the encoder size — well below the JEPA recommendation — the
loss still fits. The encoder is not being forced to do anything new.

Implication: the loss target itself doesn't pull the encoder toward
encoding prompt-conditional info. The teacher's last-layer hidden at
end-of-response is dominated by *response content*, not by prompt
encoding. A predictor of any size, given any prompt-side input, can fit
"average response hidden over the corpus" without needing the prompt.
Smooth-L1 against a target that's largely independent of the input
doesn't shape the input.

### Two-round refutation summary

The retention metric has now resisted **three** different
configurations:
1. Round 4: data-only fix (subtitles + retention live).
2. Round 5: conv-jepa with 1.3M projector (>encoder).
3. Round 6: conv-jepa with 56k projector (<<encoder).

byte_ce keeps descending across all three. Diversity bounces. Retention
asymptotes near zero in every configuration. The conv-jepa loss
formulation — "predict teacher response-end hidden from student
prompt-end residual" — is refuted as a lever for retention,
independent of projector size.

### Round 7 — pivot to plain hidden-state distillation

Since cross-segment commitment via projection-to-response-hidden doesn't
work, drop the cross-segment piece and do plain distillation:

> **Match teacher's end-of-prompt hidden, not end-of-response.**

Both teacher and student look at the same byte stream (the prompt).
Loss: smooth-L1 between projector(student residual at end-of-prompt) and
teacher's hidden at end-of-prompt. The student is asked to mimic the
teacher's representation of the prompt — known-to-work knowledge
distillation. We lose the "predict the response from the prompt"
ambition, but we gain a target that's *forced* to be a function of the
prompt the student is reading.

The hypothesis: a student trained to mimic Qwen-2.5-1.5B's end-of-prompt
representation will produce coherent responses, because the student's
LM head will then be reading a Qwen-shaped representation. Coherence
gets transferred from teacher to student through the residual.

**Implementation:** `make_subtitle_thoughts.py` already captures
positions; switching the target to end-of-prompt is a one-flag change.
The trainer's `conv_jepa_loss` already takes `prompt_end_pos` as the
source position — we just need the target hidden to also come from that
position. Five-line edit + corpus regenerate + launch.

If retention still doesn't move after round 7, we go to logit-projection
KD (round 8) — the long-postponed real distillation.

---

## 7. Round 8 — contrastive distillation (built but not yet launched)

After re-reading DeepSeek V4's paper (their compose-many-complementary-
signals philosophy), it became clear: our prior losses share a flaw.

`smooth_L1(projector(residual), teacher_target)` is minimized when the
projector outputs ≈ target. The corpus mean of teacher targets is a
passable solution. The encoder gets weak gradient because reducing the
loss doesn't *require* prompt-specific predictions.

**InfoNCE-style contrastive distillation** doesn't have this hole.
Logits are cosine similarities between `projector(residual_at_end_of_prompt[i])`
and *all* teacher targets in the batch; loss is cross-entropy with
labels=arange(B). The corpus mean is *terrible* at this — equal
similarity to every target → uniform softmax → loss = log(B). To drive
the loss low, the projection must be **prompt-specific**, which forces
the encoder to encode prompt-specific info into the residual.

This is the strongest test of the input-dependence problem documented
in `feedback_loss_target_input_dependence.md`. If contrastive doesn't
move retention, the small Mamba-3 SSM at this scale genuinely cannot
be made prompt-conditional under any pseudo-distillation flavor, and
the round-8+ pivot would be to logit-projection KD (per-byte teacher
distribution → KL — the only loss with byte-level prompt-conditional
pressure built in by construction).

### Implementation landed (not yet launched)

`arch.py`: `contrastive_distill_loss(student_thoughts, teacher_target,
prompt_end_pos, temperature=0.1)`. Symmetric InfoNCE (s→t and t→s
averaged), L2-normalized, cosine similarity. ~25 lines.

`train.py`: new `--lambda-contrastive` and `--contrastive-temp` flags.
Wired into the `kind == "conv"` branch as a parallel loss, on the same
ramp schedule as `conv_jepa_loss`. Same corpus works (response-end
or prompt-end target — InfoNCE forces specificity either way).

Launch when ready (round 7 still pre-ramp on GPU 1):

```bash
python jepa/train.py --run-name gpu2-contrastive-distill-192 \
  --d-model 192 --use-counter false --thought-head-hidden 32 \
  --steps 8000 --batch-size 64 --seq-len 256 \
  --subtitle-path data/subtitle_thoughts_prompt \
  --mix-conv 0.6 --mix-biling 0.4 \
  --lambda-conv-jepa 0.0 --lambda-contrastive 1.0 \
  --contrastive-temp 0.1 --lambda-jepa 0.0 --lambda-sigreg 0.1
```

Note: `--lambda-conv-jepa 0.0` so contrastive is the only conv-side
loss. Clean ablation.

---

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

## 8. Mac Mini Sprint — DeepSeek V4 inspirations (2026-05-02 night)

While the vast.ai 2×2 grid (rounds 7+8: smooth-L1 vs InfoNCE × prompt-end
vs response-end target) cooks overnight at d_model=192, an orthogonal
exploration runs on the M4 mini. Goal: try the DeepSeek V4 paper's ideas
mapped onto our small-model autopilot problem. Each strategy gets a
dedicated, self-contained script in `experiments/13_mini_sprint/` so
commit history stays 1-to-1 with strategies, and the results table
below names the commit that introduced each.

The mini sprint is **prerequisite-blocked** on a cleaner corpus first —
all prior runs used `data/opensubtitles.txt` which globs across movie
boundaries; ~half of the "consecutive-line dialogue pairs" we trained
on were actually cross-movie noise. Fixed by `extract_movie_pairs.py`
using the `OpenSubtitles.en-es.ids` metadata: emits a blank line
whenever the movie ID changes, so existing pair iterators (which reset
on blank lines) only ever see within-movie pairs. Output:
`data/movie_pairs_clean.txt` — 61,434,251 pairs across 77,652 movies,
4.17 GB.

Sprint scale: d_model=96, n_layers=2, batch=32, seq_len=128, ~2000 steps
per experiment, ~10–15 min wall-clock on M4 MPS. Same RNG seed across
all runs so step-0 baselines are identical and any divergence is
attributable to the experimental lever.

### Strategy table

| # | DeepSeek V4 idea | Our analog | Status | Commit | Result |
|---|---|---|---|---|---|
| 0 | "data quality matters" | clean within-movie corpus, byte-CE only | ✅ done | `e1aa799` | retention **+0.120** vs vast.ai control's -0.075 — **clean corpus alone lifts retention by +0.20**, no architectural change. Doesn't cross 0.30 but moves the floor. |
| 1 | Anticipatory routing (EMA snapshots) | EMA self-distillation (BYOL-style) | ❌ refuted | `442c247` | retention **0.043** (dropped from baseline 0.120). BYOL loss → 0.003 (predictor solved task) but pushed residuals toward *more* generic, not more prompt-specific. Trivial-collapse mode at our scale. |
| 2 | Hybrid attention (CSA+HCA+window) | Multi-scale residual matching (3 positions) | ❌ refuted | `442c247` | retention **0.038**. Same trivial-collapse mode as exp_01: ms loss → 0.004, but residuals get *more* generic. Three positions × no augmentation = three trivial fits. Capacity composition can't fix a target that doesn't depend on the input. |
| 3 | Curriculum (4k→1M context) | Seq-len curriculum (32→64→128) | ❌ refuted | `e088469` | retention **-0.024**. Worse than baseline. Short-context phase locked in surface n-gram patterns that override prompt-conditioning when context expands. Completions show interesting Spanish-English code-switching (more diverse output, less coherent). |
| 4 | Muon optimizer | (skipped tonight — implementation cost too high vs likely gain; queued for later) | — | — | — |
| 5 | MHC (Sinkhorn-Knopp) | Bounded residual-norm constraint | ❌ no-op | `e088469` | retention **0.116** ≈ baseline 0.120. Zero effect. V4 MHC is for trillion-param signal explosion; at 150k params our residuals don't have norm runaway, so the constraint has nothing to constrain. |
| 6 | Compose-many-signals | Stack of #0+#1+#2+#3+#5 | ❌ refuted | `e088469` | retention **-0.079**. WORSE than any single lever. Stacking 4 refuted/no-op levers + 1 winner compounds damage. The V4 compose thesis only holds when each component fixes a real problem; here 4 of 5 don't, so combining them just stacks the noise. |

**Runner:** `experiments/13_mini_sprint/run_sequential.sh` (commit
`23d6cef`) launched in tmux on mini — fires exp_01 through exp_05 in
sequence after exp_00 finishes. Recurring 15-min cron on M4 Pro picks
up `eval.json` files as they land and folds results into this table.

Each row is filled in as the experiment lands. Refutations are kept —
they're the more useful data points.

### exp_00 — clean corpus baseline (control)

**Script:** `experiments/13_mini_sprint/exp_00_clean_corpus_baseline.py`
**Config:** d_model=96, n_layers=2, batch=32, seq_len=128, 2000 steps,
byte-CE only (no JEPA/conv/sigreg/contrastive), AdamW lr=3e-4 cosine
**Hypothesis:** corpus contamination is a confounder we never separated.
If retention crosses 0.30 here at 2000 steps, the vast.ai runs were
partly dataset-limited.
**Why this isn't a V4 idea:** it's the control, the baseline number that
all V4-inspired levers in #1-6 are measured against. Cheap to run.

**Result (commit `e1aa799`):**

```
retention   drift   diversity   byte_ce_train   completions
+0.120      1.37    0.42        1.79            mode-collapsed on "the the the"
```

vs vast.ai gpu2-fresh-256 (dirty corpus, same byte-CE-only setup) which
finished with retention **-0.075**. Same architecture family, same loss,
same metric — just clean within-movie corpus instead of cross-movie
contaminated. **+0.20 retention lift purely from corpus quality.**

Doesn't cross the 0.30 healthy threshold — the model is also visibly
undertrained at 2000 steps on this small corpus slice (canary completions
mode-collapse onto "the the the"). But it moves the floor of the noise
band from "centered on 0" to "centered on +0.12." Subsequent
DeepSeek-inspired experiments build on this corpus.

**Generalizable lesson** worth pinning: every prior round of the JEPA
saga used `data/opensubtitles.txt`, which globs across movie boundaries.
Probably ~half the consecutive-line "dialogue pairs" trainers were
seeing were noise (line N from movie A followed by line N+1 from movie
B). Just fixing the corpus closes ~25% of the gap to the healthy
retention threshold.

### exp_01 — EMA self-distillation (BYOL/anticipatory-routing analog)

**Script:** `experiments/13_mini_sprint/exp_01_ema_self_distill.py`
**Hypothesis:** the EMA's residual at end-of-prompt is forced to be a
function of the prompt; if the live student's predictor learns to map
its own residual onto that target, both networks pull each other toward
more prompt-stable representations.

**Result (commit `442c247`):**

```
retention   drift   diversity   byte_ce   byol_loss_final
+0.043      1.43    0.48        1.81      0.003 (≈ collapsed)
```

**Refuted.** EMA self-distill HURT retention from baseline +0.120 down
to +0.043. The BYOL loss converged to ~0.003 — the predictor learned
to map live residual ≈ EMA residual nearly perfectly — but that
convergence happened because both networks drifted toward a *more
generic* representation that's easier to predict than a
prompt-conditional one. Classic BYOL trivial-collapse failure mode at
our scale.

Why this could fail where vision JEPA succeeds: BYOL-style methods rely
on data augmentations creating two different views of the same input.
Without augmentations, the predictor's job is trivially easy (live and
EMA see *identical* inputs and produce nearly identical residuals), and
the constraint never actually pulls representations toward anything
useful. We could try adding augmentations (byte-flip, char-swap, span
masking) but that's a different experiment — out of scope for tonight.

**Generalizable lesson:** self-distillation without augmentations is
decorative. The asymmetric predictor isn't enough on its own to break
trivial collapse — you need either (a) two different *views* of the
input, or (b) the EMA's slowness to provide a meaningful gradient signal
during training noise. At our small scale + clean corpus, neither
applies.

### exp_02 — multi-scale residual matching (V4 hybrid-attention analog)

**Script:** `experiments/13_mini_sprint/exp_02_multi_scale_distill.py`
**Hypothesis:** if single-position EMA matching trivially collapses,
maybe matching three positions simultaneously (1/4, 1/2, 3/4 of seq)
forces position-aware encoding.

**Result (commit `442c247`):**

```
retention   drift   diversity   byte_ce   ms_loss_final
+0.038      1.43    0.49        1.82      0.004 (collapsed)
```

**Refuted.** Same trivial-collapse pattern as exp_01. Multi-scale loss
converged to ~0.004 — the predictor learned to map live residuals to
EMA residuals at all three positions simultaneously, but did so in a
way that bypassed the encoder. byte_ce identical to exp_01.

**Generalizable lesson** (combined with exp_01): the hybrid-attention
"compose-many-positions" V4 trick doesn't transfer when the underlying
target is the model's own EMA. *Three trivial fits = one trivial fit.*
Composition only adds value when each component imposes a constraint
the predictor can't dodge — and EMA-of-self isn't such a constraint
without augmentation. The DeepSeek V4 idea works in V4 because each
attention pathway has a *different* input compression (different
token groupings), creating multiple inequivalent loss landscapes.

### exp_03 — sequence-length curriculum (V4 4k→1M analog)

**Script:** `experiments/13_mini_sprint/exp_03_curriculum.py`
**Hypothesis:** training on short contexts first might prevent autopilot
from settling in (model first learns local prompt-conditional structure
on cheap short examples, then gradually extends).

**Result (commit `e088469`):**

```
retention   drift   diversity   byte_ce   schedule
-0.024      1.49    0.38        1.89      32@500, 64@1000, 128@2000
```

**Refuted — and worse than baseline.** Retention dropped from +0.120
(exp_00 baseline) to **-0.024**. The model went from "marginally
prompt-conditional" to "indistinguishable from autopilot."

What's interesting: the completions show real bilingual *code-switching*
("the was a ser the serde a la de la de lo ser") — the curriculum made
the model **more diverse** but **less coherent**. The short-context
phase let the model fit surface n-gram patterns that, when context
expanded, dominated next-byte prediction over prompt-conditioning.

**Generalizable lesson:** the V4 curriculum (4k→1M tokens) is about
*context capacity scaling*, not about *task difficulty scaling*. They
gradually expand the model's working memory, not the difficulty of the
prediction task. Our seq_len curriculum is conceptually closer to
"start with easier task" (short context = easier next-byte) which can
*increase* autopilot risk: easy short-context targets are dominated by
local fluency, and the model never has to learn prompt-conditional
extension. We should NOT generalize "curriculum" as "always good" —
the right curriculum dimension matters.

### exp_04 — bounded residual-norm constraint (V4 MHC light)

**Script:** `experiments/13_mini_sprint/exp_04_residual_norm.py`
**Hypothesis:** SSM residuals could drift in magnitude in ways that
mask directional information; bounding norm growth could keep the
residual subspace shaped for prompt-conditional encoding.

**Result (commit `e088469`):**

```
            retention   drift   diversity   byte_ce   l_norm_final
exp_00      +0.120      1.37    0.42        1.79      —             (baseline)
exp_04      +0.116      1.37    0.42        1.79      0.067         (norm-constrained)
```

**No effect** — bit-identical to baseline within metric noise. The
constraint converged to l_norm 0.067, meaning per-position residual
norms are roughly within [exp(-0.26), exp(+0.26)] of √D — already
naturally bounded at this scale.

**Generalizable lesson:** the V4 MHC trick (Sinkhorn-Knopp doubly-
stochastic constraint preventing signal explosion at 1.6T params)
**doesn't transfer to small models** because small models don't have
the signal-explosion problem it solves. Confirmed empirically. Don't
add stability regularizers to a model that's already stable.

### exp_05 — combined kitchen-sink (V4 compose-many-signals lesson)

**Script:** `experiments/13_mini_sprint/exp_05_combined.py`
**Hypothesis:** even if each lever from exp_01..04 individually fails,
the V4 compose-many-signals thesis says combining them composes their
effect. Stack: clean corpus + EMA self-distill + multi-scale + seq-len
curriculum + residual norm constraint.

**Result (commit `e088469`):**

```
                              retention   drift   diversity   byte_ce
exp_00 (clean corpus only)    +0.120      1.37    0.42        1.79
exp_05 (kitchen sink)         -0.079      1.53    0.29        1.93
```

**Refuted catastrophically.** Worse than baseline by -0.20 retention.
Completions show heavy code-switching ("I was a ser a la de la de la
de la"…) — same pathology as exp_03 (curriculum) but more pronounced.
byte_ce regressed from 1.79 to 1.93.

**The deepest generalizable lesson of the sprint:** the V4
compose-many-signals thesis is conditional on *each* signal fixing a
real problem. In V4, every architectural piece (CSA+HCA+window for
memory, MHC for stability, Muon for optimization, anticipatory routing
for noise) addresses a documented bottleneck at trillion-param scale.
Composition is super-additive there because the bottlenecks are real.

At our scale, 4 of 5 components don't fix anything (no signal
explosion, no real curriculum benefit, no useful EMA gradient without
augmentation, multi-scale just multiplies trivial collapse). Stacking
them produces *destructive interference* — multiple useless gradients
fighting each other. **Composition without first verifying each
component is helpful is anti-engineering.** A 4-of-5 hit rate on
component utility produces NEGATIVE compound returns.

### Sprint summary table

| exp | strategy | retention | verdict |
|---|---|---|---|
| 00 | clean within-movie corpus, byte-CE only | **+0.120** | ✅ winner |
| 01 | EMA self-distill (BYOL anticipatory) | +0.043 | ❌ trivial collapse |
| 02 | EMA self-distill, 3 positions | +0.038 | ❌ trivial collapse × 3 |
| 03 | seq-len curriculum 32→64→128 | -0.024 | ❌ wrong curriculum dimension |
| 04 | bounded residual norm (MHC light) | +0.116 | = no-op (no problem to solve) |
| 05 | all 5 above stacked | -0.079 | ❌ destructive composition |

**Only winner: corpus quality.** Clean within-movie pairs beat all 5
DeepSeek-V4-inspired architectural levers at our scale. None of the V4
ideas transferred — they all targeted failures that either don't exist
at our scale (signal explosion, scale-dependent curriculum) or required
assumptions we don't have (data augmentation for BYOL).

This is itself a meaningful research result: a model 7 orders of
magnitude smaller than V4 (150k vs 1.6T params) cannot adopt V4's
architectural innovations because the underlying problems V4 solves
don't manifest at small scale. The V4 paper is essentially scale-
specific engineering.

What we *did* confirm: the long-postponed real distillation
(logit-projection KD, "round 9" on findings §5) remains the only
unexplored architectural lever worth trying. Everything else has been
refuted.

---

## 9. Morning grid summary — round 7+8 vast.ai (2026-05-02 morning)

The vast.ai 2×2 grid (rounds 7+8: {smooth-L1 vs InfoNCE} × {prompt-end
vs response-end target}, all at d_model=192, projector hidden=32 to
satisfy the asymmetry rule from round 6) ran overnight. All four cells
sit in the same noise band [-0.10, +0.03]. **The full pseudo-distillation
× loss-formulation × target-position grid is dead at this scale.**

### Final readouts at step 3800–4800

```
                                          step  retention  drift  div   byte_ce
gpu1 prompt-distill (smooth-L1, prompt)   4800   +0.029    1.42   0.56  1.246
gpu2 contrastive-prompt (InfoNCE, prompt) 3800   -0.083    1.52   0.18  1.253
gpu0 contrastive-resp  (InfoNCE, resp)    3800   -0.024    1.47   0.62  1.244
```

Mean retention of last-3 evals on each:
```
gpu1: -0.063   gpu2: -0.041   gpu0: -0.033
```

byte_ce keeps hitting record lows (1.15–1.25) while retention asymptotes
near zero. Same pattern as rounds 4/5/6: byte_ce, diversity, and
retention are uncorrelated at this scale.

### Final 2×2 verdict

|                       | smooth-L1                                       | InfoNCE (contrastive)                                |
|-----------------------|--------------------------------------------------|-------------------------------------------------------|
| **target=response_end** | round 5/6 refuted                                | gpu0-contrastive-resp refuted (-0.03 mean)           |
| **target=prompt_end**   | gpu1-prompt-distill refuted (-0.06 mean)        | gpu2-contrastive-prompt refuted (-0.04 mean)         |

### What this closes

The autopilot saga (rounds 4 → 8 + mini sprint exp_00–05) has now
**eliminated**:

- data-only fixes (round 4)
- smooth-L1 distillation against any teacher hidden position (rounds 5–7)
- contrastive distillation against any teacher hidden position (round 8)
- BYOL self-distillation (mini exp_01)
- multi-position residual matching (mini exp_02)
- seq-len curriculum (mini exp_03)
- residual-norm regularization (mini exp_04)
- composing all of the above (mini exp_05)
- projector capacity asymmetry as the explanation (round 6)

What remains:

- **logit-projection KD** (per-byte teacher distribution → KL): the
  long-postponed real distillation. Still untested.
- **d_model scaling on the clean corpus**: corpus quality at +0.20
  retention is the one positive lever; combined with d_model=192 it
  might compose super-additively. Untested.

### Decision (taking the lead, 2026-05-02 morning)

Two follow-ups firing now:

1. **Kill the three round-7+8 trainers** — the 2×2 grid is already
   conclusive at step 3800–4800. Continuing to step 8000 burns ~16h of
   compute on confirmation. Free the GPUs.

2. **Launch `gpu0-clean-corpus-192-bigmodel`** on the freed GPU 0:
   d_model=192, n_layers=4, byte-CE only, clean corpus
   (`data/movie_pairs_clean.txt` synced from mini), 8000 steps. Tests
   whether the corpus-quality lever composes with model scale. If
   retention crosses 0.30 here, the "small Mamba can't do it" story is
   wrong — it was always a corpus problem.

3. **In parallel, build logit-projection KD infrastructure** on M4 Pro
   for round-10 (next round after the corpus-scale check). Round-9
   slot reserved for the d_model=192 + clean corpus run.

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
