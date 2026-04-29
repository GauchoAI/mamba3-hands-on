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

## 3. Conclusion (will be written when the run stops)

*Pending — to be filled in once we hit step 30000 on all runs or
decide to stop early.*

Open questions to resolve in the conclusion:
- Did `λ_aux=2.0` on gpu3-bigaux unlock a non-zero `count_acc`? If yes,
  the lever is the aux weight. If no, we need a different aux design
  (probably explicit length-target supervision).
- Did the bilingual-vs-zero-jepa gap hold or close at step ~10000?
- Was λ_jepa=0.3 (gpu1) consistently best, or did gpu0 catch up at
  full ramp + more steps?
- What is the JEPA-loss floor? It was 0.25 at step 4000 and still falling.

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
