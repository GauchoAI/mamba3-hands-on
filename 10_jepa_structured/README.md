# Chapter 10 — JEPA structured-data

**Status:** **active**. The current daily-driver experiment.

# jepa_structured_data — structure-in-data for JEPA-Cortex

Sister experiment to `jepa/`. Same student (1M-param byte-level Mamba-3 with
Cortex residual primitives), same losses (byte CE + JEPA latent regression +
Cramér-Wold isotropy + primitive aux). What changes is **the data**.

## Why

`jepa/` runs (commit `e70187d`) showed JEPA-on regularizes and JEPA-off
mode-collapses — confirming that denser-than-next-byte supervision matters.
But the JEPA target is itself derived from a teacher reading raw OpenSubtitles
+ bilingual text. The student inherits whatever structure the *teacher's
hidden states* already encode; nothing in our pipeline puts structure into
the data we feed the teacher.

The literature converges on one claim: **structure must enter somewhere — in
the architecture, in the loss, or in the data**. JEPA-Cortex has it in two
of the three. This experiment adds the third.

| Source | Lesson |
|---|---|
| [Hamiltonian NNs (Greydanus et al., 2019)](https://arxiv.org/abs/1906.01563) | Parameterize the conserved quantity as a coordinate; tiny model, exact extrapolation. Same move as our `CounterPrimitive`. |
| [I-JEPA (Assran et al., CVPR 2023)](https://arxiv.org/abs/2301.08243) | Predict latents, not pixels — the target lives in semantic space, not surface space. Our `jepa_loss` is the byte-stream version. |
| [LeCun, A Path Towards Autonomous Machine Intelligence (2022)](https://openreview.net/pdf?id=BZ5a1r-kVsf) | JEPA as the predictor inside a world model with planning. Frames the long-term direction. |
| [Faith and Fate (Dziri et al., NeurIPS 2023)](https://arxiv.org/abs/2305.18654) | Transformers reduce compositional reasoning to "linearized subgraph matching"; errors compound exponentially with depth. |
| [The Reversal Curse (Berglund et al., ICLR 2024)](https://arxiv.org/abs/2309.12288) | "A is B" trained → "B is A" not learned. Token geometry ≠ semantic geometry. |
| [Textbooks Are All You Need / phi-1 (Gunasekar et al., 2023)](https://arxiv.org/abs/2306.11644) | 1.3B model, 7B textbook-quality + synthetic-exercise tokens, 50.6% HumanEval. Data quality > scale at small budgets. |

The thread connecting them: surface tokens do not carry the geometry of
meaning. A pendulum's pixels carry physics for free; a sentence's bytes do
not carry truth conditions or equivalence classes for free. Either we
install structure in the architecture (Cortex primitives — done), or we
manufacture it in the data (this experiment).

## What

Three additive moves, each runnable independently, each producing a separate
ablation against the current `jepa/` baseline.

### Move 1 — Paraphrase pairs (cheapest; attacks token-geometry directly)

Build a paraphrase-pair corpus and modify the JEPA loss to enforce that
*the student's intent embedding is invariant under paraphrase*.

- **Data**: for each teacher response `r`, generate `paraphrase(r)` (same
  language) and `translate(r)` (cross-lingual). We already have bilingual
  data; this just adds the alignment.
- **Loss change**: existing `jepa_loss` predicts teacher hidden at byte t+16.
  Add a term: `||intent(prompt) - intent(paraphrase(prompt))||²`. Optional
  stronger version: predict the *intersection* of the two teachers' hiddens.
- **What it tests**: does explicit paraphrase invariance reduce the
  reversal-curse-style failures (Berglund 2023) on a small held-out probe?

### Move 2 — Textbook-style synthetic core (highest leverage; phi-1 lesson)

Replace OpenSubtitles as the JEPA-supervised core with Qwen-generated
"textbook-quality" Q&A: one concept per example, explicit reasoning chain,
worked solution.

- **Data**: pipeline mirrors `jepa/make_teacher_thoughts.py` but with a
  curated prompt template that produces (concept, question, step-by-step
  solution, paraphrase of solution). Target ~10–50 MB to start (an order
  of magnitude smaller than current OpenSubtitles, intentionally).
- **Mix**: textbook = JEPA-supervised primary; OpenSubtitles + bilingual.txt
  kept as fluency-maintenance side dish at low weight (e.g. 20% of batches).
- **What it tests**: does data quality at fixed compute budget beat the
  current setup on (a) byte CE on held-out bilingual, (b) reasoning probes
  (small arithmetic, simple logic), (c) sample quality?

### Move 3 — HNN-style primitive expansion (deepest; but smallest data lift)

Promote three more conserved-quantity primitives from the Lego library into
the Cortex residual stream, alongside `CounterPrimitive`:

- **`StackDepthPrimitive`** — Hanoi-style nested-state depth; conserved
  under matched push/pop pairs.
- **`ParityPrimitive`** — boolean parity over a sequence; flips on each
  qualifying byte.
- **`SortednessPrimitive`** — running monotonicity indicator over a
  numeric span.

Each is a scalar coordinate in the residual, like the existing counter,
parameterized so the LM learns *when to project onto it* rather than having
to invent the quantity from token statistics. Same hard-gate inference
trick that gave 16× OOD generalization for counting.

- **What it tests**: does broader primitive coverage extend OOD
  generalization to compositional probes (e.g. balanced-bracket prediction,
  parity of a subsequence) without re-training the LM?

## How (sketch)

```
experiments/jepa_structured_data/
├── README.md                 ← this file
├── make_paraphrase_corpus.py ← Move 1 data generator
├── make_textbook_corpus.py   ← Move 2 data generator
├── primitives_extra.py       ← Move 3: StackDepth/Parity/Sortedness
├── arch.py                   ← extends jepa/arch.py with paraphrase loss
├── train.py                  ← copies jepa/train.py, adds three switches
│                                 --paraphrase {off|on}
│                                 --data {opensubs|textbook|mixed}
│                                 --primitives {counter|counter+stack+parity+sort}
└── eval/
    ├── reversal_probe.py     ← measures Berglund-style reversal accuracy
    ├── paraphrase_probe.py   ← P(answer|Q) vs P(answer|paraphrase(Q))
    └── compositional_probe.py ← parity, balanced brackets, monotone-runs
```

Following the `jepa/` convention: this folder ships **its own copy** of
`cortex_counting.py`, `mamba3_minimal.py`, and the SSM scan helpers, so
edits don't leak into the top-level baseline. Refresh by hand if upstream
fixes need to land.

## Phased plan

| Phase | Move | Effort | Compute | Decision gate |
|---|---|---|---|---|
| **P0** | Scaffold folder + copy jepa/ files | 1–2h | none | — |
| **P1** | Move 1 (paraphrase) | 1 day | M4 Pro overnight | Does paraphrase-probe accuracy beat baseline by ≥5 pts? |
| **P2** | Move 2 (textbook corpus) | 2–3 days | teacher-gen on cluster, train on M4 | Does held-out byte CE drop ≥0.05 vs baseline at same step count? |
| **P3** | Move 3 (primitive expansion) | 1–2 days | M4 Pro | Do compositional probes hit ≥80% at OOD depths the baseline fails on? |
| **P4** | Combined run (1+2+3) | 1 day | full overnight | Synthesis findings entry. |

Each phase is independently shippable as its own findings entry; we only
combine after all three pass their gates.

## Decisions I need from you before P0

1. **Folder location confirmed?** I put it at `experiments/jepa_structured_data/`
   matching the recent `experiments/harness_3stage_mamba3/` convention.
   Alternative: sister to `jepa/` at top level.
2. **Sister-fork vs in-place?** Mirroring `jepa/`'s "ship your own copy"
   convention is safest but doubles file count. OK to proceed that way?
3. **Teacher model**: keep Qwen-2.5-1.5B (matches `jepa/`) or upgrade for
   textbook generation? Stronger teacher → cleaner synthetic data, but
   needs more VRAM and changes the JEPA target dimension.
4. **Compute target**: M4 Pro + MPS only (per recent handoff), or is the
   vast.ai cluster back? Phase 2 teacher-gen wants ≥16 GB VRAM somewhere.
5. **Phase ordering**: I recommend P1 → P2 → P3 (cheapest first, biggest-
   payoff middle, deepest last). Want a different order?
6. **Synthetic-data budget**: phi-1 used 1B tokens; for a 1M-param student,
   I'd start at 10–50 MB (~10–50M bytes). Confirm or override.
7. **Eval set source**: build the reversal/paraphrase/compositional probes
   from scratch in `eval/`, or reuse parts of `evals/` if there's anything
   close already? (I haven't audited that yet.)
8. **Primitive priority**: of `StackDepth`, `Parity`, `Sortedness`, which
   would you like first? My pick is `Parity` (simplest, sharpest probe).

## What I'll need access to during the run

- **Read**: `jepa/`, `data/bilingual.txt`, `data/teacher_thoughts.{bin,idx}`,
  the Lego library under `seeds/` or `generators/` (whichever has the step
  functions for stack/parity/sortedness).
- **Write**: only inside `experiments/jepa_structured_data/` and a new
  `data/structured/` subdir for the synthetic corpora.
- **Compute**: one M4 Pro overnight slot per phase, plus one teacher-gen
  pass for P2 (~2 hours on 4070 Ti or ~6–8 hours on MPS).
