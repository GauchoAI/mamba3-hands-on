# Session handoff — 2026-04-28 → 29

End-of-session state for the cortex + language + MLX work. Whoever
picks up next: this is the snapshot. `PLAN.md` is the strategic
document and stays canonical; this file is the situation report.

**Branch:** `main` (tracking `origin/main`).

**Update 2026-04-29 (autonomous block):** the in-flight bilingual
training finished. F16 + F17 ran. Findings: PARTIAL composition —
plugin interface holds and LM is not destabilised, but the counter
is off-by-one even in-distribution; full byte-perfect did not
transfer to the language-trained host. See findings.md entry
"Counter primitive on a frozen bilingual LM" for the full read.

A scale=30 diagnostic (3× louder counter signal) is currently
running to test whether the off-by-one is purely signal-magnitude.

---

## TL;DR

- **Infrastructure: complete.** MLX port, bf16 trainer, counter-attach
  script, teacher corpus generator — all written, tested, committed.
- **Experiments: pending.** The headline cortex composition demo is
  blocked on the in-flight bilingual LM (~3h ETA). Distillation A/B
  is blocked on the teacher corpus filling on m4-mini (~24-37h ETA).
- **Two long jobs running**, both nohup'd / backgrounded; neither
  needs babysitting; both produce useful output incrementally.

---

## Running right now

| Process | Where | ETA | Output |
|---|---|---|---|
| PyTorch 10k bilingual LM training | m4-pro MPS, background `bqm427drg` | ~3h to `step_FINAL.pt` | `checkpoints/lm/step_NNNNNN.pt` + `training.log` |
| Teacher corpus generation (Qwen-2.5-1.5B-4bit) | m4-mini Metal, PID 27750, nohup'd | ~24-37h to 20 MB | `~/mamba3-hands-on/data/teacher_corpus.txt` (m4-mini side) |

Status checks:

```bash
# PyTorch training progress
tail -3 /tmp/lm_train.log

# Teacher corpus progress (on m4-mini)
ssh miguel_lemos@192.168.0.170 \
  'cd ~/mamba3-hands-on && wc -l -c data/teacher_corpus.txt && tail -3 teacher.log'
```

---

## What's complete (committed and validated)

| Workstream | Item | Notes |
|---|---|---|
| A — MLX baseline | port + parity + trainer + `mx.compile` scan | parity max-abs-diff 3.58e-7; MLX 1.42× over PyTorch MPS |
| B — bf16 | `--dtype bfloat16` flag in MLX trainer | 2.11× over fp32; **2.87× over PyTorch MPS combined** |
| D — OpenSubtitles prep | `make_opensubtitles_corpus.py`, 500 MB sample built | cache in `data/_opensubtitles_cache/` (gitignored) |
| E12 — Teacher corpus generator | `make_teacher_corpus.py` running on m4-mini | Qwen-2.5-1.5B-4bit via mlx-lm |
| F15 — Counter-attach script | `train_counter_attach.py` smoke-tested | frozen 472,960-param LM + 1,028 trainable counter params |

---

## What's NOT yet run — the actual experiments

These are blocked only on the long jobs above. No code is missing.

### F16 + F17 — the headline cortex composition demo (highest value)

Blocker: bilingual LM finishing (~3h).

```bash
python train_counter_attach.py \
    --lm-ckpt checkpoints/lm/step_FINAL.pt \
    --steps 1000

# F17 (script not yet written): side-by-side demo showing
#   - 'count to fifty:'           (en) → 50 a's, byte-perfect
#   - 'cuenta hasta cincuenta:'   (es) → 50 a's, byte-perfect
# with hard_gates_inference = True
```

Validation gate: counter aux loss converges, bilingual loss does NOT
regress (frozen LM weights are unchanged), at hard-gates inference
the counter emits byte-perfect output at OOD N.

### D11 — train MLX student on OpenSubtitles

Blocker: M4 Pro GPU is busy until PyTorch finishes.

```bash
python train_bilingual_mlx.py \
    --corpus data/opensubtitles.txt \
    --dtype bfloat16 \
    --steps 10000 \
    --ckpt-dir checkpoints/lm_mlx_opensubs
```

Expected: ~4-5h at 1.5s/step bf16. The 500 MB OpenSubtitles corpus
is much richer than Tatoeba (real movie/TV dialogue) — student
should reach significantly better sample quality.

### E14 — distillation A/B run (Path A pseudo-label)

Blocker: teacher corpus filling (~24-37h to 20 MB; usable from
~5 MB at ~7-9h).

```bash
# 1. Sync teacher corpus from m4-mini
rsync -av miguel_lemos@192.168.0.170:~/mamba3-hands-on/data/teacher_corpus.txt data/

# 2. Train student on teacher corpus
python train_bilingual_mlx.py \
    --corpus data/teacher_corpus.txt \
    --dtype bfloat16 \
    --steps 5000 \
    --ckpt-dir checkpoints/lm_mlx_distill

# 3. Compare sample evolution at matched steps vs Tatoeba baseline
diff <(grep -A 9 "step  1000" checkpoints/lm/training.log) \
     <(grep -A 9 "step  1000" checkpoints/lm_mlx_distill/training.log)
```

Honest framing: this is **pseudo-label distillation, not rigorous
KL match.** Tokenizer mismatch (byte-level student vs BPE teacher)
makes direct logit matching infeasible. We're checking whether
teacher-generated text gives the student a quality bump over
Tatoeba/OpenSubtitles — a real question even if non-textbook.

---

## Deferred — real but lower-leverage

- **B7 — Lion optimizer.** Easy 1.3-2× convergence-rate win.
- **B8 — Cluster data-parallel.** 2× wall-clock with both nodes;
  needs LAN gradient-sync code on top of `cluster_dispatch.py`.
- **C9-10 — Chunked SSM scan.** After `mx.compile`, the scan is
  <1% of step time. Chunked scan gives only marginal further gains
  at our shapes. Keep on roadmap; don't prioritize.

---

## Pickup checklist (in priority order)

1. `git log --oneline -15` — read the commit log.
2. `cat PLAN.md` — strategic picture.
3. Check both running processes (commands above).
4. **When `checkpoints/lm/step_FINAL.pt` exists, run F16 + F17**
   (the cortex composition experiment). That's the headline.
5. When `data/teacher_corpus.txt` (m4-mini) is ≥5 MB, sync it and
   start the distillation A/B (E14).
6. Compare both trained students against the Tatoeba baseline.
7. If the cortex composition demo works, write it up as a
   `findings.md` entry. **That's the moment to stop and consolidate.**

---

## Parallel JEPA-Cortex work (already in tree)

Discovered in `jepa/` after sign-off: a substantial parallel experiment
ships with its own copy of `cortex_counting.py` + `mamba3_minimal.py`
to isolate edits, plus `arch.py` (ThoughtHead, jepa_loss, sigreg
Cramér-Wold isotropy regularizer), `make_teacher_thoughts.py`,
`data_loader.py`, `train.py`, `eval_daemon.py`, `talk.py`. Designed
to run on a rented CUDA box with hidden-state-trajectory distillation
from Qwen-2.5-1.5B-Instruct. **Do not duplicate this work** — it's
the rigorous distillation path the README edit was anchoring to.

## Risks and unknowns

- **Counter-attach generalization.** The actual research question:
  can a 1k-param plugin learn to fire on byte 42 (`*`) using a
  language-trained LM's hidden state, when the LM was trained on
  bilingual + 5%-mixin? If yes → train-free composition existence
  proof. If no → we learn the plugin port needs more design. This
  is the single highest-information experiment queued.

- **Pseudo-label distillation marginal value.** Real KL distillation
  is infeasible (tokenizer mismatch). Path A may produce diminishing
  returns vs just using the bigger OpenSubtitles corpus, which is
  already real bilingual dialogue. The teacher run will tell us
  empirically.

- **Bilingual LM ceiling at 473k params + 17.7 MB Tatoeba.** The
  trained LM will not be conversational. The cortex composition
  demo will be "counter works on a barely-coherent LM" rather than
  "LM that talks plus counter." That's still informative for the
  architectural thesis but framing matters in the write-up.

---

## Commit log this session

```
efddb81  PLAN.md: distillation workstream pivot to pseudo-label (Path A)
6d9bbc9  make_teacher_corpus.py: pseudo-label distillation via Qwen-2.5-1.5B
f3c04d2  PLAN.md: linear commit sequence + status
b9d6506  Counter-attach experiment: cortex primitive on a frozen bilingual LM
e91b34b  Add --dtype bfloat16 flag to MLX trainer (2.11x over fp32)
81f9125  mx.compile the SSM scan: MLX now 1.42x faster than PyTorch MPS
46b6070  MLX training loop for CortexLM
7fa151d  MLX port of Mamba-3 + CortexLM, with PyTorch parity test
63fb43b  Bilingual LM training pipeline: corpus prep + trainer
100e7d3  Cortex experiment: counter primitive + plug architecture
```

(Plus two interleaved commits from a parallel thread:
`26d0ff7  Slot-fill renderer` and `17b982a  Pointer-Networks copy`.)

---

## End state

Tree clean. 10 of my commits + 2 yours. Two long jobs running. One
headline experiment queued behind a checkpoint that lands in ~3 hours.
One A/B experiment queued behind a corpus that fills overnight.

Pickup = `git pull && git log --oneline -15 && cat handoff/2026-04-28-mlx-port-and-experiments-queued.md`.
