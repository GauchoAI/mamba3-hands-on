# Implementation plan — small LM that talks via cortex primitives

End state target: a small Mamba-3 byte-level LM that can both
**speak (English + Spanish, conversational register)** and **reason
(via residual-stream primitives)**, runnable on M4 Pro.

This document is the linear sequence of work. Each entry is one
commit boundary, each commit gated by a validation step. Status as
of last update is in brackets `[done]` / `[in flight]` / `[planned]`.

---

## Strategic levers (ranked by leverage on the end-state)

| # | Lever | Speedup | Unlocks |
|---|---|---|---|
| 1 | **MLX port + bf16** | 3-5× wall-clock | Dissolves the F.pad fallback; bf16 halves memory + ~1.7× compute |
| 2 | **Chunked SSM scan** | additional 5-10× | Sequential `for t in range(L)` is the bottleneck after mx.compile; chunked associative scan from Mamba-2 paper makes it parallel |
| 3 | **Pre-tokenized corpus + cluster fan-out** | 2× wall-clock | Data prep stops being on critical path; m4-mini doubles throughput |
| 4 | **Distillation from teacher LM** | 5-10× *convergence rate* + quality at fixed param count | Decides whether "small LM that talks" is reachable on Apple Silicon at all |
| 5 | **Counter-attach to bilingual LM** | qualitative | The cortex composition demo. Tests the train-free composition gate |

Commit #1 + bf16 + mx.compile: **2.87× over PyTorch MPS** — done.

---

## Linear commit sequence

### Workstream A — MLX baseline

| # | Commit | Validation gate | Status |
|---|---|---|---|
| 1 | `mamba3_mlx.py`: MLX port of Mamba-3 + CortexLM + Primitive base | shape-correct forward | **done** (`7fa151d`) |
| 2 | `parity_mlx.py`: PyTorch ↔ MLX parity test | max-abs-diff < 1e-3 | **done** (3.58e-7, `7fa151d`) |
| 3 | `train_bilingual_mlx.py`: MLX training loop | 100-step smoke, non-NaN | **done** (`46b6070`) |
| 4 | `mx.compile` the SSM scan body | parity preserved + benchmark | **done** (1.42× over PyTorch, `81f9125`) |

### Workstream B — MLX speed levers

| # | Commit | Validation gate | Status |
|---|---|---|---|
| 5 | `--dtype bfloat16` flag | smoke run, no NaN, loss converges | **done** (2.11× over fp32, `e91b34b`) |
| 6 | Pre-tokenize + memmap dataloader | step-time benefit | **dropped** (trivial for byte-level) |
| 7 | Lion optimizer + tuned LR | convergence-rate winner | **planned** |
| 8 | Cluster fan-out (m4-pro + m4-mini) | wall-clock 1 vs 2 nodes | **planned** |

### Workstream C — Chunked SSM scan

| # | Commit | Validation gate | Status |
|---|---|---|---|
| 9 | Chunked associative scan (chunk=32 or 64) | parity vs sequential within 5e-4 | **planned** |
| 10 | Drop-in via `--scan chunked` flag, benchmark | tokens/sec + final loss | **planned** |

> Note (after benchmarking): with mx.compile already fused, the scan
> is <1% of step time. Chunked scan likely yields modest gains. Keep
> on roadmap but don't sequence ahead of the headline experiment.

### Workstream D — Real corpus

| # | Commit | Validation gate | Status |
|---|---|---|---|
| 11 | Train MLX model on 500 MB OpenSubtitles | sample evolution shows English/Spanish gains | **planned** |

OpenSubtitles 500 MB sample is built (`data/opensubtitles.txt`) and
the cache (`data/_tatoeba_cache/`, `data/_opensubtitles_cache/`) is
in place.

### Workstream E — Distillation (the meta-lever)

**Reality check: tokenizer mismatch (byte-level student, BPE teacher)
ruled out direct logit KL distillation.** Pivoted to Path A —
pseudo-label distillation: generate text from teacher, train student
on it as plain corpus.

| # | Commit | Validation gate | Status |
|---|---|---|---|
| 12 | `make_teacher_corpus.py`: Qwen-2.5-1.5B-4bit on m4-mini via mlx-lm; produces `<en> :: <es>\n` pairs + cortex unary mixin | smoke parses cleanly, samples are real bilingual text | **done** (`6d9bbc9`) |
| 13 | (no new script) `train_bilingual_mlx.py --corpus data/teacher_corpus.txt` | n/a — existing trainer is corpus-agnostic | **ready** |
| 14 | Run student training on teacher corpus, compare to Tatoeba baseline at same step count | sample-quality A/B at fixed steps | **awaiting teacher corpus** |

**Teacher run live state:** PID 27750 on m4-mini, nohup'd.
Target 20 MB, generation rate ~9-15 KB/min (smoke-test measured at
~9 KB/min, will be a touch faster with bigger pairs-per-prompt).
ETA: ~24-37 hours to fill the target.

Corpus accumulates incrementally; usable from ~5 MB (~7 hours) for
preliminary student training. Full 20 MB is overnight + day.

Sync when ready:
```bash
rsync -av miguel_lemos@192.168.0.170:~/mamba3-hands-on/data/teacher_corpus.txt data/
```

### Workstream F — The headline experiment

| # | Commit | Validation gate | Status |
|---|---|---|---|
| 15 | `train_counter_attach.py`: load frozen bilingual LM, attach fresh CounterPrimitive, freeze LM, train only counter adapters | smoke forward + losses | **done** (`b9d6506`) |
| 16 | Run the actual fine-tune (1000 steps mixed batch) | counter aux loss converges, bilingual loss stable | **awaiting bilingual LM** |
| 17 | `demo_cortex.py`: side-by-side `count to fifty:` (en) / `cuenta hasta cincuenta:` (es) demonstrating compositional behavior | byte-perfect emissions on both prefixes | **planned** |

---

## What's running right now

| Process | Status | Where |
|---|---|---|
| PyTorch 10k bilingual training | step 2700+/10000, loss 1.14 bpc 1.64 | local, MPS, background |
| OpenSubtitles 500 MB sample | done | `data/opensubtitles.txt` |
| MLX bf16 trainer | benched at 1.47 sec/step | ready when bilingual LM done |
| Counter-attach script | code committed | `train_counter_attach.py` |

---

## Risks I'm tracking

- **MLX parity miss** — passed (3.58e-7).
- **Chunked scan accuracy loss** — not yet exercised; will require care.
- **Distillation teacher cost** — bound by storing top-K logits per token (top-100 keeps shard size sane); m4-mini disk has the room.
- **Counter-attach failure on language-trained LM** — the actual research uncertainty. If the counter cannot find the bilingual LM's "I am reading unary form" signals despite the 5% mixin in training data, we learn what the plugin port needs (frozen-LM + adapter fine-tune is insufficient → need a designed plugin protocol).
- **Frozen bilingual LM quality ceiling** — at 473k params + 17.7 MB Tatoeba, the LM will not be conversable. The distillation lever is the only path to useful quality at this size; without it, the cortex composition demo is "counter works on a barely-coherent LM".

---

## Resume instructions

To pick up: run `git log --oneline --all` and read commits with the
`81f9125`..HEAD prefix. The active in-flight work is in `/tmp/lm_train.log`
(PyTorch training). Counter-attach is ready to fire as soon as
`checkpoints/lm/step_FINAL.pt` exists:

```bash
python train_counter_attach.py --lm-ckpt checkpoints/lm/step_FINAL.pt --steps 1000
```

Distillation requires teacher download first; do that on m4-mini:

```bash
ssh m4-mini 'cd ~/mamba3-hands-on && python distill_teacher.py'  # to be written
```
