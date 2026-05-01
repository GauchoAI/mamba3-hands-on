# Deployment — JEPA-Cortex on rented vast.ai box

## Box

| Item | Value |
|---|---|
| Provider | vast.ai |
| GPUs | 4× RTX 4070 Ti (12 GB GDDR6X each, ~40 TF BF16, ~504 GB/s) |
| Access | `ssh -p 37347 root@ssh4.vast.ai -L 8080:localhost:8080` |
| Port-forward | `8080` → local `8080` (live eval dashboard) |

Vast.ai SSH ports rotate when an instance is destroyed or paused; treat
the line above as ephemeral and re-document on each new rental.

## Workflow — commit, push, pull

Code lives in git; data and checkpoints are kept out of git per the
existing `data/.gitignore` convention. Round-trip a change like this:

```
local M4 Pro                         vast.ai box
─────────────                        ─────────────────
edit jepa/*.py                       (idle)
git add jepa/                        ↓
git commit -m "..."                  ↓
git push origin main          ─────► git pull   (fast-forward only)
                                     uv run python jepa/train.py --run-name ...
                              ◄───── rsync runs/ + checkpoints/ back when done
```

For one-time data transfers (e.g. shipping a freshly-built
`data/bilingual.txt` to skip the Tatoeba download on the box) use
`rsync` over the same SSH connection:

```bash
rsync -avz -e "ssh -p 37347" data/bilingual.txt \
    root@ssh4.vast.ai:/workspace/mamba3-hands-on/data/
```

## 4× 4070 Ti — 4 parallel experiments, not 1 big run

The model is 1–5M params and easily fits in 12 GB. Useful parallelism is
**fan-out across hyperparameter variants**, not data-parallel on a single
run. Plan A:

| GPU | Role |
|---|---|
| 0 | Reference: `λ_jepa=1.0, λ_sigreg=0.1, λ_aux=0.5` — canonical baseline |
| 1 | Variant A: `λ_jepa=0.3` — lower JEPA pressure |
| 2 | Variant B: `λ_sigreg=0.5` — harder isotropy |
| 3 | W1 first (teacher generation), then Variant C once `data/teacher_thoughts.bin` is sufficient (~5 MB is enough to start training) |

## First-time box setup

```bash
ssh -p 37347 root@ssh4.vast.ai -L 8080:localhost:8080

cd /workspace
git clone <repo-url> mamba3-hands-on
cd mamba3-hands-on

# uv on PyTorch images: install if missing
which uv || curl -LsSf https://astral.sh/uv/install.sh | sh

# Project deps via uv (pyproject.toml drives this).
# Optional JEPA group adds transformers / accelerate / sentencepiece.
uv pip install -e .
uv pip install transformers accelerate sentencepiece

# Sanity: 4 GPUs, ~12 GB each
nvidia-smi --query-gpu=index,name,memory.total --format=csv
```

## Phase 1 — datasets

Two ways to obtain `data/bilingual.txt` on the box:

**A. Regenerate from Tatoeba (slow only the first time, ~7 MB download):**

```bash
uv run python make_bilingual_corpus.py
# produces data/bilingual.txt (~17 MB) + data/_tatoeba_cache/
```

**B. Rsync the local copy (skip the download):**

```bash
# from the M4 Pro
rsync -avz -e "ssh -p 37347" \
    data/bilingual.txt root@ssh4.vast.ai:/workspace/mamba3-hands-on/data/
```

Then build the JEPA paired dataset (one-shot, runs on GPU 3 in tmux). The
trainer can begin once `.bin` reaches ~5 MB; `--resume` keeps appending so
the corpus grows in parallel with training:

```bash
tmux new -s teacher
CUDA_VISIBLE_DEVICES=3 uv run python jepa/make_teacher_thoughts.py \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --target-mb 80 \
    --out data/teacher_thoughts \
    --device cuda --dtype bfloat16
# Ctrl-b d to detach
```

## Phase 2 — four parallel trainers

One `tmux` per GPU, each pinned with `CUDA_VISIBLE_DEVICES`. Checkpoints
land in `checkpoints/jepa_cortex/<run_name>/` and the eval daemon watches
`runs/jepa_cortex/<run_name>/`.

```bash
# pane 0 — reference run
tmux new -s gpu0
CUDA_VISIBLE_DEVICES=0 uv run python jepa/train.py \
    --run-name gpu0-ref --steps 30000 \
    --lambda-jepa 1.0 --lambda-sigreg 0.1 --lambda-aux 0.5

# pane 1 — variant A (less JEPA)
tmux new -s gpu1
CUDA_VISIBLE_DEVICES=1 uv run python jepa/train.py \
    --run-name gpu1-lowjepa --steps 30000 --lambda-jepa 0.3

# pane 2 — variant B (more isotropy)
tmux new -s gpu2
CUDA_VISIBLE_DEVICES=2 uv run python jepa/train.py \
    --run-name gpu2-highsig --steps 30000 --lambda-sigreg 0.5

# pane 3 — variant C (after teacher generation finishes)
tmux new -s gpu3
CUDA_VISIBLE_DEVICES=3 uv run python jepa/train.py \
    --run-name gpu3-stride8 --steps 30000 \
    --override-stride-bytes 8
```

## Phase 3 — live dashboard

`eval_daemon.py` polls each run's `MANIFEST.jsonl`, evaluates new light
checkpoints on a held-out slice, and serves the rolling table on `:8080`.
With the SSH tunnel open, the laptop browser opens `http://localhost:8080`.

```bash
tmux new -s dash
uv run python jepa/eval_daemon.py --serve --port 8080 --device cuda:0 \
    --runs runs/jepa_cortex/gpu0-ref,runs/jepa_cortex/gpu1-lowjepa,runs/jepa_cortex/gpu2-highsig,runs/jepa_cortex/gpu3-stride8
```

## Disk layout on the box

```
/workspace/mamba3-hands-on/
  data/
    bilingual.txt                 # rsync'd or regenerated
    teacher_thoughts.{bin,idx}    # produced by jepa/make_teacher_thoughts.py
  checkpoints/jepa_cortex/<run>/  # AsyncCheckpointer writes here
  runs/jepa_cortex/<run>/         # loss.jsonl, samples.jsonl, metrics.jsonl
```

vast.ai instances usually expose `/workspace` as the persistent volume —
keep the repo there, not in `/root` or `~`, which are ephemeral on most
templates.

## Disk-budget quick math

- HF cache for Qwen weights: ~3 GB
- `teacher_thoughts.bin` at 80 MB target + index: ~80 MB
- Per-run light checkpoints (~50 retained × 10 MB): ~500 MB
- Per-run heavy checkpoints (~120 retained × 30 MB): ~3.6 GB
- 4 runs × 4 GB ≈ 16 GB total

A 50 GB persistent volume is plenty.

## Shutdown / preserve

When experiments finish, sync the meaningful artifacts back to the M4 Pro:

```bash
# on the laptop
rsync -avz -e "ssh -p 37347" \
    root@ssh4.vast.ai:/workspace/mamba3-hands-on/checkpoints/jepa_cortex/ \
    checkpoints/jepa_cortex/
rsync -avz -e "ssh -p 37347" \
    root@ssh4.vast.ai:/workspace/mamba3-hands-on/runs/jepa_cortex/ \
    runs/jepa_cortex/
```

Then pause or destroy the vast.ai instance from the web UI to stop billing.
