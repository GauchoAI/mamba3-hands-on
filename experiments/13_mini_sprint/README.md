# Mac Mini Sprint — DeepSeek V4 Inspirations

Small, fast iterations of ideas from the DeepSeek V4 paper, run on the M4
mini (`miguel_lemos@192.168.0.170`) overnight to bracket the autopilot
problem in `experiments/07_jepa/`. Each experiment is a self-contained
script in this folder so the commit history maps 1-to-1 to a strategy.
Findings: see `experiments/07_jepa/findings.md` §8.

## Layout

- `extract_movie_pairs.py` — prereq, builds `data/movie_pairs_clean.txt`
  from the OpenSubtitles raw cache + `.ids` movie metadata. Blank line
  between movies, so consumers get only within-movie consecutive pairs.
- `exp_00_clean_corpus_baseline.py` — control: biling-only on clean corpus
- `exp_01_ema_self_distill.py` — V4 anticipatory-routing analog
- `exp_02_multi_scale_distill.py` — V4 hybrid-attention analog
- `exp_03_curriculum.py` — V4 curriculum, short→long pairs
- `exp_04_muon.py` — V4 Muon optimizer vs AdamW
- `exp_05_residual_norm_constraint.py` — V4 MHC light
- `exp_06_combined.py` — V4 compose: best 2-3 stacked

Each file's docstring spells out the hypothesis, the lever, and the exact
config. They all import shared model/loss code from `experiments/07_jepa/`
to avoid duplicating mamba3 / SSM kernels.

## Run conventions

- Trainer scale: `d_model=96`, `n_layers=2`, `batch=32`, `seq_len=128`,
  `steps=2000` — keeps each experiment under 30 min wall-clock on M4.
- Eval after run: 8-prompt canary retention/drift/diversity, captured to
  `runs/<exp_name>/eval.json`.
- All runs start from the same RNG seed so step-0 baselines are
  identical and any divergence is the experimental lever.
