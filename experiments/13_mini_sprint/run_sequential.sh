#!/usr/bin/env bash
# Run exp_01 through exp_05 sequentially after exp_00 finishes.
#
# Usage on the mini:
#   tmux new-session -d -s exp_seq \
#     "cd ~/mamba3-hands-on && bash experiments/13_mini_sprint/run_sequential.sh"
#
# Each experiment writes its own runs/<name>/{loss.jsonl, eval.json}. The
# cron tick on the mac will pick up eval.json files as they land and fold
# the result into findings.md §8.

set -e
cd ~/mamba3-hands-on
PY=.venv/bin/python

# Wait for exp_00 to finish writing its eval.json (signals completion)
echo "[runner] waiting for exp_00 to complete..."
while [ ! -f runs/exp_00_clean_corpus_baseline/eval.json ]; do
  sleep 30
done
echo "[runner] exp_00 done."

for exp in exp_01_ema_self_distill exp_02_multi_scale_distill \
           exp_03_curriculum exp_04_residual_norm exp_05_combined; do
  echo "[runner] === starting $exp ==="
  $PY experiments/13_mini_sprint/${exp}.py \
    --corpus data/movie_pairs_clean.txt \
    2>&1 | tee runs/${exp}/run.log || true
  if [ -f runs/$exp/eval.json ]; then
    echo "[runner] $exp eval:"
    grep -E '"retention"|"drift"|"diversity"' runs/$exp/eval.json | head -3
  fi
done
echo "[runner] all experiments done."
