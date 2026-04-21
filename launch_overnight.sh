#!/bin/bash
# 10-hour curriculum run on H100
# Resumes from current checkpoint if available
# ~600K steps at 60ms/step
pkill -9 -f exp_h100 2>/dev/null
sleep 2
cd /root/mamba3-hands-on
git pull

# Copy current run's checkpoints so the overnight run can resume
# The plain model checkpoints from h100_curriculum run use prefix "opt_plain"
# We keep using the same prefix so auto-resume finds them

# Regenerate data with all generators
.venv/bin/python -c "
import sys; sys.path.insert(0, '.')
from generators.level0_patterns import generate_dataset
import json
examples = generate_dataset(10000)
with open('data/level0/patterns.jsonl', 'w') as f:
    for ex in examples:
        f.write(json.dumps(ex) + '\n')
print('Generated %d examples' % len(examples))
"

# Resume from checkpoint, run to 600K total steps
nohup .venv/bin/python -u exp_h100_optimized.py \
    --steps 600000 \
    --d-model 128 \
    --batch-size 128 \
    --headdim 16 \
    --lr 0.001 \
    --weight-decay 0.1 \
    --grokfast-alpha 0.98 \
    --eval-every 500 \
    > h100_overnight.log 2>&1 &
echo "PID: $!"
echo "Expected runtime: ~10 hours"
