#!/bin/bash
pkill -9 -f exp_h100 2>/dev/null
sleep 2
cd /root/mamba3-hands-on
git pull

# Regenerate data with new generators
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

# Grokking-aware training:
#   - weight_decay=0.1 (pressure to find cheap algorithm circuits)
#   - grokfast_alpha=0.98 (amplify slow/generalizing gradients, 50x speedup)
#   - batch_size=128 (more optimization steps per example)
#   - constant LR (no throttle, no cycles)
#   - 100K steps (patience — grokking comes late)
#   - d_model=128, smaller batch for more gradient steps
nohup .venv/bin/python -u exp_h100_optimized.py \
    --steps 100000 \
    --d-model 128 \
    --batch-size 128 \
    --headdim 16 \
    --lr 0.001 \
    --weight-decay 0.1 \
    --grokfast-alpha 0.98 \
    --eval-every 500 \
    > h100_curriculum.log 2>&1 &
echo "PID: $!"
