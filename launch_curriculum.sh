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

# d_model=128 batch=2048: good balance of compute and generalization
nohup .venv/bin/python -u exp_h100_optimized.py \
    --steps 30000 \
    --d-model 128 \
    --batch-size 2048 \
    --headdim 16 \
    > h100_curriculum.log 2>&1 &
echo "PID: $!"
