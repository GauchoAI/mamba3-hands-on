#!/bin/bash
# Three ptxd tests:
#   1. Single fixed n_bits=3, default config (fast)
#   2. Curriculum (matches problems/parity/problem.yaml), small model
#   3. Winning config (d=64 L=4) with curriculum
set -e
cd /root/mamba3-hands-on/engine/ptx
cat <<'JSONEOF' | timeout 600 ./target/release/ptxd 2>&1
{"id":"j1_fixed3","task":"parity","n_bits":3,"d_model":32,"d_state":16,"headdim":16,"n_layers":1,"vocab_size":260,"lr":0.001,"weight_decay":0.1,"steps":2000,"batch_size":16,"target_acc":0.95,"seed":12345}
{"id":"j2_curric","task":"parity","d_model":32,"d_state":16,"headdim":16,"n_layers":2,"vocab_size":260,"lr":0.001,"weight_decay":0.1,"steps":4000,"batch_size":16,"target_acc":0.95,"seed":7,"stages":[{"min_len":2,"max_len":4,"advance_at":0.9},{"min_len":3,"max_len":8,"advance_at":0.9},{"min_len":4,"max_len":16,"advance_at":0.95}]}
{"id":"j3_big","task":"parity","d_model":64,"d_state":8,"headdim":16,"n_layers":4,"vocab_size":260,"lr":0.001,"weight_decay":0.1,"steps":1000,"batch_size":256,"target_acc":0.95,"seed":42,"stages":[{"min_len":2,"max_len":4,"advance_at":0.9},{"min_len":3,"max_len":8,"advance_at":0.9},{"min_len":4,"max_len":16,"advance_at":0.95}]}
JSONEOF
