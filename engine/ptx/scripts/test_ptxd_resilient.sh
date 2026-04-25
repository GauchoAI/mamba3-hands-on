#!/bin/bash
# Resilient ptxd smoke test: writes output to a file so SSH drops don't lose results.
# Run in nohup so the process survives any SSH reconnect.
set -e
cd /root/mamba3-hands-on/engine/ptx
LOG=/root/ptxd_smoke.log
: > $LOG
nohup bash -c '
echo "=== ptxd smoke test, $(date) ==="
echo "--- Test 1: L=2 fixed n_bits=3 ---"
echo '"'"'{"id":"j_L2","task":"parity","n_bits":3,"d_model":32,"d_state":16,"headdim":16,"n_layers":2,"vocab_size":260,"lr":0.001,"weight_decay":0.1,"steps":2000,"batch_size":16,"target_acc":0.95,"seed":12345}'"'"' | timeout 60 ./target/release/ptxd 2>&1 | tail -15
echo "--- Test 2: curriculum, d=32 L=2 ---"
echo '"'"'{"id":"j_curric","task":"parity","d_model":32,"d_state":16,"headdim":16,"n_layers":2,"vocab_size":260,"lr":0.001,"weight_decay":0.1,"steps":4000,"batch_size":16,"target_acc":0.95,"seed":7,"stages":[{"min_len":2,"max_len":4,"advance_at":0.9},{"min_len":3,"max_len":8,"advance_at":0.9},{"min_len":4,"max_len":16,"advance_at":0.95}]}'"'"' | timeout 120 ./target/release/ptxd 2>&1 | tail -25
echo "=== done ==="
' > $LOG 2>&1 &
echo "started, pid=$!. tail -f $LOG to follow"
