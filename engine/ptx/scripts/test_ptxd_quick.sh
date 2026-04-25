#!/bin/bash
# Quick ptxd smoke test: one fast d=32 L=1 job, should converge in seconds.
set -e
cd /root/mamba3-hands-on/engine/ptx
echo '{"id":"j1","task":"parity","n_bits":3,"d_model":32,"d_state":16,"headdim":16,"n_layers":1,"vocab_size":260,"lr":0.001,"weight_decay":0.1,"steps":2000,"batch_size":16,"target_acc":0.95,"seed":12345}' | timeout 60 ./target/release/ptxd 2>&1
