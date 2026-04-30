"""RLF-Cortex experiment package.

Forks from jepa/ at commit 7866268 ("cortex: --n-loops flag for
RLF-inspired layer recursion"). Adds the layer-recursion + lifeline
re-injection ideas from batteryphil/mamba2backbonerecursion as
additional architectural variants on top of our existing JEPA-Cortex
stack. Future iterations may add LoopRoPE, HaltingHead, and the
prefix-scratchpad pieces — each as a new commit in this folder.

Self-contained per the project's immutability principle: local copies
of cortex_counting.py, mamba3_minimal.py, arch.py, data_loader.py,
checkpoint.py, eval_daemon.py, train.py, talk.py, and the SSM scan
helpers. The jepa/ folder is left clean of RLF additions so the two
experiments can run in parallel without contaminating each other's
state. Run as: uv run python rlf_cortex/train.py ...
"""
