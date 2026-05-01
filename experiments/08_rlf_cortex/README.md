---
title: RLF Cortex
chapter: "08"
status: archival
sections: true
summary: "Recursive Latent Forcing fork, kept separate so the book can show it as an isolated experiment line."
---

# Chapter 08 — RLF cortex

**Status:** archival.

## rlf_cortex/ — RLF-inspired Cortex experiment

Fork of `jepa/` (at commit `7866268`) that explores **layer recursion**
and the broader Recursive Latent Forcing (RLF) idea-set as architectural
additions to our existing CortexLM. Goal: see whether the RLF tricks help
on the same talking + reasoning task that jepa/ is targeting, isolated
from jepa/ so neither experiment contaminates the other.

## Inspiration

[`batteryphil/mamba2backbonerecursion`](https://github.com/batteryphil/mamba2backbonerecursion)
— a Mamba SSM project pursuing the same broad goal as our Cortex thesis
(get small models to *think more*, not have *more parameters*) but via
a different mechanism: loop the model's hidden state through itself
multiple times before emitting a token, instead of injecting algorithmic
primitives into the residual stream.

We did a thorough sweep of their repo and folded the honest assessment
into [`findings.md`](findings.md) — what's substantive (the 2.8B's 75%
BIG-Bench result, O(1) VRAM proof, prefix-scratchpad / lifeline / LoopRoPE
ideas), what's hype (the 1.4B port currently fails at ~5% chain accuracy,
UEFI baremetal claim is unproven), and which techniques we want to try
on our own stack.

## What's different from `jepa/`

Identical baseline (same `cortex_counting.py`, `train.py`, `arch.py`,
etc.) at the moment of the fork, plus:

| Variant | Mechanism |
|---|---|
| `--n-loops 1` (default) | identical to jepa/'s baseline |
| `--n-loops N` (N > 1) | re-runs the SSM stack N times per token; primitives re-fire each loop; original embedding re-injected with `0.5^loop_i` decay (lifeline analog) |

Future variants planned in this folder (each a separate commit):
- **LoopRoPE** — geometric loop-index encoding to break fixed-point collapse
- **HaltingHead** — separate sigmoid MLP that learns when to stop emitting
- **Prefix scratchpad** — M learnable tokens prepended, evolved each loop
- **Cognitive Router prefix** — `[COUNT]` vs `[TALK]` system 1/2 routing

## Layout

| File | Same as jepa/ | Difference |
|---|---|---|
| `cortex_counting.py` | mostly | adds `n_loops` to CortexLMConfig, recursion in `forward()` |
| `train.py` | mostly | `--n-loops` CLI flag; default `ckpt_root`/`runs_root` switched to `rlf_cortex/` |
| `mamba3_minimal.py`, `ssm_*.py`, `arch.py`, `data_loader.py`, `checkpoint.py`, `eval_daemon.py`, `talk.py`, `make_teacher_thoughts.py`, `cerebras_corpus.py` | identical at fork time | none yet |

## Running

```bash
# Same precondition as jepa/: needs data/teacher_thoughts.bin and
# data/bilingual.txt populated. The two experiments share the data
# directory — only checkpoints + runs are namespaced separately.

uv run python rlf_cortex/train.py \
    --run-name gpu3-recurse-n3 \
    --steps 10000 --batch-size 32 --seq-len 256 \
    --lambda-jepa 0.3 --lambda-sigreg 0.3 \
    --mix-unary 0 --mix-teacher 0.7 --mix-biling 0.3 \
    --n-loops 3
```

## How to read the comparison

`jepa/` and `rlf_cortex/` produce checkpoints in different folders
(`checkpoints/jepa_cortex/<run>/` vs `checkpoints/rlf_cortex/<run>/`)
and runs metadata in different folders (`runs/jepa_cortex/<run>/` vs
`runs/rlf_cortex/<run>/`). The `eval_daemon` in either folder can be
pointed at any run dir — useful for cross-experiment comparison without
forcing one experiment to know about the other's structure.

To compare gpu0-pure-bilingual (jepa, n_loops=1) against gpu3-recurse-n3
(rlf_cortex, n_loops=3) on the same canary set, point one daemon at both
run paths:

```bash
uv run python rlf_cortex/eval_daemon.py --serve --port 8090 \
    --device cuda:0 \
    --runs runs/jepa_cortex/gpu0-pure-bilingual,runs/rlf_cortex/gpu3-recurse-n3
```

The diversity metric, byte CE on bilingual, and counter accuracy are
all compute-cost-comparable across the two — the only thing changing
is what's inside the model.
