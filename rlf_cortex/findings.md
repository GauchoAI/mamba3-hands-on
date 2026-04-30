# RLF-Cortex — experiment journal

Living document for the RLF-Cortex experiment branch. Same three-section
structure as `jepa/findings.md`:
1. **Hypothesis** — what we're testing, written before launch.
2. **Live observations** — timestamped notes as the run unfolds.
3. **Conclusion** — written when training stops.

## 1. Hypothesis (ex ante, 2026-04-30)

### Why this experiment exists

After the first JEPA round (`jepa/findings.md`) all four runs degraded
qualitatively past step ~3000 — falling into the unary attractor,
losing prompt-conditioning, while byte_ce_biling kept improving (the
metric/quality disconnect we documented). One open question from that
round: **is the late-training collapse fixable by giving the model more
*compute per token* without more *parameters per token*?**

A nearby Reddit post pointed at
[`batteryphil/mamba2backbonerecursion`](https://github.com/batteryphil/mamba2backbonerecursion),
which argues exactly this — that small Mamba models can reason if you
loop their hidden states back through the same layers N times before
emitting each token (the "Recursive Latent Forcing" trick). The repo
contains a more sophisticated version of the idea: prefix scratchpad
tokens, lifeline re-injection of the original prompt at each loop,
LoopRoPE for geometric loop-index distinction, and a HaltingHead that
learns when to stop.

We did a careful read of that repo (full sweep in commit messages on
`main`). Honest take:

| Claim | Status |
|---|---|
| 2.8B variant achieves 75% on BIG-Bench Lite | ✅ real, in repo logs |
| O(1) VRAM across 20+ loops | ✅ empirically demonstrated |
| RLF idea-set is novel | ✅ — combination of prefix-scratchpad + lifeline + LoopRoPE is genuinely different |
| 1.4B port hits 70-75% as advertised | ❌ — eval logs show ~5% chain accuracy, mostly empty outputs |
| UEFI baremetal inference works | ⚠️ — code present, no boot validation |
| Auto-N entropy scaler (per Reddit post) | ❌ — not in code, code uses HALT token instead |

So: **the conceptual work is solid, the 2.8B validation is real, but
the 1.4B reproduction is currently broken in their repo.** Worth
trying the techniques, with eyes wide open.

### What this branch is testing

A **minimum viable RLF on our 1M-param Mamba-3 stack**: just the layer
recursion (re-run the SSM stack N times per token) + a simple lifeline
(re-add decayed original embedding each loop). Zero new parameters; only
compute increases by ~N×.

This is intentionally *less* than what RLF does. The reason: their full
stack (prefix scratchpad + LoopRoPE + dedicated loop-SSM + low-rank
bridge + HaltingHead) is six interacting pieces. Their own 1.4B port
shows that getting all six to cooperate is hard. We isolate one variable
at a time:

| Round | What we add | What we'd learn |
|---|---|---|
| **1 (this round)** | bare layer recursion + lifeline | Does pure recursion alone help? Or does the model need the structural additions? |
| 2 (later, if 1 fails) | + LoopRoPE | Does breaking fixed-point collapse fix what 1 didn't? |
| 3 (later, if 2 helps) | + HaltingHead | Can we learn when to stop, attacking our counter-doesn't-stop bug? |
| 4 (later, if 3 helps) | + prefix scratchpad | Full RLF, ours-flavored |

### Variant we're launching first

Single variant on GPU 3 (other three GPUs continue the jepa/ round 2
runs from gpu0-pure-bilingual / gpu1-no-cortex / gpu2-tinier):

```
gpu3-recurse-n3:
  --n-loops 3                  ← the one variable being tested
  --batch-size 32              ← halved because n_loops=3 → 3× SSM compute
  --seq-len 256
  --lambda-jepa 0.3            ← matches jepa/'s gpu1-lowjepa sweet spot
  --lambda-sigreg 0.3          ← bumped from 0.1 (gpu2-highsig kept best diversity)
  --mix-unary 0                ← no unary attractor in the corpus
  --steps 10000
```

### Predictions, ranked

- **Most likely (50%):** byte_ce_biling within 0.05 of jepa/'s gpu0-pure-bilingual
  at the same step count (so recursion is roughly compute-neutral). Counter
  accuracy still 0 because we have no halting mechanism. Diversity perhaps
  marginally better because the multi-loop forward gives the SSM more
  chances to develop varied responses per prompt.
- **Plausible (25%):** byte_ce_biling materially better (~0.1 lower) because
  recursion gives the model more capacity to "think". This would be a real
  result worth following up on with LoopRoPE.
- **Plausible (15%):** byte_ce_biling materially *worse* — the lifeline
  re-injection plus the same-weight repeated application produces a fixed
  point the model can't escape. Would tell us we need LoopRoPE before
  recursion can help at all.
- **Unlikely (10%):** counter accuracy spontaneously becomes nonzero
  because the recursive SSM eventually produces a representation where
  the reset_gate fires on the right position. This would be the surprising
  upside.

### What this experiment will NOT tell us

- Whether RLF's full stack works at our scale. We're testing one piece.
- Whether HaltingHead beats our hard-gates-inference. Different round.
- Whether LoopRoPE matters. Different round.
- Whether 1.4B-RLF eventually works in their repo. That's their problem.

The honest scope here is "does layer recursion + lifeline alone help on
a 1M-param byte-level Mamba-3 with JEPA distillation?"

---

## 2. Live observations

*(Will be filled in once gpu3-recurse-n3 is running on the box.)*

---

## 3. Conclusion

*(Pending.)*

Open questions to resolve in the conclusion:
- Did `n_loops=3` improve byte_ce_biling at matched compute vs jepa/'s
  gpu0-pure-bilingual at matched compute (i.e. step-count-adjusted)?
- Did the diversity metric (mean Jaccard distance across canary
  completions) hold higher than jepa/'s runs at later training?
- Did greedy-decode samples avoid mode collapse longer than jepa/'s
  runs did at step 4000?
- If counter_acc stayed 0 → does that close the door on "recursion alone
  can fix the stop signal", and force us into HaltingHead territory?

---

## How to reproduce

```bash
# Same prerequisites as jepa/ (data/bilingual.txt, data/teacher_thoughts.bin):
uv run python rlf_cortex/train.py \
    --run-name <my_run> \
    --steps 10000 --batch-size 32 --seq-len 256 \
    --lambda-jepa 0.3 --lambda-sigreg 0.3 \
    --mix-unary 0 --mix-teacher 0.7 --mix-biling 0.3 \
    --n-loops 3

# Dashboard (can include both jepa/ and rlf_cortex/ runs at once):
uv run python rlf_cortex/eval_daemon.py --serve --port 8090 \
    --device cuda:0 \
    --runs runs/jepa_cortex/gpu0-pure-bilingual,runs/rlf_cortex/<my_run>
```
