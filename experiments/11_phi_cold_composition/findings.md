# Phi cold composition findings

## Iteration log

### Iteration 1 - setup

Goal: create the smallest cold-composition probe against a Microsoft Phi host.

Criterion: same frozen Phi checkpoint, same prompt, baseline generation versus
primitive-biased generation. No Phi weight update.

Checkpoint: the implementation now uses a generic `SolverPort`, not a
counter-only path. Any puzzle solver can plug in by implementing:

```text
token_bias(decoded_prefix, tokenizer) -> {token_id: logit_bias}
is_done_token(token_id, tokenizer) -> bool
```

The unary counter is only the first solver behind this interface. Hanoi,
sorting, or another trained specialist should use the same port if it can map
its next action to token biases.

Adjustment: `microsoft/phi-1_5` was not locally cached beyond tokenizer files,
so the first run spent minutes fetching weights. Switched the default host to
the locally cached `microsoft/Phi-3-mini-4k-instruct` to keep iterations short.

First Phi-3 run loaded and generated, but decoded text inserted spaces between
the emitted `a` tokens (`§§§: a a a a`). Character-level parsing made the
successful token bias look like failure. The port contract is now token-ID
based, which is also the right design for Hanoi/sorting action tokens.

Checkpoint result:

```text
prompt:   § § § :
baseline: § § § : 100        -> 0 emitted `a` tokens
cortex:   § § § : a a a      -> 3 emitted `a` tokens
```

Phi trainable params: 0. Same frozen Phi-3 host, generic solver port attached.
The prompt is cold symbolic notation, not a unary mixin seen by a local LM.

### Iteration 2 - wider N check

Command:

```bash
.venv/bin/python experiments/11_phi_cold_composition/phi_cold_counter.py --ns 3,8,16,32 --max-new-extra 1
```

Result:

```text
baseline pass: 0/4
cortex port pass: 4/4
Phi trainable params: 0
elapsed: 29.018s
```

Verbatim examples:

```text
N=8
prompt:   § § § § § § § § :
baseline: § § § § § § § § :

1. **Introduction to the Con
cortex:   § § § § § § § § : a a a a a a a a

N=32
baseline begins an essay about software packages.
cortex emits exactly 32 `a` tokens.
```

Interpretation: this is the first clean cold-composition positive in this
chapter. It is not trained adapter composition yet; it is a deterministic
solver plugged through a generic Phi token-bias port. The strong next step is
to put Hanoi or sorting behind the same `SolverPort` and show action-token
rollout, not a new port.
