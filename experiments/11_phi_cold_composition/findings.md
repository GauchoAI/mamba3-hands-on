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

### Iteration 3 - gated registry, release, negative control, extra solvers

Implemented the larger chapter shape:

```text
SolverPort registry
  - unary counter
  - sort solver
  - factual override solver
```

The counter now requires an explicit prefix protocol:

```text
<LAB:count> § § § :
```

Without `<LAB:count>`, the port stays inactive.

Run:

```bash
.venv/bin/python experiments/11_phi_cold_composition/phi_cold_counter.py --ns 3,8 --max-new-extra 6
```

Result:

```text
baseline pass: 0/2
cortex pass: 2/2
negative control: inactive
sort demo: OK
fact override demo: OK
Phi trainable params: 0
elapsed: 16.888s
```

Verbatim:

```text
count prompt: <LAB:count> § § § :
cortex:       <LAB:count> § § § : a a a

negative:     § § § :
output:       § § § : 10000

sort prompt:  <LAB:sort> 3 1 2 :
output:       <LAB:sort> 3 1 2 :  1 2 3

fact prompt:  <LAB:fact:capital-au> The capital of Australia is:
output:       ... Australia is: Sydney.
```

Important correction: release was not automatic. If the port simply stopped
biasing after the correct number of `a` tokens, Phi continued the local pattern
and emitted extra `a`s. The port now writes a boundary token, marks the solver
complete, and hands control back to Phi. This is the resume-after-intervention
mechanism.

### Iteration 4 - logic plugin

Added a fourth solver to the same registry:

```text
<LAB:logic> ( true and false ) or ( not false ) :
```

Output:

```text
<LAB:logic> ( true and false ) or ( not false ) : TRUE
```

This is the first small step toward the real objective: Phi handles text, while
a deterministic/reasoning organ handles the logical computation and writes the
answer through the same token-bias port. It is still a toy grammar, but it is
now aimed at better reasoning rather than only symbolic counting.

### Iteration 5 - request compiler

Added a deterministic front door from user-facing text into the Lab protocol.

Input:

```text
Please solve this logic statement: ( true and false ) or ( not false )
```

Compiled prefix:

```text
<LAB:logic>  ( true and false ) or ( not false ) :
```

Phi + solver-port output:

```text
<LAB:logic>  ( true and false ) or ( not false ) : TRUE
```

This is still software compilation, not learned language understanding. But it
is the correct control-flow skeleton for the strategic direction:

```text
user language -> formal protocol -> reasoning organ -> Phi-visible answer
```

### Iteration 6 - Hanoi behind the same port

Added `HanoiSolver` to the existing `SolverPort` registry.

Input:

```text
<LAB:hanoi> 3 :
```

Output:

```text
<LAB:hanoi> 3 :  A>C A>B C>B A>C B>A B>C A>C
```

Then Phi resumes:

```text
### Response:The given problem is a classic example...
```

This is the important integration checkpoint. The adapter did not change for
Hanoi. Counter, sorting, fact override, boolean logic, and Hanoi all share the
same Phi-facing mechanism:

```text
solver computes next action / answer
solver returns token biases
frozen Phi emits the answer tokens
boundary releases control back to Phi
```
