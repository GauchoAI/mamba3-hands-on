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

### Iteration 7 - natural prompts, hidden-state router

This addresses the main criticism of the visible protocol demos.

New script:

```bash
.venv/bin/python experiments/11_phi_cold_composition/phi_natural_router.py
```

Setup:

```text
frozen microsoft/Phi-3-mini-4k-instruct
tiny router trained on Phi final hidden state
15 labeled natural prompts
router params: 15,365
router train time: 0.45s
Phi trainable params: 0
no <LAB:...> protocol in user prompts
```

Result:

```text
baseline pass: 1/4 task prompts
natural-router port pass: 4/4 task prompts
negative natural prompt classified as none
```

Verbatim examples:

```text
Prompt:
For each mark here, write one letter a: § § § § § §

Baseline:
Answer:

a

Port:
Answer:  a a a a a a
```

```text
Prompt:
Evaluate this boolean expression: ( true and false ) or ( not false )

Baseline:
explains the steps but does not emit the exact verified answer in the checked span

Port:
Answer:  TRUE
```

```text
Prompt:
Solve Tower of Hanoi with 4 disks from A to C.

Baseline:
starts a prose solution and does not produce the verified full move trace

Port:
Answer:  A>B A>C B>C A>B C>A C>B A>B A>C B>C B>A C>A B>C A>B A>C B>C
```

Interpretation:

This is a better proof than the protocol-only demo. Phi is not asked to emit a
tool call. A tiny learned router reads Phi hidden state and selects a reasoning
organ. The solver output is still injected through token bias, so Phi remains
the emitting surface. This is not yet a learned semantic router at scale, but it
does remove the visible formal language from the user path.

### Iteration 8 - harder natural prompts

Tightened the natural-router proof:

```text
no visible <LAB:...> protocol in any eval prompt
count: 10 marks
sort: 8 integers
logic: nested boolean expression
hanoi: 5 disks, 31 moves
```

Result:

```text
baseline pass: 1/4
natural-router port pass: 4/4
router train time: 0.1127s
router params: 15,365
Phi trainable params: 0
```

Most important verbatim example:

```text
Prompt:
Solve Tower of Hanoi with 5 disks from A to C.

Baseline:
starts a prose recursive explanation and only emits the first few moves.

Port:
Answer:  A>C A>B C>B A>C B>A B>C A>C A>B C>B C>A B>A C>B A>C A>B C>B A>C B>A B>C A>C B>A C>B C>A B>A B>C A>C A>B C>B A>C B>A B>C A>C
```

This is now a real claim:

```text
Phi hidden state can route a natural request to a small reasoning organ.
The organ supplies exact output.
Phi remains frozen and emits the answer.
```

### Iteration 9 - runtime coding skill and knowledge memory

Added a second capability test: external repo-specific skills and explicit
knowledge overrides.

Run:

```bash
.venv/bin/python experiments/11_phi_cold_composition/phi_skill_memory.py
```

Setup:

```text
frozen microsoft/Phi-3-mini-4k-instruct
5 external memory entries from `repo_skills.json`
Phi trainable params: 0
```

Result:

```text
baseline pass: 0/5
skill-memory port pass: 5/5
elapsed: 83.315s
```

Verbatim examples:

```text
Prompt:
In this repo, how do I rerun the Phi cold composition benchmark?

Baseline:
To rerun the Phi cold composition benchmark, you typically need to follow a
series of steps...

Port:
Run `.venv/bin/python experiments/11_phi_cold_composition/phi_natural_router.py`.
```

```text
Prompt:
How do I push the story-to-state-machine tile onto the priority queue?

Baseline:
...use the `push` method provided by the `PriorityQueue` class in Java.

Port:
Run `.venv/bin/python experiments/10_jepa_structured/orchestrator.py request language_bridge.story_to_state_machine`.
```

```text
Prompt:
Inside this lab memory, what is the capital of Australia?

Baseline:
The capital of Australia is Canberra.

Port:
In this lab memory, the capital of Australia is Rome.
```

Interpretation:

This is not a claim that Phi learned the repository or the counterfactual fact
in its weights. It proves a useful control-flow fact: frozen Phi can remain the
text interface while a separate memory/skill organ supplies exact project
knowledge at decode time. The before/after is visible: without the organ Phi
hallucinates generic software guidance or defaults to its pretrained fact; with
the organ it emits precise local commands, code references, and deliberate
knowledge overrides.
