---
title: Stack operator transfer findings
chapter: "11"
status: active
sections: true
summary: "A 177-parameter learned stack operator trained in 0.104s and guided valid generation across parenthesis, bracket, and block surfaces."
---

# Stack Operator Transfer Findings

## Entry — A tiny learned operator can steer formal generation across surfaces

**Date:** 2026-05-01

**Question.** Can a learned transition operator, trained from stack-role traces,
participate during generation and transfer across different token surfaces?

**Setup.** The experiment trains a 177-parameter MLP over role-state features:

```text
x = opens_frac, closes_frac, depth_frac, boundary flags, candidate role
y = candidate role is valid / invalid
```

Roles are surface-independent:

```text
OPEN, CLOSE, END
```

Generation then remaps those roles to multiple surfaces:

- parentheses: `(` and `)`
- brackets: `[` and `]`
- word blocks: `BEGIN` and `END`

The operator is inside the generation loop. A noisy base generator proposes a
candidate role, the operator filters invalid candidates for the current stack
state, the harness emits a valid role, and the verifier checks the completed
sequence.

## Result

Default run:

```text
elapsed_s:            0.7141
train_elapsed_s:      0.1036
parameters:           177
train_examples:       249
train_acc:            1.0000
heldout_state_acc:    1.0000
baseline_pass_rate:   0.0125
guided_pass_rate:     1.0000
```

The baseline noisy generator produced valid parenthesis strings only 1.25% of
the time. With the learned operator in the loop, generation reached 100% pass
rate across parenthesis, bracket, and block surfaces, including held-out target
lengths larger than the training traces.

## Interpretation

This is a small result, but it is the right kind of result for the new route.
It does not show that a language model became smarter. It shows that:

1. a tiny learned policy can encode a reusable state/action constraint;
2. the learned policy can operate on roles instead of raw tokens;
3. the same role policy can transfer across surfaces;
4. one-minute iterations are enough to get a falsifiable signal.

That makes this chapter different from ordinary tool calling. The operator is
not called after generation as a tool. It constrains generation at every step.

## Limits

The state features are still hand-designed. This does not yet prove automatic
state discovery. The harness knows stack depth, completion boundaries, and the
role vocabulary.

The next step is not to add a bigger model. The next step is to make the state
abstraction less hand-given:

- train from raw role traces instead of explicit depth features;
- transfer from stack roles to JSON-like grammar roles;
- compare deterministic masks, learned masks, and unmasked generation;
- log each run through the Lab telemetry path.

## Reproduce

```bash
.venv/bin/python experiments/11_stack_operator_transfer/stack_operator.py
```

Fast smoke:

```bash
.venv/bin/python experiments/11_stack_operator_transfer/stack_operator.py \
  --epochs 150 --trials 20
```

Artifacts:

- `experiments/11_stack_operator_transfer/artifacts/stack_operator_results.json`
- `experiments/11_stack_operator_transfer/artifacts/stack_operator.pt` locally
  when generated; `.pt` files are ignored by git.

