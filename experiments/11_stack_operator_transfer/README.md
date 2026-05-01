---
title: Stack operator transfer
chapter: "11"
status: active
sections: true
summary: "One-minute experiment discipline: learn a tiny transition operator from stack traces, then reuse it across surface languages."
---

# Chapter 11 — Stack operator transfer

**Status:** active. This chapter starts the post-long-training discipline.

## Question

Can a tiny learned transition operator, trained only from stack traces, steer
sequence generation across different surface languages without fine-tuning a
language model?

This is not ordinary tool calling. The operator participates at each generation
step:

```text
generator proposes role candidates
  -> learned operator predicts which roles are valid in the current state
  -> harness emits one valid role
  -> state updates
  -> verifier checks the final sequence
```

## One-Minute Rule

Every iteration in this chapter must finish in under one minute on CPU unless
explicitly marked as a later confirmation run. Training should normally be
subsecond. If an idea needs long training to look promising, it is the wrong
iteration for this chapter.

## First Operator

The first operator is a supervised binary classifier:

```text
x = stack state + candidate role
y = candidate is valid / invalid
```

Roles:

- `OPEN`
- `CLOSE`
- `END`

The same learned policy is tested through different renderers:

- parentheses: `(`, `)`
- brackets: `[`, `]`
- block words: `BEGIN`, `END`

The important test is whether the operator learned stack discipline over roles,
not a single string surface.

## Run

```bash
.venv/bin/python experiments/11_stack_operator_transfer/stack_operator.py
```

Fast smoke:

```bash
.venv/bin/python experiments/11_stack_operator_transfer/stack_operator.py \
  --epochs 150 --trials 20
```

Outputs:

- `artifacts/stack_operator.pt`
- `artifacts/stack_operator_results.json`

## Success Criteria

- Training finishes under one minute.
- The learned operator reaches high held-out state accuracy on larger `n` than
  training.
- Operator-guided generation produces valid sequences across remapped surfaces.
- The unconstrained noisy generator fails often enough to make the operator's
  contribution visible.
