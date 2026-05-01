---
title: Latent operator discovery
chapter: "13"
status: active
sections: true
summary: "Train a recurrent learner from raw prefixes, then test whether hidden state recovers the stack operator without explicit counts."
---

# Chapter 13 - Latent Operator Discovery

**Status:** active. This is the first stricter test after the introductory
operator scaffolding.

## Question

Can a small recurrent learner discover the latent state needed for a stack
operator from raw role prefixes alone?

The model does not receive `opens`, `closes`, or `depth`. It receives only:

- the raw prefix roles observed so far;
- the target pair count as task context;
- one candidate role to judge.

The candidate classifier must decide whether the next role is valid. If it
works on held-out longer lengths, the useful state had to be reconstructed in
the recurrent hidden state.

## Control Flow

```text
raw role prefix
  -> GRU encoder updates hidden state
  -> candidate role is scored from hidden state
  -> held-out prefixes test whether latent state generalizes
```

This is still a small experiment, but it removes the earlier mistake: the
learner is no longer handed the stack counters as features.

## Run

```bash
.venv/bin/python experiments/13_latent_operator_discovery/latent_operator_discovery.py
```

Fast smoke:

```bash
.venv/bin/python experiments/13_latent_operator_discovery/latent_operator_discovery.py \
  --epochs 180 --max-train-pairs 12 --min-heldout-pairs 13 --max-heldout-pairs 16
```

## Success Criteria

- Training finishes under one minute on CPU.
- Held-out candidate-validity accuracy is above 95%.
- A frozen-hidden linear probe recovers boundary facts such as `can_close` and
  `can_end`, showing that the hidden state carries the missing operator state.
