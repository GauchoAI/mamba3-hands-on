---
title: Phi cold composition
chapter: "11"
status: active
sections: true
summary: "Cold-attach a computational primitive to a frozen Microsoft Phi host and test whether it adds a capability the host was not trained for."
---

# Chapter 11 - Phi Cold Composition

## Question

Can a computational primitive add a new algorithmic behavior to a frozen
Microsoft Phi language model without retraining Phi?

This is the strong cortex claim in its smallest executable form:

```text
frozen Phi logits
  + primitive head bias from a symbolic counter
  -> corrected decoding
```

## Run

```bash
.venv/bin/python experiments/11_phi_cold_composition/phi_cold_counter.py
```

## First Result

With frozen `microsoft/Phi-3-mini-4k-instruct`:

```text
prompt:   § § § :
baseline: § § § : 100
cortex:   § § § : a a a
```

Across `N={3,8,16,32}` the baseline produced zero valid unary outputs; the
same frozen Phi host with the generic solver port produced the exact count in
all four cases. Phi trainable parameters: `0`.

## Subchapters

- Explicit prefix gating: interventions require `<LAB:...>` protocol markers.
- Resume after intervention: the port writes a boundary and releases control.
- Negative control: ungated symbolic text leaves Phi untouched.
- Plugin registry: counter, sorting, factual override, and boolean logic share
  one Phi-facing adapter.
- Logic direction: `<LAB:logic> ( true and false ) or ( not false ) :` emits
  `TRUE` through the same port.
- Request compiler: user-facing text can be compiled into the Lab protocol
  before the forward-pass intervention.
- Hanoi: `<LAB:hanoi> 3 :` emits `A>C A>B C>B A>C B>A B>C A>C`
  through the same port, then releases control.
- Natural hidden-state router: normal user prompts are classified from Phi's
  final hidden state, then solvers emit answers without visible protocol tokens.
- Runtime skill/knowledge memory: repo-specific coding skills and external
  facts are retrieved from JSON memory and emitted through frozen Phi, turning
  generic answers into exact project commands or deliberate knowledge overrides
  without updating Phi.
- Chess mate-in-one organ: a wider Torch MLP learns generated expert traces,
  predicts held-out legal checkmates, and supplies the UCI move to frozen Phi.
- Paired chess benchmark: raw Phi, notation-skill Phi, and Phi plus the chess
  MLP are scored on the same held-out positions with semantic chess parsing.
