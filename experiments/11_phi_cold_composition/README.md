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
