---
title: Latent operator discovery findings
chapter: "13"
status: active
summary: "A GRU prefix encoder recovers enough latent stack state to classify held-out candidate roles without explicit counters."
---

# Findings

This chapter is a stronger claim than the operator intro. The learner is not
given stack counters; it has to reconstruct them from raw prefixes.

The result should be read carefully. The recurrent model is still trained
supervised on candidate validity, so this is not unsupervised concept formation.
But it does demonstrate that the useful operator state can live inside a small
learned hidden state rather than only in hand-written features.

The next step should make the discovery pressure stronger again: multiple
operator families, shared hidden state, and a held-out family that tests whether
the learned representation transfers beyond stack traces.

