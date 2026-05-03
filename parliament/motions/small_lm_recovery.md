---
motion_id: small-lm-recovery
kind: roadmap
chapter: language_models
status: active
---

The lab wants to continue toward small language models, but with discipline
learned from prior failures and wins.

Context:

- Long language-model training runs were expensive and often produced
  autopilot, retention failure, or unconvincing gains.
- Small expert/puzzle specialists trained quickly and sometimes reached exact
  correctness.
- Cortex and Phi composition suggest that host language models may be useful as
  translators or narrators while small specialists provide precise reasoning.
- Every unattended iteration should produce something inspectable in one to
  five minutes: a checkpoint, metric, trace, benchmark result, proposal, or
  falsifier.

Question: what is the next concrete small-language-model experiment that should
be attempted, and how should it use the existing wins without repeating the
known failures?

Speakers should propose work that can be validated quickly and should state
whether it needs training, evaluation, archive/book work, or cluster dispatch.
