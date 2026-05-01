---
title: Operator curriculum intro
chapter: "12"
status: active
sections: true
summary: "The previous one-minute operator steps are kept here as scaffolding, not as separate claims."
---

# Chapter 12 - Operator Curriculum Intro

**Status:** active scaffolding. This chapter collects the one-minute operator
curriculum pieces that were previously promoted too aggressively.

The evidence is useful, but modest: if the Lab exposes a good state/action
interface, tiny policies can become reliable quickly. That does not yet prove
that the model discovers the operator state by itself.

## What This Chapter Is

This is an introductory map over the small operator pieces:

| Section | Script | Role |
| --- | --- | --- |
| Raw trace stack | `experiments/12_raw_trace_stack/raw_trace_stack.py` | Trace-derived stack predicate |
| Multi-surface stack | `experiments/13_multi_surface_stack/multi_surface_stack.py` | Same role policy over several surfaces |
| Guided decoding | `experiments/14_operator_guided_decoding/operator_guided_decoding.py` | Operator as per-step decoder constraint |
| Comparator transfer | `experiments/15_comparator_transfer/comparator_transfer.py` | Tiny compare/swap policy |
| Trace search | `experiments/16_trace_to_operator_search/trace_to_operator_search.py` | Search over candidate state encodings |
| Registry | `experiments/17_operator_registry/operator_registry.py` | Evidence registry plumbing |
| Composition | `experiments/18_operator_composition/operator_composition.py` | Stack parser plus comparator |
| Language role trace | `experiments/19_language_to_role_trace/language_to_role_trace.py` | Text command to structured role task |
| Runtime episode | `experiments/20_runtime_learning_episode/runtime_learning_episode.py` | One tiny rule learned during a run |
| Lab organ demo | `experiments/21_lab_organ_demo/lab_organ_demo.py` | Evidence bundle |

## Honest Boundary

These sections are not the big idea. Most of the wins come from structured
features, hand-built harnesses, or tiny MLPs learning nearly direct predicates.
They belong in one chapter because they prepare vocabulary and plumbing.

The next real claim is stricter:

```text
raw prefix trace
  -> recurrent learner builds hidden state
  -> candidate-role classifier reads that hidden state
  -> held-out longer traces test whether a latent operator state emerged
```

That is the first place where we ask the model to recover the state instead of
handing it the state.

## Run The Intro Sections

```bash
.venv/bin/python experiments/12_raw_trace_stack/raw_trace_stack.py
.venv/bin/python experiments/13_multi_surface_stack/multi_surface_stack.py
.venv/bin/python experiments/14_operator_guided_decoding/operator_guided_decoding.py
.venv/bin/python experiments/15_comparator_transfer/comparator_transfer.py
.venv/bin/python experiments/16_trace_to_operator_search/trace_to_operator_search.py
.venv/bin/python experiments/18_operator_composition/operator_composition.py
.venv/bin/python experiments/19_language_to_role_trace/language_to_role_trace.py
.venv/bin/python experiments/20_runtime_learning_episode/runtime_learning_episode.py
.venv/bin/python experiments/17_operator_registry/operator_registry.py
.venv/bin/python experiments/21_lab_organ_demo/lab_organ_demo.py
```

