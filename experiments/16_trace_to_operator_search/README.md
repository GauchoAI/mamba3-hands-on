---
title: Trace-to-operator search
chapter: "16"
status: active
lab_book: subsection
summary: "Search over candidate state encodings and keep the smallest one that verifies by rollout."
---

# Chapter 16 — Trace-To-Operator Search

Try small feature subsets for a compare/swap policy and select the first
operator that reaches both training accuracy and rollout verification.

```bash
.venv/bin/python experiments/16_trace_to_operator_search/trace_to_operator_search.py
```
