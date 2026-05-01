---
title: Raw trace stack findings
chapter: "12"
status: active
summary: "A tiny GRU learns stack validity from raw OPEN/CLOSE/END prefixes under the one-minute rule."
---

# Raw Trace Stack Findings

This chapter removes the explicit depth feature from Chapter 11. The model sees
only a padded prefix of emitted roles plus a candidate next role, then predicts
whether that candidate is valid.

The goal is not final language. The goal is to test whether state can begin to
move from hand-coded features into a learned transition memory.

