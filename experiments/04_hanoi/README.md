---
title: Hanoi
chapter: "04"
status: archival
summary: "Unbounded-computation chapter: Hanoi, LoopCounter, and byte-perfect extrapolation, cross-linked with the live task and model surfaces where available."
---

# Chapter 04 — Hanoi

**Status:** archival.

**Dates:** 2026-04-27 → 2026-04-28.

**Synopsis.** A multi-phase line of attack on Tower of Hanoi as a
testbed for "true unbounded computation" in a small SSM. Started by
diagnosing the bounded-counter ceiling (the model isn't doing
last-digit attention; it's running out of register depth), passed
through the EOS-bias-gating breakthrough (HANOIBIN to n=256, FIB-unary
to F=6765 = 123× extrapolation, FIB-decimal F(40) byte-perfect via
per-position iter_token), arrived at the parameter-free LoopCounter
(HANOIBIN extended to n=100,000 = 5,000× extrapolation, byte-perfect),
then pivoted to **tool use over neural memory**: a Python state-tool
plus feedback embedding produces a perfect Hanoi step-function in 1.9
seconds of training and extends to n=12 (4,095 moves) with no learned
register bank.

**Findings.** [`docs/findings/hanoi.md`](../docs/findings/hanoi.md).

**What's here.** All the discovery / training / validation scripts
for Hanoi: `discover_hanoi_*.py`, `hanoi_*.py`, `train_hanoi_*.py`,
`length_gen_hanoi*.py`, the role / repeat-count diagnostic
(`analyze_hanoi_errors.py`, `analyze_role_errors.py`,
`analyze_repeat_count.py`, `timeline_repeat_count.py`).
