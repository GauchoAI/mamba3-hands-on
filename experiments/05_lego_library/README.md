---
title: Lego library
chapter: "05"
status: archival
summary: "Composable specialist library: small rule modules, orchestration without retraining, and a bridge toward the current Lab ecology."
---

# Chapter 05 — Lego library

**Status:** archival.

**Dates:** 2026-04-28.

**Synopsis.** Five step-function specialists (Hanoi + GCD + Conway +
Bubble + Maze) at 2.2 k params total, ~5 s combined training time.
Composite tasks orchestrated by Python with zero retraining; the
"each rule is a Lego" pattern. A speed-regime study against NumPy
(NumPy 3-5× faster on simple CAs like Conway / WireWorld at 1000²×100;
gap shrinks with rule branchiness). Plus the **Light-CA** Lego —
1,009-param structured MLP that renders a Cornell-flat scene in 70 ms
by adapting path tracing into a teachable rule (5 materials × 4
directions × RGB), the first regression Lego and the first multi-channel
state. Hard gating in the orchestrator + soft MLP per category gave
byte-perfect rollouts on 3D Cornell (32 k voxels × 96 steps, max diff
0.0000) where soft gates compounded errors badly.

**Findings.** [`docs/findings/lego.md`](../docs/findings/lego.md).

**What's here.** Step-function specialists
(`bubble_step_function.py`, `conway_step_function.py`,
`gcd_step_function.py`, `light_step_function.py`,
`light_sh_step_function.py`, `maze_step_function.py`,
`wireworld_step_function.py`), their trainers (`train_*_step.py`,
`train_step_function.py`), the discoverers (`discover_*_step.py`),
the Cornell rendering / showdown scripts (`cornell_*.py`), the
speed-showdown comparisons (`conway_speed_showdown.py`,
`wireworld_speed_showdown.py`), and the universal-step exploration
(`exp_universal_step.py`). Cornell render PNGs included.
