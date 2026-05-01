---
title: PTX engine
chapter: "02"
status: archival
summary: "CUDA/PTX engine era: archived narrative, still useful as infrastructure context for the current {{lab.workers}} worker live-control plane."
---

# Chapter 02 — PTX engine

**Status:** closed / archival (engine moved to `pod-archive` branch when
the H100/vast.ai pod era ended; daily-driver path moved back to
PyTorch + MPS).

**Dates:** 2026-04-24 → 2026-04-26

**Synopsis.** A hand-written PTX Mamba-3 engine for CUDA — owning
forward + backward + scheduler — built from scratch over ~3 days.
Achieved bit-parity with CPU on forward and 1e-6 parity on backward;
converged on parity training; ran ~14× faster than the same stream on
a PyTorch baseline; picked up a streaming batch protocol that made it
task-agnostic over the existing 30+ problems; learned to resume from
PyTorch `.pt` checkpoints; gained real (non-stub) Lion / Focal /
LabelSmooth kernels for the GA mutation surface; and shipped a
hot-plug `ptxd` daemon the orchestrator could submit jobs to without
restart.

**Why closed.** vast.ai instability ended the H100 pod era. The
engine remains usable when a CUDA box is provisioned; the source
is preserved on the `pod-archive` branch. Local files here:
`findings.md` and the historical `Cargo.lock` only.

**Findings.** [`findings.md`](findings.md) (128 KB; the ~28-entry
arc).
