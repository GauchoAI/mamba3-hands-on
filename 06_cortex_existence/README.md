# Chapter 06 — Cortex existence proof

**Status:** archival.

**Dates:** 2026-04-28.

**Synopsis.** The original cortex existence proof: a 772-param counter
primitive embedded in the residual stream of a 151 k Mamba-3 LM
trained on `*N:aN` only (synthetic counting-only LM, N ≤ 30) achieved
byte-perfect counting at N=500 — **16.7× past the training distribution**
— under hard-gate inference. First demonstration that a primitive can
extend a host LM's behaviour past its training distribution when the
host has been pre-conditioned to encode the primitive's domain.

A subsequent refactor made `CortexLM` primitive-agnostic via the
pluggable `Primitive` base class.

**Findings.** [`docs/findings/cortex.md`](../docs/findings/cortex.md)
("Cortex: a 772-param counter primitive lets a 151k Mamba LM count
byte-perfect …" entry).

**What's here.** The synthetic-LM existence-proof scripts that lived
at root: `modular_counting.py` (the counting-only LM training),
`compose_logic_gate.py` (early multi-primitive composition probe).
The `Primitive` base class itself lives in `cortex_counting.py` at
the project root since multiple chapters import it.
