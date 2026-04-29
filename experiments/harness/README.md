# experiments/harness

A first-class harness over a registry of specialist tools — the
"language as translation layer, not reasoning substrate" thesis made
concrete. Three Mamba-3-class models in series:

  natural language → router → specialist → renderer → natural language

Plus the orchestrator handles language detection, structured payload
construction, and deterministic slot-fill substitution.

## Files

- `assistant.py` — the harness itself: Tool registry, router/renderer
  hooks, language detector, slot-fill, payload-fidelity guard. The
  CLI you actually run.
- `train_tool_router.py` — Mamba-3 byte classifier (45,589 params,
  5-way head: hanoi_solver, gcd, gcdhanoi, fibonacci, factorial).
- `train_tool_renderer.py` — Mamba-3 LM template renderer (74,400
  params). Templates use `$N`, `$OPTIMAL`, etc. placeholders;
  orchestrator substitutes from payload. Lion optimizer by default.
- `train_tool_renderer_copy.py` — `CopyMamba3LM`, the proper
  Pointer-Networks-style copy mechanism (78,625 params). Alternative
  to slot-fill where the LM does its own digit copy via gate +
  attention into the prefix.
- `probe_copy_long_context.py` — length-generalization probe for the
  copy LM. Prepends random ASCII filler before the real payload and
  measures whether the SSM hidden state still encodes enough to
  produce correct copies at long prefixes.
- `_test_copy_inference.py` — small inspection helper that prints
  per-output-byte gate + attention-argmax decisions from a saved
  CopyMamba3LM checkpoint.
- `_path_shim.py` — single-line `sys.path` shim so scripts in this
  folder can import top-level modules (`mamba3_minimal`, `mamba3_lm`,
  `gcd_step_function`, etc.) without the repo being installed as a
  package.

## Run

Always **from the repo root** so the `checkpoints/...` paths resolve:

```bash
# Demo with all three Mamba-3 stages active:
.venv/bin/python experiments/harness/assistant.py \
    --router-checkpoint   checkpoints/tool_router_mamba3.pt \
    --renderer-checkpoint checkpoints/tool_renderer_mamba3.pt \
    "Solve Tower of Hanoi with 12 disks"

# Spanish input → Spanish output via _detect_lang() heuristic:
.venv/bin/python experiments/harness/assistant.py \
    --router-checkpoint   checkpoints/tool_router_mamba3.pt \
    --renderer-checkpoint checkpoints/tool_renderer_mamba3.pt \
    "Resuelve la Torre de Hanoi con 12 discos"

# Train a fresh router:
.venv/bin/python experiments/harness/train_tool_router.py --steps 1500 --device cpu

# Train a fresh renderer (slot-fill, with Lion):
.venv/bin/python experiments/harness/train_tool_renderer.py --steps 600 --optimizer lion --device cpu

# Train the copy-mechanism alternative:
.venv/bin/python experiments/harness/train_tool_renderer_copy.py --steps 2000 --device cpu

# Inspect what the copy LM is actually doing:
.venv/bin/python experiments/harness/_test_copy_inference.py
```

## Where this came from

Built across 2026-04-28 → 2026-04-29. The chronology and design
decisions are in `signoff/2026-04-28-mlx-port-and-experiments-queued.md`
and `signoff/2026-04-29-bilingual-five-tool-harness-and-honest-composition.md`.

Lab-notebook entries (each commit-by-commit narrative) are in
`findings.md` at the repo root, under the harness-related entries.
