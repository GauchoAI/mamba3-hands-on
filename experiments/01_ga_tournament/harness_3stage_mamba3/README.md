# experiments/harness_3stage_mamba3

A first-class harness over a registry of specialist tools — three
Mamba-3-class models composed in series, embodying the project's
"language as translation layer, not reasoning substrate" thesis.

The naming convention is `experiments/harness_<distinctive>/` so future
harness variants (different routing strategy, voting ensembles,
retrieval-augmented, agentic, etc.) slot in as siblings. This one is
the **3-stage Mamba-3** baseline.

---

## Why we built this

Transformer-class models have **quadratic memory in context length**.
The community spends large amounts of energy on context engineering —
at 1M tokens, even the best frontier setups need expensive compaction
strategies (e.g., calling a frontier model just to track dependency
graphs across a long session). It scales badly.

When the **Mamba-3 paper** (Lahoti et al., ICLR 2026) landed promising
linear memory and improved precision via data-dependent RoPE on B and
C, this project started two days later. The bet: a linear-memory
recurrent SSM, paired with the right composition layer, could replace
the quadratic context window with something cheaper and more honest.

Underneath the architectural bet are four convictions:

- **Overparameterized models memorize. Constraining size forces the
  pattern to surface.** A model with enough capacity always finds a
  way to look up the answer rather than learn the rule. We
  deliberately keep models small so the gradient has nowhere to hide.
  Empirically, the minimal model is the one that extends: HANOIBIN
  n=100,000 byte-perfect (5,000× extrapolation), the 45,318-param
  Hanoi GRU at n=23 with 100% prediction. We've made the bet pay off
  several times in this repo.
- **Language is the translation layer, not the reasoning substrate.**
  Treat natural language as a thin layer between humans and machines
  (and between machines), not as the medium reasoning happens in. The
  reasoning is an inner computation; the language is the API.
- **Composable, not monolithic.** Instead of one large model that
  pretends to be intelligent because it has memorized all the
  patterns, a constellation of minimal experts collaborating at
  runtime via a harness. Same family as David Silver's new
  super-learner work — interact with structured environments where
  the answer is *checkable*, not memorize a static corpus.
- **The harness is a first-class citizen.** Tool-calling shouldn't be
  bolted on around a model; it should be an inner primitive.

This experiment is the smallest concrete demonstration of those
convictions. Same payload, two output languages, picked by an
orchestrator-detected language flag and emitted by the LM. Reasoning
happens in inner specialists; values flow through structured payloads;
the LM produces only language form.

---

## What's in the folder

The pipeline is **natural language → router → specialist → renderer →
natural language**, with the orchestrator handling language detection
and deterministic value substitution.

| Stage | File | Architecture | Params | Result |
|---|---|---|---|---|
| Router | `train_tool_router.py` | Mamba-3 byte classifier (1 block, d=64), 5-way head | 45,589 | val_acc 100 % |
| Specialist (Hanoi) | `assistant.py` (loads `discover_hanoi_invariant.HanoiInvariantGRU`) | Order-invariant GRU over disk-peg sequence | 45,318 | byte-perfect at any n ≤ 23 |
| Specialist (GCD) | `assistant.py` (loads `train_gcd_step.GCDStepMLP`) | 4-bit-state MLP iterated by orchestrator | 331 | exact via subtraction Euclidean |
| Renderer (default) | `train_tool_renderer.py` | Mamba-3 LM with placeholder templates, lang-conditional | 74,400 | val_loss 0.0000 |
| Renderer (alt) | `train_tool_renderer_copy.py` | `CopyMamba3LM`: Pointer Networks on Mamba-3 | 78,625 | val_loss 0.0099, has digit-stop bug |

Five tools currently registered: `hanoi_solver`, `gcd`, `gcdhanoi`,
`fibonacci`, `factorial`.

Other files:
- `assistant.py` — the orchestrator. Tool registry, language detector,
  router/renderer hooks, slot-fill substitution, payload-fidelity
  guard. The CLI you actually run.
- `probe_copy_long_context.py` — length-generalization probe for the
  copy-mechanism renderer.
- `_test_copy_inference.py` — small inspection helper that prints
  per-output-byte gate + attention-argmax decisions from a saved
  `CopyMamba3LM` checkpoint.
- `_path_shim.py` — single-line `sys.path` shim so scripts in this
  folder can import top-level modules (`mamba3_minimal`, `mamba3_lm`,
  `gcd_step_function`, etc.) without packaging the repo.

### Two design choices worth noting

**Slot-fill renderer (default).** The LM emits templates with named
placeholders (`$N`, `$OPTIMAL`, `$A`, `$B`, `$GCD`, `$MOVES_A`,
`$MOVES_B`, `$RESULT`); the orchestrator substitutes from the
structured payload. The LM never sees a digit during training. This
sidesteps the canonical Mamba selective-copy weakness: a small SSM
compresses the prefix into a fixed-size hidden state and can't
reliably copy specific digits from arbitrary positions. The
boundary is clean — LM owns shape, orchestrator owns values.

**Pointer Networks alternative.** `train_tool_renderer_copy.py`
implements the proper ML answer (Vinyals 2015 / Gu 2016) — at every
output position the LM emits a vocab distribution, an attention
distribution over prefix positions, and a gate ∈ [0, 1] that mixes
them. The model learns *when* to copy (gate near 0 at digit
positions) and *where to point* (attention peaks on the right prefix
byte). It works qualitatively (gate=0.000 at digit positions, attn
pointing at the right `'4'`/`'6'`/`'2'` bytes) but has a
"when-to-stop-copying" bug at 3-digit numbers. Caught by the
payload-fidelity guard with a fallback to template.

---

## How to run

Always **from the repo root** so checkpoint paths (`checkpoints/...`)
resolve.

### Demo (uses pre-trained checkpoints)

```bash
.venv/bin/python experiments/harness_3stage_mamba3/assistant.py \
    --router-checkpoint   checkpoints/tool_router_mamba3.pt \
    --renderer-checkpoint checkpoints/tool_renderer_mamba3.pt \
    "Solve Tower of Hanoi with 12 disks"
```

Spanish input → Spanish output (via the `_detect_lang()` heuristic on
trigger words):

```bash
.venv/bin/python experiments/harness_3stage_mamba3/assistant.py \
    --router-checkpoint   checkpoints/tool_router_mamba3.pt \
    --renderer-checkpoint checkpoints/tool_renderer_mamba3.pt \
    "Resuelve la Torre de Hanoi con 12 discos"
```

### Train fresh checkpoints

```bash
# Router (Mamba-3 byte classifier, ~45k params, 5-way)
.venv/bin/python experiments/harness_3stage_mamba3/train_tool_router.py \
    --steps 1500 --batch 128 --device cpu \
    --save-to checkpoints/tool_router_mamba3.pt

# Slot-fill renderer (Mamba-3 LM with placeholder templates, lang-conditional)
.venv/bin/python experiments/harness_3stage_mamba3/train_tool_renderer.py \
    --steps 600 --batch 32 --lr 3e-3 --optimizer lion --device cpu \
    --save-to checkpoints/tool_renderer_mamba3.pt

# Copy-mechanism renderer (Pointer Networks on Mamba-3)
.venv/bin/python experiments/harness_3stage_mamba3/train_tool_renderer_copy.py \
    --steps 2000 --batch 64 --device cpu \
    --save-to checkpoints/tool_renderer_copy.pt
```

### Inspect what the copy LM is doing

```bash
# Prints per-byte gate + attn-argmax decisions for sample payloads
.venv/bin/python experiments/harness_3stage_mamba3/_test_copy_inference.py
```

### Probe long-context behavior

```bash
# Prepend filler before a real payload, watch the SSM hidden state
# generalize (or not) past training length
.venv/bin/python experiments/harness_3stage_mamba3/probe_copy_long_context.py
```

---

## Actual output (captured 2026-04-29)

Six paired prompts (EN / ES) through the trained checkpoints:

```
=== EN: Hanoi ===
Router:   Mamba-3 (45,589 params, val_acc=100.0000%)  via checkpoints/tool_router_mamba3.pt
Renderer: Mamba-3 LM (74,400 params, byte-level AR)  via checkpoints/tool_renderer_mamba3.pt
Mamba-3 specialist harness — type 'quit' to exit.
Registered tools:
  hanoi_solver    Solve Tower of Hanoi for any number of disks, optimally.
  gcd             Greatest common divisor of two integers.
  gcdhanoi        GCD of the optimal-move counts of two Hanoi instances.
  fibonacci       Nth Fibonacci number (F(0)=0, F(1)=1).
  factorial       Factorial n! of a non-negative integer.

> Solve Tower of Hanoi with 12 disks
  [trace] router(mamba3): probs={hanoi_solver=1.000, gcd=0.000, gcdhanoi=0.000, fibonacci=0.000, factorial=0.000}; chose=hanoi_solver; args={'n': 12}  lang=en
  [tool ] calling Hanoi GRU via hanoi_solver({'n': 12})
  [spec ] hanoi_invariant_gru_offtrace (45,318 params, order-invariant GRU)  timing=1523 ms
The optimal solution to Tower of Hanoi with 12 disks requires 4,095 moves.

=== ES: Hanoi ===
> Resuelve la Torre de Hanoi con 12 discos
  [trace] router(mamba3): probs={hanoi_solver=0.998, gcd=0.000, gcdhanoi=0.001, fibonacci=0.000, factorial=0.001}; chose=hanoi_solver; args={'n': 12}  lang=es
  [tool ] calling Hanoi GRU via hanoi_solver({'n': 12})
  [spec ] hanoi_invariant_gru_offtrace (45,318 params, order-invariant GRU)  timing=1500 ms
La solución óptima de la Torre de Hanoi con 12 discos requiere 4,095 movimientos.

=== EN: GCD ===
> What is the gcd of 1729 and 1001?
  [trace] router(mamba3): probs={hanoi_solver=0.000, gcd=0.996, gcdhanoi=0.003, fibonacci=0.001, factorial=0.001}; chose=gcd; args={'a': 1729, 'b': 1001}  lang=en
  [tool ] calling GCD tool via gcd({'a': 1729, 'b': 1001})
  [spec ] GCD step Lego (331 params, 6 iter subtraction steps)  timing=2 ms
The greatest common divisor of 1729 and 1001 is 91.

=== ES: GCD ===
> Cuál es el máximo común divisor de 1729 y 1001?
  [trace] router(mamba3): probs={hanoi_solver=0.002, gcd=0.991, gcdhanoi=0.000, fibonacci=0.005, factorial=0.003}; chose=gcd; args={'a': 1729, 'b': 1001}  lang=es
  [tool ] calling GCD tool via gcd({'a': 1729, 'b': 1001})
  [spec ] GCD step Lego (331 params, 6 iter subtraction steps)  timing=1 ms
El máximo común divisor de 1729 y 1001 es 91.

=== EN: Composite gcdhanoi ===
> Compute the gcd of Hanoi 6 and Hanoi 9
  [trace] router(mamba3): probs={hanoi_solver=0.003, gcd=0.006, gcdhanoi=0.990, fibonacci=0.001, factorial=0.001}; chose=gcdhanoi; args={'a': 6, 'b': 9}  lang=en
  [tool ] calling composite: Hanoi×GCD via gcdhanoi({'a': 6, 'b': 9})
  [spec ] composite: hanoi_solver(a=6, 25ms) + hanoi_solver(b=9, 190ms) + gcd  timing=217 ms
Hanoi(6) needs 63 moves; Hanoi(9) needs 511 moves; their gcd is 7.

=== ES: Composite gcdhanoi ===
> Calcula el mcd de Hanoi 6 y Hanoi 9
  [trace] router(mamba3): probs={hanoi_solver=0.196, gcd=0.007, gcdhanoi=0.595, fibonacci=0.014, factorial=0.188}; chose=gcdhanoi; args={'a': 6, 'b': 9}  lang=es
  [tool ] calling composite: Hanoi×GCD via gcdhanoi({'a': 6, 'b': 9})
  [spec ] composite: hanoi_solver(a=6, 26ms) + hanoi_solver(b=9, 188ms) + gcd  timing=215 ms
Hanoi(6) requiere 63 movimientos; Hanoi(9) requiere 511 movimientos; su mcd es 7.
```

What the trace lines mean:

- **`[trace] router(mamba3): probs={...}; chose=...; args={...}  lang=...`** —
  the byte-level Mamba-3 classifier's softmax over the 5 tools, the
  argmax pick, the regex-extracted args, and the language detected
  by the orchestrator's heuristic.
- **`[tool] calling <label> via <name>(<args>)`** — which specialist
  the registry routes to.
- **`[spec] <human label> timing=<ms>`** — how long the inner
  computation actually took, with the trained-model param count.
  For the composite case it shows the timings of each leg of the
  chain.
- **The final line** is the rendered answer, in the matched language,
  with values substituted by the orchestrator from the structured
  payload.

### What this proves

Five learned models compose at runtime to answer the composite
prompts:

  router (45,589 params)
   → Hanoi GRU (45,318 params, called twice in `gcdhanoi`)
   → GCD step Lego (331 params)
   → renderer (74,400 params)

≈ 211k learned parameters total in the chain, plus a 4-byte
language-detection heuristic and a regex argument extractor. Two
output languages from the same payload structure. Numbers exact in
every case.

The thesis is now testable end-to-end: same content, two output
languages, picked by the orchestrator-detected language and emitted
by the LM. Language is the translation layer; values come from the
orchestrator-detected language flag + slot-fill substitution; the LM
never tries to do arithmetic.

---

## Where this came from

Built across **2026-04-28 → 2026-04-29**. The chronology, design
decisions, and the failure-then-fix arc on the renderer are in:

- `signoff/2026-04-28-mlx-port-and-experiments-queued.md`
- `signoff/2026-04-29-bilingual-five-tool-harness-and-honest-composition.md`

Lab-notebook entries (commit-by-commit narrative) are in `findings.md`
at the repo root, under the harness-related entries.

---

## What's still scaffolding

Honestly listed so future contributors don't think the system is more
finished than it is:

- **Argument extraction is regex.** The router classifies the tool;
  a regex pulls integers from the prompt. A small Mamba-3 head could
  learn that too; it just isn't done.
- **Fibonacci and factorial are Python iterative loops.** Labeled as
  "placeholder for the FIB Mamba-3 LM". The FIB models in the repo
  are trajectory-supervised and need oracle hints during decoding,
  so they don't fit cleanly here. A real Fibonacci specialist would
  be its own training run.
- **`_detect_lang()` is a heuristic.** Spanish trigger words via
  string matching. A real Mamba-3 language classifier would be
  basically free given the router architecture.
- **Tool taxonomy is hardcoded.** `REGISTRY` has 5 entries; the
  harness dispatches among them. The thesis says the harness should
  *find* and *create* specialists at runtime; we don't yet.
- **CopyMamba3LM digit-stop bug.** The Pointer Networks renderer has
  a "when to stop copying" bug at 3-digit numbers (appends an extra
  copy). Fix is curriculum (gate-forced-to-0 pretrain).
- **Synthetic training data.** Templates with random params, not real
  human prompts.
