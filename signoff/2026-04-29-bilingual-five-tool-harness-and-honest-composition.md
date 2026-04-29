# Signoff — 2026-04-29: bilingual 5-tool harness + honest composition

End-of-session doc for 2026-04-29. Full state of the harness work and
where it sits relative to the project's thesis.

---

## What shipped today

The harness from yesterday went from 3 tools / EN-only / regex argument
extraction / placeholder composite to:

- **5 specialist tools** registered: `hanoi_solver`, `gcd`, `gcdhanoi`,
  `fibonacci`, `factorial`.
- **Bilingual rendering** (EN + ES). The renderer LM was retrained on
  templates in both languages; at inference it picks one stochastically
  per call. Same payload, two output languages.
- **Honest composition** in `gcdhanoi`. Now actually invokes
  `hanoi_run` twice (each runs the GRU through every Hanoi move), then
  `gcd_run` — three real specialist calls chained at runtime, not the
  `math.gcd((1<<a)-1, (1<<b)-1)` shortcut that was there yesterday.
  The trace shows per-leg timing.
- **Lion optimizer** as the renderer's default. Inline 15-line class
  (sign-of-momentum updates, no extra dep). Reaches val_loss 0.0155 on
  the bilingual task in 600 steps; the AdamW single-template baseline
  reached 0.006 in 400 steps but on a 3× simpler target distribution.
  Lion delivered comparable convergence on the harder task at similar
  wall-clock.

| Component | Architecture | Params | Result |
|---|---|---|---|
| Router | Mamba-3 byte classifier (1 block, d=64), 5-way head | 45,589 | val_acc 100 % |
| Hanoi specialist | Order-invariant GRU | 45,318 | byte-perfect at any n ≤ 23 |
| Renderer (slot-fill) | Mamba-3 LM, bilingual templates | 74,400 | val_loss 0.0155 |
| Renderer (copy mech, alt path) | CopyMamba3LM | 78,625 | val_loss 0.0099 (digit-stop bug) |

Six demo prompts (mix of EN and ES inputs) all run end-to-end through
the LM with no template fallbacks. Three of the LM outputs land in
EN, three in ES — the model picks. Numbers are exact in every case;
the orchestrator does the substitution from the structured payload.

Demo run with the trained checkpoints:

```bash
.venv/bin/python assistant.py \
    --router-checkpoint checkpoints/tool_router_mamba3.pt \
    --renderer-checkpoint checkpoints/tool_renderer_mamba3.pt \
    "<prompt>"
```

| Input | Output |
|---|---|
| Solve Tower of Hanoi with 12 disks | The optimal solution to Tower of Hanoi with 12 disks requires 4,095 moves. |
| What is the gcd of 1729 and 1001? | El máximo común divisor de 1729 y 1001 es 91. |
| Compute the gcd of Hanoi 6 and Hanoi 9 | Hanoi(6) requiere 63 movimientos; Hanoi(9) requiere 511 movimientos; su mcd es 7. |
| fibonacci 25 | The 25-th Fibonacci number is 75,025. |
| What is 7!? | El factorial de 7 es 5,040. |
| el factorial de 6 | El factorial de 6 es 720. |

## Why this matters for the thesis

The thesis from the project's "why" section says:

> Language is a translation layer, not the reasoning substrate.

The slot-fill renderer literally implements that. The LM owns *shape*
(emits templates with `$`-placeholders); the orchestrator owns
*values* (deterministic substitution from the structured payload).
Adding a second output language without changing the orchestrator
demonstrates the boundary holds — the LM is doing language-form-only
generation, the values are completely separable.

The honest `gcdhanoi` composite is a small but real demonstration
that the harness composes *specialists* at runtime, not just
hand-coded helpers. Given a high-level prompt, three Mamba-3-class
models (router + GRU twice + render) run in series and the result
gets translated to the user's language by a fourth.

## What's still scaffolding (closing one, listing the rest)

The honest inventory the user asked for, after today's work:

**Closed today:**
- ~~`gcdhanoi` doesn't actually invoke the Hanoi GRU.~~ Done.

**Still scaffolding (deliberate, with upgrade paths):**
- **`gcd`, `fibonacci`, `factorial` are Python stdlib.** The Lego library
  (`discover_gcd_step.py`) and the FIB Mamba-3 LM (from
  `fib_validate.py` / `fib_decimal_validate.py`) exist and could be
  swapped in. Not done because the priority was harness coverage, not
  every-leg-must-be-learned.
- **Slot-fill itself.** The orchestrator does digit substitution
  because the LM can't reliably copy. CopyMamba3LM is the proper ML
  fix; it works but has a "when to stop copying" bug at 3-digit
  numbers (gets caught by the payload-fidelity guard).
- **Argument extraction is regex.** The router classifies the tool;
  a regex pulls integers from the prompt. Could be a small Mamba-3
  head; not done.

**Still hacks to address:**
- **Language matching.** Renderer picks EN or ES randomly. To match
  output language to input language we'd pass a detected `lang` field
  through the payload. ~50-line change.
- **Tool taxonomy is hardcoded.** `REGISTRY` has 5 entries; the harness
  dispatches among them. The thesis says the harness should *find*
  and *create* specialists at runtime; we don't.
- **Training data is synthetic.** Templates with random params, not
  real human prompts. Robust to OOD paraphrases within the template
  span; not robust beyond it.

## Speed observations

- **Lion vs AdamW** for the renderer: Lion lost the first 100 steps
  (cold start at val_loss 1.17 vs AdamW's lower start), but caught up
  by step 200 and converged to val_loss 0.0155 by step 600 on the
  bilingual task. AdamW on the single-template task hit 0.006 by step
  400. Hard to compare directly across task difficulty, but Lion is
  in the same ballpark.
- **PyTorch MPS pad-fallback** finding from yesterday holds — for our
  74k-param Mamba-3, MPS is only ~1.5× CPU at batch=32 and gets
  *slower* at batch=256 because the `F.pad` >3D fallback scales with
  batch. The proper Apple-Silicon GPU path is MLX or hand-written
  Metal kernels. Still queued.

## Pick up here

Roughly in priority order:

1. **Replace placeholder tools with their trained equivalents.** GCD
   step Lego, FIB Mamba-3 LM. Each is its own retraining + integration.
2. **Language matching in the renderer.** Detect input language at the
   router, pass `lang=es` or `lang=en` through the payload, retrain
   the renderer with the lang-conditional. The thesis test gets
   stronger when input EN → output EN, input ES → output ES.
3. **Fix CopyMamba3LM's "when to stop copying" bug.** Curriculum:
   pretrain with the gate forced to 0 for a few hundred steps so
   attention learns precise pointing first, then unfreeze. From the
   Pointer Networks literature.
4. **MLX port of `Mamba3Block`.** The big GPU lever. Documented in
   yesterday's signoff. Day-2 project, not the next-hour project.

## Files and commits today

**Modified:** `assistant.py` (added 2 tools + slot map + payload
strings + composite Hanoi calls), `train_tool_router.py` (5-way),
`train_tool_renderer.py` (bilingual + Lion).

**Commits, in order:**
- `2125067` expand harness: +fibonacci +factorial tools, bilingual templates, Lion optimizer
- `5e6aa13` bilingual 5-tool harness lands: 6/6 demos render cleanly through Mamba-3 LM
- `429c2c5` gcdhanoi: actually invoke the Hanoi GRU twice (close hack #1)

Plus this signoff doc.

## Long-context probe (revisit)

The probe from yesterday (`probe_copy_long_context.py`) showed that
the trained CopyMamba3LM has 4× zero-shot length generalization on
the simpler `hanoi_solver` task (trained at 256, works at prefix
1081), then degrades. The composite `gcdhanoi` task collapses with
any added filler. The architectural property (linear-memory SSM)
holds; the *trained representation* is bounded by what it saw during
training. The clean experiment to actually push the context-window
claim is to train at long `max_seq_len`. Queued behind items 1–3
above.
