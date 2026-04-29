# Handoff — 2026-04-28 EOD

State of the project at the end of the harness session. If you're picking
this up tomorrow, read top-to-bottom; the "Pick up here" line is the
single most important thing.

---

## What shipped today

A working **three-stage Mamba-3 harness** that takes a natural-language
prompt, routes to a specialist, runs the inner computation, and renders
the answer. Three Mamba-3-class models in series, ~165k parameters
total, plus the 45,318-param order-invariant GRU specialist.

| Stage | Architecture | Params | Status | Checkpoint |
|---|---|---|---|---|
| Router | Mamba-3 byte classifier (1 block, d=64) | 45,459 | val_acc 100 % | `checkpoints/tool_router_mamba3.pt` |
| Specialist | Order-invariant GRU (Hanoi solver) | 45,318 | optimal at n≤23 | `checkpoints/hanoi_invariant_gru_offtrace.pt` |
| Renderer (slot-fill) | Mamba-3 LM emitting templates | 74,400 | val_loss 0.006 | `checkpoints/tool_renderer_mamba3.pt` |
| Renderer (copy mech) | Pointer-Networks-on-Mamba-3 | 78,625 | training tonight | `checkpoints/tool_renderer_copy.pt` |

Five demo prompts run end-to-end through the slot-fill pipeline with
no fallbacks:

```
> Solve Tower of Hanoi with 12 disks
The optimal solution to Tower of Hanoi with 12 disks requires 4,095 moves.

> What is the gcd of 462 and 252?
The greatest common divisor of 462 and 252 is 42.

> Compute the gcd of Hanoi 6 and Hanoi 9
Hanoi(6) needs 63 moves; Hanoi(9) needs 511 moves; their gcd is 7.

> I'd like the move count for an 8-disk tower puzzle    # OOD paraphrase
The optimal solution to Tower of Hanoi with 8 disks requires 255 moves.

> common divisor of 144 and 60 please                   # OOD paraphrase
The greatest common divisor of 144 and 60 is 12.
```

Run any of these with:

```bash
.venv/bin/python assistant.py \
    --router-checkpoint   checkpoints/tool_router_mamba3.pt \
    --renderer-checkpoint checkpoints/tool_renderer_mamba3.pt \
    "Solve Tower of Hanoi with 12 disks"
```

The router and renderer use Mamba-3; the specialist is a GRU. Trace
output (`--quiet` off) shows per-stage probabilities, args, and timing.

## What's in flight

A **CopyMamba3LM** — proper Pointer-Networks-style copy mechanism on
top of Mamba-3 — is training on `m4-mini` overnight (3000 steps, batch
64, CPU, ETA ~90 min from launch at 23:18). Goal: a renderer that
copies digits *internally* without the orchestrator's `SLOT_MAP`. The
trace will show per-output-byte gate values and attention-argmax
positions — visible interpretability of when the model copies vs
generates.

Architecture sketch in `train_tool_renderer_copy.py`:
- Standard Mamba-3 stack (2 layers, d=64) over the full sequence.
- At each output step, three things in parallel:
  - `p_vocab(b)` — vocab distribution from the LM head
  - `p_copy_attn(s)` — softmax attention from h_t to all prefix hidden
    states, masked to prefix only (this is "where to copy from")
  - `gate ∈ [0, 1]` — sigmoid head; 1 = generate from vocab, 0 = copy
- Final mixture: `p_final(b) = gate · p_vocab(b) + (1 − gate) · p_copy(b)`
- Standard NLL loss on `p_final(target)` at every answer position.

Smoke at 100 steps already shows the right qualitative behavior — at
the digit position in the answer, the gate dropped to 0.024 and the
attention pointed at a `'1'` byte in the prefix. Pointer training is
notoriously slow due to the chicken-and-egg between gate and attention,
so 3000 steps required.

Monitor via:
```bash
tail -f /tmp/cluster_dispatch_logs/copy-train.log
```

## Pick up here

**1. (in flight) Copy mechanism training finishes overnight.**
Check `/tmp/cluster_dispatch_logs/copy-train.log`. If `best val loss`
is below ~0.2 and the sample-with-trace shows clean digit copies (gate
near 0 at digit positions, attn pointing at the right prefix bytes),
the experiment succeeded. Then:

  a. Rsync the checkpoint back:
     ```bash
     rsync -av miguel_lemos@192.168.0.170:~/mamba3-hands-on/checkpoints/tool_renderer_copy.pt checkpoints/
     ```
  b. Wire `CopyMamba3LM` into `assistant.py` as a third renderer mode
     (alongside `template` and `slot-fill`) — `--renderer-mode=copy
     --renderer-checkpoint=checkpoints/tool_renderer_copy.pt`.
  c. Add a `--show-copy-trace` flag that prints the per-byte gate +
     attn-argmax decisions in the harness output. This is the
     *interpretability win* — the user explicitly asked to *see* this.

If the copy mechanism didn't converge well at 3000 steps, the next
step is curriculum: pre-train with the gate forced to 0 (copy-only)
for a few hundred steps, then unfreeze. Pointer literature does this.

**2. (next experiment) Mamba+attention hybrid block.**
Drop a single attention head into one of the Mamba-3 blocks in the
renderer (or the router). This is the more "general" answer to copy —
attention solves it for free, no gate to learn. Compare param count
and quality vs the pure-copy mechanism. The user is keen on seeing
both work.

**3. (later) Additional Legos.**
The Lego library currently has Hanoi + GCD + Conway + Bubble + Maze +
Light-CA. Adding a sort step Lego that drives sort_suite at runtime,
or a Fibonacci-step Lego, would round out the demo surface for the
harness. The router will need a few more keyword/example phrasings;
add a tool entry in `assistant.py`.

## Open risks / known issues

- `cluster_sync.py` excludes `checkpoints/` (sensible default for the
  100s of `.pt`s in there); specialist + renderer + router checkpoints
  have to be `rsync`'d explicitly when moving between nodes. Cleanest
  fix is a `specialist_checkpoint` field on `Tool` that triggers a
  per-tool sync; deferred.
- Background Bash with `... 2>&1 | tail -N` strands Python processes
  (block-buffered stdout never flushes). Use `python -u`, no pipe.
  Documented in `feedback_background_python_output.md`.
- For tiny Mamba-3 models on M4 Mac mini, PyTorch MPS hits a
  CPU-fallback on `F.pad` (>3D constant padding). At our scale CPU is
  within ~1.5× of MPS; bigger batches make MPS *worse*, not better.
  The proper Apple-Silicon GPU path is MLX or hand-written Metal
  kernels — same pattern as the PTX engine on H100. Documented in
  `feedback_mps_pad_fallback.md`.

## Files touched today

| File | Purpose |
|---|---|
| `assistant.py` | Three-stage harness with Tool registry, router/renderer hooks, slot-fill substitution, payload-fidelity guard |
| `train_tool_router.py` | Mamba-3 byte classifier router trainer |
| `train_tool_renderer.py` | Mamba-3 LM template renderer trainer (slot-fill) |
| `train_tool_renderer_copy.py` | CopyMamba3LM (Pointer-Networks-style) trainer — running overnight |

Plus updates to `findings.md`, `index.md`, `index.html` (regenerated
via `md2html.js`), and `docs/introduction/index.html` (the WSJ-styled
public page).

Today's commits, in order:
- `dc5144b` first-class harness over a Tool registry
- `af99190` Mamba-3 router replaces regex
- `1359194` renderer hook + Mamba-3 LM trainer
- `afa4dd2` renderer LM + payload-fidelity guard (digit-copy failure)
- `26d0ff7` slot-fill renderer (sidesteps the SSM copy weakness)
- `17b982a` CopyMamba3LM (proper ML answer)
