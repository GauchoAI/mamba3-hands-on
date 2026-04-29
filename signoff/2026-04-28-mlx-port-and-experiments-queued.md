# Signoff — 2026-04-28: MLX port + experiments queued

State of the project at the end of session 2026-04-28. Two items
matter most for tomorrow:

1. **MLX port is now on the table.** Today's GPU benchmarking surfaced
   that PyTorch's MPS backend is not a real GPU path for our model
   class — bigger batches make it *slower*, not faster. The honest
   GPU answer for Apple Silicon is MLX (or hand-written Metal), the
   same pattern as the PTX engine on H100.

2. **Two experiments are queued behind the MLX decision.** A
   Pointer-Networks-style copy mechanism is training overnight on the
   m4-mini; a Mamba+attention hybrid block is the next experiment
   after that. Both are architectural answers to the canonical Mamba
   selective-copy weakness we hit today.

Plus a working three-stage Mamba-3 harness that demonstrates the
project's thesis end-to-end. That's the headline ship of the day.

---

## 1. MLX port decision (the new lever)

**What surfaced today.** Training the renderer LM on m4-mini, I dispatched
the same job to CPU and to MPS to compare. Result:

  - MPS at batch=32: ~1.5× faster than CPU
  - MPS at batch=256: *slower* than batch=32 CPU

Cause is a known PyTorch limitation: `F.pad` >3D triggers a CPU view-ops
fallback (`UserWarning: MPS: The constant padding of more than 3
dimensions is not currently supported natively. It uses View Ops
default implementation to run.`). The fallback cost scales with batch
size, so amortizing kernel launch overhead by batching — the standard
GPU lever — actively hurts us.

**The MLX port.** MLX is Apple's native ML framework. It compiles
directly to Metal, doesn't hit PyTorch's `F.pad` fallback, has
operator fusion for transformer/SSM patterns, and zero-copies between
CPU and GPU on Apple Silicon's unified memory. For a small model
(74k params, 256-byte sequences) we're nowhere near GPU compute
bound, so the gain isn't 10× — but the *architecture* is right and
the path scales as our models grow.

This is the same pattern we hit on H100: when PyTorch's precision
broke our training, we wrote our own kernels in PTX (`pod-archive`
branch). The Apple-Silicon equivalent is MLX or hand-written Metal,
and either is a real engineering project, not a one-line flag.

**Decision for tomorrow.** Don't start the port yet — finish the two
queued experiments first (they're scoped and educational on their own
terms). After that, port `mamba3_minimal.Mamba3Block` to MLX as a
standalone module, benchmark forward + backward against the PyTorch
reference for bit parity, then port the rest of the stack. Save the
PTX engine pattern as the model: layered phases, parity-gated
between phases.

Documented in memory as `feedback_mps_pad_fallback.md`.

## 2. Experiments queued

### 2a. Copy mechanism (in flight overnight)

**Why it exists.** Today's renderer LM (74,400-param Mamba-3 LM)
fluently produced template language but consistently dropped/swapped
the specific digits when asked to copy them from arbitrary prefix
positions. This is the canonical Mamba selective-copy failure — a
small SSM compresses the prefix into a fixed-size hidden state and
loses arbitrary positions. We've seen this throughout the repo:
token-stream Hanoi was unblocked only when we added EOS-bias gating +
parameter-free `LoopCounter`; HANOIBIN n=100k worked once we stopped
trying to make the SSM count.

**The textbook ML fix: Pointer Networks (Vinyals 2015) / CopyNet (Gu
2016).** At each output position, the model produces three things in
parallel:

  - `p_vocab(b)` — vocab distribution from the LM head
  - `p_copy_attn(s)` — softmax attention from current hidden state
    (query) to all prefix hidden states (keys), masked to prefix only
  - `gate ∈ [0, 1]` — sigmoid head; 1 = generate from vocab, 0 = copy

Final mixture: `p(b) = gate · p_vocab(b) + (1 − gate) · p_copy(b)`.
Standard NLL loss on `p(target)`. The model learns *when* to copy
(gate goes near 0 at digit positions) and *where to point* (attention
peaks on the right prefix byte), jointly with the LM.

**File.** `train_tool_renderer_copy.py`. Implements `CopyMamba3LM`
with the architecture above (~78,625 params, only ~4k more than the
plain LM for the copy heads). Smoke at 100 steps shows the right
qualitative behavior already:

```
payload: hanoi_solver|n=12|optimal=4095|...
trace: pos 6: char='1' gate=0.024 pointed at prefix idx 15 ('1')
```

The gate dropped to 0.024 (almost full copy) and the attention
pointed at a `'1'` byte in the prefix. Pointer mechanism is *learning
to point*. val_loss at 100 is still 2.6 because pointer training is
notoriously slow (chicken-and-egg between gate and attention) — 3000
steps required for real quality.

**Status.** Training on m4-mini at batch=64 / 3000 steps via
`cluster_dispatch.py`. ETA ~90 min from launch at 23:18.

**Pick up here.** Check `/tmp/cluster_dispatch_logs/copy-train.log`.
If `best val loss < 0.2` and the sample-with-trace shows clean digit
copies (gate near 0 at digit positions, attn pointing at the right
prefix bytes), the experiment succeeded. Then:

```bash
rsync -av miguel_lemos@192.168.0.170:~/mamba3-hands-on/checkpoints/tool_renderer_copy.pt checkpoints/
```

…and wire `CopyMamba3LM` into `assistant.py` as a third renderer mode
(`--renderer-mode=copy`). Add a `--show-copy-trace` flag that prints
per-byte gate + attn-argmax decisions; this is the interpretability
win the user explicitly asked to *see*.

If 3000 steps wasn't enough, the standard fix from the Pointer
Networks literature is curriculum: pretrain with the gate forced to 0
(copy-only) for a few hundred steps, then unfreeze.

### 2b. Mamba+attention hybrid block (next)

**Why.** Attention solves selective-copy for free — every output
position can directly look up arbitrary prefix positions. The
mainstream consensus when an SSM-only model can't do copy: interleave
attention layers with the SSM ones. Jamba, Striped Hyena, Samba,
H3/M3 all land here.

**Plan.** Drop a single attention head into one of the two Mamba-3
blocks in the renderer (or the router). Compare param count and
quality vs the pure-copy mechanism from §2a. The user is keen on
seeing both work — the comparison is the educational result, not just
either result alone.

**Why after copy mechanism.** Copy is the *surgical* answer (specific
to the "must contain something the input contains" pattern, ~4k extra
params). Attention is the *general* answer (any cross-position
dependency, more params, more compute). Doing copy first means we
have a baseline to measure attention against.

## 3. What shipped today (the headline)

A working **three-stage Mamba-3 harness** that takes a natural-language
prompt, routes to a specialist, runs the inner computation, and renders
the answer. Three Mamba-3-class models in series, ~165k parameters
total, plus the 45,318-param order-invariant GRU specialist.

| Stage | Architecture | Params | Status | Checkpoint |
|---|---|---|---|---|
| Router | Mamba-3 byte classifier (1 block, d=64) | 45,459 | val_acc 100 % | `checkpoints/tool_router_mamba3.pt` |
| Specialist | Order-invariant GRU (Hanoi solver) | 45,318 | optimal at n≤23 | `checkpoints/hanoi_invariant_gru_offtrace.pt` |
| Renderer (slot-fill) | Mamba-3 LM emitting templates | 74,400 | val_loss 0.006 | `checkpoints/tool_renderer_mamba3.pt` |

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

The slot-fill renderer is the *engineering* answer to the digit-copy
problem: the LM emits templates with named placeholders (`$N`,
`$OPTIMAL`, `$A`, `$B`, `$GCD`, `$MOVES_A`, `$MOVES_B`); the
orchestrator does deterministic substitution from the structured
payload. The boundary is clean — LM owns shape, orchestrator owns
values, neither tries to do the other's job.

The copy mechanism (§2a) is the *learned* answer to the same problem.
Both should land in the harness as selectable renderer modes.

Run any demo:

```bash
.venv/bin/python assistant.py \
    --router-checkpoint   checkpoints/tool_router_mamba3.pt \
    --renderer-checkpoint checkpoints/tool_renderer_mamba3.pt \
    "Solve Tower of Hanoi with 12 disks"
```

## 4. Files and commits today

**Files added:**
- `assistant.py` — three-stage harness with Tool registry, router/renderer hooks, slot-fill substitution, payload-fidelity guard
- `train_tool_router.py` — Mamba-3 byte classifier router trainer
- `train_tool_renderer.py` — Mamba-3 LM template renderer trainer (slot-fill)
- `train_tool_renderer_copy.py` — `CopyMamba3LM` (Pointer-Networks-style) trainer

**Updated:** `findings.md`, `index.md`, `index.html`, `docs/introduction/index.html`.

**Commits, in order:**
- `dc5144b` first-class harness over a Tool registry
- `af99190` Mamba-3 router replaces regex
- `1359194` renderer hook + Mamba-3 LM trainer
- `afa4dd2` renderer LM + payload-fidelity guard (digit-copy failure)
- `26d0ff7` slot-fill renderer (sidesteps the SSM copy weakness)
- `17b982a` `CopyMamba3LM` (proper ML answer)

## 5. Open risks / known issues

- **`cluster_sync.py` excludes `checkpoints/`.** Sensible default for
  the 100s of `.pt`s in there, but specialist + renderer + router
  checkpoints have to be `rsync`'d explicitly when moving between
  nodes. Cleanest fix is a `specialist_checkpoint` field on `Tool`
  that triggers a per-tool sync; deferred.
- **Background Bash with `... 2>&1 | tail -N` strands Python
  processes.** Block-buffered stdout never flushes; the output file
  stays 0B while CPU burns. Use `python -u`, no pipe. Documented in
  `feedback_background_python_output.md`.
- **PyTorch MPS pad-fallback.** The exact issue that motivated the
  MLX port decision in §1. Documented in `feedback_mps_pad_fallback.md`.

## Pick up here

```
1. Check /tmp/cluster_dispatch_logs/copy-train.log
2. If success: rsync the copy checkpoint, wire into assistant.py
3. If not: curriculum (gate-forced-to-0 pretrain) per §2a
4. Then: Mamba+attention hybrid block (§2b)
5. Then: MLX port of mamba3_minimal.Mamba3Block (§1)
```
