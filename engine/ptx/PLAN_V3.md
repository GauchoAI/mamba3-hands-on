# PTX Engine v3 — Training + Scheduler

**End state.** `engine/ptx/` owns the GPU. A slot-scheduler daemon accepts training job submissions, runs them in fixed-N parallel slots (Tetris-packed), each slot using the PTX engine for forward + backward + AdamW. No Python in the inner loop. `three_populations.py` becomes a client that submits mutate-and-train jobs and reads back accuracies. No more accidental contention, no more CPU-core starvation, no more variance.

**This plan is huge. We execute in layers, each layer is a commit that works end-to-end.**

---

## Layer 1 — PTX training (forward cache + backward + AdamW)

Forward pass already works. Adds the rest of the training step.

### 1.1 Forward with activation cache
Extend `PtxModel::forward` to `forward_cached`:
- Same compute as forward, but persists intermediates in scratch so backward can read them.
- Saves: `layer_inputs[li]` (x into each layer), `layer_projs[li]` (in_proj output), `layer_y_inner[li]` (SSM output pre-out_proj), `layer_scan_states[li]` (full SSM state sequence, needed for adjoint scan), `x_before_head` (after final norm).
- SMEM layout already has all but `scan_states` — add that as a new `PtxScratch` field sized `n_layers × (L+1) × H × hd × ds`.

### 1.2 Cross-entropy loss + gradient
One kernel. Grid (L,), block (256,). Per timestep:
- Find max logit (warp reduce)
- Compute exp + sum (warp reduce)
- Softmax → d_logits = softmax; subtract 1.0 at target index
- Loss = -log(softmax[target])
- Scale d_logits and loss by 1/L
- Return loss in a single-float device buffer

### 1.3 AdamW optimizer step
One kernel, elementwise. Grid-stride loop over all params:
- `p *= (1 - lr·wd)`
- `m = β1·m + (1-β1)·g`
- `v = β2·v + (1-β2)·g²`
- `m_hat = m / (1 - β1^step)`, `v_hat = v / (1 - β2^step)`
- `p -= lr · m_hat / (√v_hat + eps)`
- Takes `lr, β1, β2, eps, wd, step` as scalar args.

### 1.4 Backward — matmul
No new kernel; reuse `matmul_t`. For `out = A @ B^T`:
- `dA = dOut @ B` — call matmul_t(dOut, swap(B), M, K, N). Actually easier: add a second kernel `matmul_ab` for `C = A @ B` (no transpose) since dA path needs that. Or — simpler — store A/B in the layout that matmul_t wants. I'll write a tiny `matmul_ab_tiled` alongside.
- `dB = dOut^T @ A` — needs `matmul_tT` or equivalent. Also tiled.

### 1.5 Backward — layer_norm
One kernel per call. Grid (L,), block (32,). Per row:
- Recompute mean, var from saved x (cache the x that went into this LN)
- Standard LN-backward math (see `backward.rs` layer_norm_backward)
- Accumulate d_w, d_b via atomicAdd across rows
- Write d_x

### 1.6 Backward — embedding scatter
Atomic add into `d_embed` by token id. Grid (L,), block (min(d, 256),).

### 1.7 Backward — SSM scan (the hard one)
Grid (H,), block (hd×ds). One head per block. Reverse scan (t = L-1 down to 0):
- Adjoint state `dh[p,n]` in registers per-thread
- Per timestep: accumulate `dh += dy_pre ⊗ C[t]`, derive d_C, d_x, d_D contributions via warp-reductions
- Propagate: `dh *= decay[t]`
- Accumulate d_z_silu from saved pregated-y
- Output: d_scan_inp, d_decay, d_cp, d_x, d_z_silu, d_d

This mirrors `backward.rs::ssm_scan_backward` one-to-one. Sequential over t inside the block; parallel over (p,n) within head.

### 1.8 Backward — silu/gate, trap, bx-outer-product
Elementwise + tiny accumulators. Uses saved z_raw, x_raw, bp.

### 1.9 PtxTrainer end-to-end
Rust-side orchestrator:
- `PtxTrainer::new(model, lr, wd)` — owns m, v optimizer state buffers (one big `CudaSlice<f32>` per param vector, flat).
- `PtxTrainer::train_step(tokens, targets) -> loss: f32` — runs forward_cached → CE → backward → AdamW → returns loss (single f32 transfer).

### 1.10 Correctness ladder
- Unit test each backward kernel vs CPU reference at fixed seed
- `test_single_grad_step`: params after one step match CPU to 1e-5
- `test_parity_train`: runs 5000 steps on the `run_parity_training` config, expects ≥ 95% accuracy

### 1.11 Commit points for Layer 1
Each a compiling + passing commit:
- L1.a `adamw + ce_loss kernels + grad diag test`
- L1.b `forward_cached + scan_states buffer`
- L1.c `ssm_scan_backward kernel + test`
- L1.d `matmul/layernorm/embed backward kernels`
- L1.e `PtxTrainer::train_step end-to-end`
- L1.f `parity training hits 95%+`

---

## Layer 2 — Slot scheduler (the daemon)

### 2.1 Architecture
Rust binary `ptxd` (PTX daemon). Long-running. Owns the CUDA context, the PtxContext, the compiled PTX module (compiled once).

```
┌─────────────────────────────┐
│  three_populations.py       │
│  (or any client)            │
└──────────┬──────────────────┘
           │  unix socket / JSON
┌──────────▼──────────────────┐
│  ptxd (Rust)                │
│   ├─ Job queue              │
│   ├─ N fixed slots          │
│   │   ├─ PtxModel (weights) │
│   │   ├─ PtxTrainer         │
│   │   ├─ CUDA stream        │
│   │   └─ Scratch buffers    │
│   └─ GPU resource arbiter   │
└─────────────────────────────┘
```

### 2.2 Slot model
- Each slot pre-allocates: weight buffers, scratch, optimizer state, for the LARGEST config it'll see (say d=128 L=6).
- N slots = `gpu_memory / per_slot_estimate`. For tiny specialist models: 20+ slots fit on an H100 easily. Each has its own CUDA stream.
- Multi-stream lets slots run truly concurrently. GPU scheduler decides SM sharing.
- No oversubscription: if all N slots are busy, new jobs queue; once a slot frees it pulls the next job.

### 2.3 Job lifecycle
1. Client sends `{job_id, task, config, steps, target_acc}` over socket.
2. `ptxd` queues.
3. When slot free: load weights (from scratch or from checkpoint), initialize PtxTrainer.
4. Run `steps_per_cycle` of train_step, periodically eval.
5. If target_acc reached or plateaued: return checkpoint + accuracy.
6. Slot released.

### 2.4 Protocol
Line-delimited JSON over unix socket `/tmp/ptxd.sock`:
```
C→S: {"op": "submit", "id": 1, "task": "parity", "config": {...}, "steps": 2000, "target": 0.95}
S→C: {"op": "accepted", "id": 1, "slot": 3}
S→C: {"op": "progress", "id": 1, "step": 500, "loss": 0.12, "acc": 0.78}
S→C: {"op": "done", "id": 1, "best_acc": 0.96, "steps": 1800}
```

### 2.5 Integration with `three_populations.py`
Replace `spawn_worker` in `three_populations.py` with a ptxd client. No subprocess fork; just send JSON and wait for `done`. Orchestrator gets capacity from daemon (`{"op": "capacity"}` → `{"slots": 20, "free": 17}`), submits up to that.

### 2.6 Commit points for Layer 2
- L2.a `ptxd skeleton: unix socket, job protocol, hello-world submit`
- L2.b `single-slot training via PtxTrainer, returns loss/acc`
- L2.c `N slots + multi-stream concurrent`
- L2.d `three_populations.py PTX backend swap`
- L2.e `end-to-end GA run on PTX, no CPU workers`

---

## Layer 3 — Benchmarks for the real thing

Once L1 + L2 ship:
- Measure training throughput: parity steps/sec, PTX vs CPU specialist_trainer
- Measure parallelism: 20 concurrent trainings on H100 — aggregate steps/sec
- Measure end-to-end GA round time with PTX backend

---

## Order of execution (this session and next)

This session: L1.a → L1.b → ideally L1.c (if scan backward math turns out clean)
Next session: L1.d → L1.e → L1.f → L2.a
Then: L2.b+ onward

Each commit must keep all existing tests green (forward correctness 7.6e-6 stays locked).

---

## What we are explicitly NOT doing

- Rewriting the mutation strategies (`coordinator.mutate_config`) — they work, just get called from Rust via Python subprocess still.
- Replacing SQLite (`state_db.py`) — Python can keep writing to it, Rust only does training.
- Multi-GPU — one H100, one daemon, N slots.
- Fancy optimizer variants (Lion, etc.) — start with AdamW that matches CPU.
