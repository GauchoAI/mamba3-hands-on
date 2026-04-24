# PTX Engine v2 — Beat CPU on the run_length_next Benchmark

**Binding objective.** `ptx-bench` on `/tmp/run_length_next.bin` (d=64, 3 layers, 28K params, 7-token input) must report `PTX + CUDA Graph` **ms/inference strictly below** the `CPU (mamba3-engine)` value in the same run. We are not allowed to change the model, the benchmark, or the harness. We win here, with these numbers.

**Baseline (v1 landing).**
- CPU: ~2.6 ms / 2,700 tok/s (shared vast.ai H100 container, rayon+SIMD on EPYC)
- PTX + CUDA Graph: ~3.3–4.5 ms / 1,500–2,100 tok/s (correct to 7.6e-6)

**Gap to close: 1.5–2× latency reduction, with no regression in correctness.**

---

## Why we're currently losing, analytically

The H100 compute budget for this forward is trivial:
- Total weight traffic ≈ 112 KB; HBM bandwidth ~3 TB/s → **40 ns** memory-bound compute-time floor.
- Actual kernel compute is ~hundreds of microseconds.

We are at milliseconds, 10–100× above that floor. Everything above ~100 µs is **host↔device overhead**. Specifically:
1. `stream.synchronize()` — waits for ALL pending compute, has driver-fixed minimum cost.
2. `stream.memcpy_dtov(&logits)` — allocates a 7,280-byte host Vec and DMA-transfers logits back. Even small transfers have fixed setup cost.
3. `stream.memcpy_htod(tokens, &mut dst)` — uploads 28 bytes per call.
4. cudarc per-call bookkeeping (bind_to_thread, error checks).
5. CUDA Graph launch itself — minor but non-zero.
6. Small launches with heavy underutilization (naive matmul for 7×64 inputs uses ~4% of an SM's threads).

The persistent single-kernel design attacks #1, #4, #5, #6 in one shot. Pinned memory attacks #2, #3. Argmax-on-device attacks #2.

---

## Execution plan — 7 commits, measured after each

We commit after each step and record the latency delta. We do not skip ahead and we do not give up. If a step goes sideways we fix it in the next commit, not by reverting.

### Commit v2.1 — Diagnostic timing

**What.** Add per-phase timing to the benchmark binary (not behind a feature flag, always on). Break each `forward_graph` call into: upload-tokens, graph-launch, synchronize, logits-memcpy-dtov. Use CUDA events and `Instant::now()`.

**How.** Wrap the body of `forward_graph` with timestamps; report averages over N=100 calls after warmup. Print as a separate section in `./ptx-engine`.

**Why.** Confirms which of sync / memcpy / launch dominates. Every subsequent step is sized against this number.

**Commit message.** `ptx-engine: diagnostic timing breakdown for forward_graph`

**Benchmark expectations.** No latency change; new output lines.

---

### Commit v2.2 — Argmax kernel, return predictions not logits

**What.** Add `argmax_f32` kernel: input (L, V), output (L,) u32. Replace the 7,280-byte `memcpy_dtov(&logits)` with a 28-byte `memcpy_dtov(&preds)`. Expose both `forward_graph` (returns logits, kept for tests) and `forward_graph_argmax` (returns predictions).

**How.** Simple 1-block-per-timestep kernel, 256 threads, shared-memory reduction over V=260. One more kernel node in the graph. Adjust main.rs to call the argmax variant in the benchmark loop; correctness check still compares logits (so keep `forward_graph` available).

**Why.** Cuts readback by 260×. If readback is the bottleneck this alone may cover the gap.

**Commit message.** `ptx-engine: argmax kernel; benchmark returns predictions (28 bytes vs 7.3 KB)`

**Benchmark expectations.** Expect 200–800 µs improvement depending on how much readback cost we had.

---

### Commit v2.3 — Pinned host memory for I/O

**What.** Allocate pinned host buffers for (a) input tokens and (b) output predictions. `PtxModel` holds them. `upload_tokens` memcpy's from pinned buffer; `forward_graph_argmax` DMAs into pinned buffer and returns a `&[u32]` view.

**How.** cudarc exposes `CudaContext::alloc_host` or equivalent. Hold `*mut u32` pinned buffer pointers in PtxModel.

**Why.** Pinned host memory gives 5–10× DMA throughput and removes pageable-memory staging. For our tiny transfers, this changes the fixed overhead floor.

**Commit message.** `ptx-engine: pinned host memory for tokens and predictions`

**Benchmark expectations.** Another 100–400 µs off.

---

### Commit v2.4 — Fuse params+dt_mean+phase into one kernel

**What.** One kernel `compute_ssm_prep_full` replaces `compute_ssm_params` + `compute_dt_mean` + `compute_phase`. Grid (1,), Block (max(L*H, L, n_angles)) threads. Three phases separated by `__syncthreads()`: compute dt/decay/trap for all (t,h), reduce dt to dt_mean[t], sequential phase[t,k] accumulate.

**How.** Single block because phase is sequential in t. Layout: warp per t for the per-(t,h) work, then block-wide reduce for dt_mean, then the sequential phase loop with thread-per-k.

**Why.** Removes 2 kernel nodes per layer = 6 total. With graph launch, node count isn't the perf driver, but the kernels each have setup/shutdown costs on the SM.

**Commit message.** `ptx-engine: fuse SSM params + dt_mean + phase (3→1 kernel per layer)`

**Benchmark expectations.** Small win, 50–150 µs.

---

### Commit v2.5 — Inline z_silu + skip into ssm_scan_sequential

**What.** Delete `compute_z_silu` kernel. Inside `ssm_scan_sequential`, read `z = proj[t*dip + h*hd + p]` and compute `z * sigmoid(z)` at the point of use. Also inline `d_param[h] * x_val` which is already there.

**How.** Small edit to the scan kernel. Remove the z_silu scratch buffer and its launch.

**Why.** One less kernel, one less scratch buffer allocation, better cache behavior since z_raw is read right next to where we use it.

**Commit message.** `ptx-engine: inline z_silu into ssm_scan; drop separate kernel`

**Benchmark expectations.** Small win, 30–100 µs.

---

### Commit v2.6 — Tiled matmul with shared-memory staging

**What.** Replace the naive `matmul_t` (one thread per output element, no reuse) with a 32×32 tile shared-memory matmul. Block (32, 32) = 1024 threads, each computing one output; A-tile and B-tile staged in SMEM cooperatively.

**How.** Standard tiled matmul pattern. K-axis loop over tiles of size 32. `__syncthreads()` between load and compute.

**Why.** Naive matmul is 3 TFLOPS on H100 (5% of peak). Tiled gets 10+ TFLOPS. For our in_proj/out_proj shapes (7×64→320), compute is small but launch config becomes more efficient.

**Commit message.** `ptx-engine: tiled matmul_t with 32x32 shared-memory staging`

**Benchmark expectations.** 100–300 µs win, more if matmul time is currently significant.

---

### Commit v2.7 — Persistent single-kernel forward

**What.** ONE kernel `mamba3_forward_persistent` does the entire forward. Takes all weights (by pointer), tokens, outputs logits. Internal phases separated by `__syncthreads()`. Launched once per `forward()` call.

**How.** Single-block kernel, 1024 threads. Layout:
- Phase embed: 32 threads per token gather + block-wide embed-norm
- For each layer (loop inside kernel):
  - Pre-norm (32 threads/token)
  - In_proj matmul: 1024 threads compute proj[L × dip] cooperatively, block-tiled
  - SSM prep (dt/decay/trap/dt_mean/phase/bp/cp/z_silu) — sequential within block
  - SSM scan: 8 warps × 32 threads, one warp per head, scan loops over t inside the warp; warp shuffle reduces across n for the output
  - Out_proj matmul: 1024 threads cooperative
  - Residual: elementwise
- Final norm + LM-head matmul

The block processes everything sequentially at the op level but each op uses all 1024 threads for its parallelism.

For the tiny model (h=8, hd=16, ds=16), this fits 1024 threads comfortably. Per-head SSM scan has 32 threads with 8 (p,n) pairs each — partial tree reduction with warp shuffle.

**Why.** Eliminates every inter-kernel cost: driver overhead per launch, graph-node overhead, cudarc per-call bookkeeping. Entire forward is one launch. Combined with the sync+pinned-memcpy on the boundary, this is where we cross under CPU.

**Commit message.** `ptx-engine: persistent single-kernel forward — entire Mamba-3 pass in one launch`

**Benchmark expectations.** Target total ≤ 1.5 ms median. This is the commit that wins.

---

### Commit v2.8 — If v2.7 doesn't win, event-based completion instead of synchronize

**What.** Record a CUDA event after the forward kernel. Use `cudaEventQuery` in a tight spin loop instead of `cudaStreamSynchronize`. If the driver's sync minimum-cost is what's left on the table, polling skips it.

**Why.** Fallback. Only fire if v2.7 is still above CPU median.

**Commit message.** `ptx-engine: spin-poll CUDA event instead of stream sync on forward`

---

### Commit v2.9 — Tuning loop

**What.** Run the benchmark 10 times, record best-case latency (not just median). Tune block sizes in the persistent kernel if p-best leaves headroom.

---

## Running order, in one pass

```
git pull / build on H100
./target/release/ptx-engine --model /tmp/run_length_next.bin  # baseline
# for step in 2.1 2.2 2.3 2.4 2.5 2.6 2.7:
#   edit, commit, push
#   ssh H100 → pull, build, run, record numbers
# after 2.7 : if still not winning, apply 2.8 ; then 2.9
```

Each commit is a single logical change. No "while I'm in there" refactors. If a commit makes things worse and we've already moved on, the next commit targets the regression explicitly — we don't revert.

---

## Non-negotiables

- Correctness check `max |PTX − CPU| < 1e-3` must stay PASS on every commit.
- Graph vs per-op check must stay 0.0 max-diff.
- Single-command reproducibility: `cargo run --release --bin ptx-engine -- --model /tmp/run_length_next.bin`.

---

## What this plan explicitly does not include (and why)

- **Tensor cores (TF32/FP16/mma.sync, wgmma).** They win on compute-bound big matmul; our bottleneck is not compute. Revisit when we scale models up.
- **Cooperative groups / multi-block grid sync.** The persistent kernel is single-block; cooperative launch adds complexity with no expected win at this scale.
- **Training kernels.** Separate objective. This plan is inference-latency only.
- **TMA, cp.async, cluster SMEM.** Hopper-specific bandwidth tricks for larger data. Our data fits in registers + a few KB of SMEM.

---

## Definition of done

```
$ ./target/release/ptx-engine --model /tmp/run_length_next.bin
...
CPU (mamba3-engine)      2.XXX  ms
PTX + CUDA Graph         1.YYY  ms    (1.YYY < 2.XXX on the same run)
```

Printed on the H100, committed to `main`, row added to `GPU_PERFORMANCE.md` showing PTX beating CPU on the same row.
