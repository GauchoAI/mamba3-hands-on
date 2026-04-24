# PTX Mamba-3 Engine — Implementation Plan

Author's framing: this is **not** a faithful port of the wgpu engine. The wgpu engine sits below six translation layers (WGSL → naga → SPIR-V → Vulkan → NVIDIA driver → PTX → SASS) and inherits every lowest-common-denominator choice each layer makes. PTX is one layer above SASS. Now we have direct hardware access — tensor cores, warp intrinsics, configurable shared memory, async copies, persistent kernels, mbarrier, cp.async, TMA, cluster shared memory.

Game-engine mindset: Teardown doesn't run a CPU voxel simulation inside a shader; it redesigns the data flow around how the silicon actually wants to work. Same move here.

**Plan structure:** v1 establishes correctness (bit-exact vs CPU reference) and a baseline number. v2+ iterates on creative hardware exploits.

---

## 0. Ground truth: what we're porting

**Reference:** `engine/wgpu/src/model.rs` (CPU), `engine/wgpu/src/gpu_full.rs` (fastest GPU path), `engine/wgpu/src/shaders/*.wgsl`, `engine/wgpu/src/backward.rs`, `engine/wgpu/src/train.rs`.

**Forward pass, per Mamba-3 block (from exploration of model.rs):**

```
x_normed    = layer_norm(x, ln_w, ln_b, eps=1e-5)
proj        = x_normed @ in_proj_w.T                              ∈ [L, dip]
  split: z_raw, x_raw  ∈ [L, d_inner]   (d_inner = 2*d_model)
         bp_raw, cp_raw ∈ [L, d_state]
         dd_dt, dd_a, trap_raw ∈ [L, n_heads]
         angles ∈ [L, num_rope_angles = d_state/2]
bp          = layer_norm(bp_raw, b_norm_w, b_norm_b)              ∈ [L, d_state]
cp          = layer_norm(cp_raw, c_norm_w, c_norm_b)              ∈ [L, d_state]
dt          = softplus(dd_dt + dt_bias)                           ∈ [L, n_heads]
a           = max(-softplus(dd_a), -1e-4)                         ∈ [L, n_heads]
decay       = exp(a * dt)                                         ∈ [L, n_heads]
trap        = sigmoid(trap_raw)                                   ∈ [L, n_heads]
dt_mean[t]  = mean_h(dt[t,h])
phase[t,k]  = cumulative_sum_t'(angles[t',k] * dt_mean[t'])       ∈ [L, num_rope_angles]
apply RoPE rotation (cos/sin of phase) to consecutive pairs of bp, cp
bx_cur[h,p,n] = x_raw[h*headdim+p] * bp[n]
inp[t,h,p,n]  = dt[t,h] * (trap[t,h]*bx_cur + (1-trap[t,h])*bx_prev[h,p,n])
bx_prev      ← bx_cur                                             (persistent across t)
z_silu       = z_raw * sigmoid(z_raw)                             ∈ [L, d_inner]

SSM recurrence, per (h, p), sequential over t:
  state[p,n] = decay[t,h] * state[p,n] + inp[t,h,p,n]
  y[t,h,p]   = (sum_n state[p,n]*cp[t,n] + d_param[h]*x_raw[h*headdim+p]) * z_silu[...]

y_out = y @ out_proj_w.T                                          ∈ [L, d_model]
x    += scale * y_out                                             (residual)
```

Final: `logits = layer_norm(x, final_ln_w, final_ln_b) @ embed_w.T   ∈ [L, vocab]` (weight-tied).

**Weight layout in .bin (row-major, f32, little-endian u32 header):**
- Header: `d_model, d_state, headdim, n_layers, vocab_size` (5 × u32)
- `embed_w` : `[vocab, d_model]`
- `embed_norm_w, embed_norm_b` : `[d_model]` × 2
- Per layer, in order:
  - `in_proj_w` : `[dip, d_model]` where `dip = 2*d_inner + 2*d_state + 3*n_heads + num_rope_angles`
  - `out_proj_w` : `[d_model, d_inner]`
  - `dt_bias` : `[n_heads]`
  - `d_param` : `[n_heads]`
  - `b_norm_w, b_norm_b` : `[d_state]` × 2
  - `c_norm_w, c_norm_b` : `[d_state]` × 2
  - `layer_norm_w, layer_norm_b` : `[d_model]` × 2
  - `scale` : `[1]`
- `final_norm_w, final_norm_b` : `[d_model]` × 2

**Backward:** reverse-time adjoint scan over SSM, standard matmul/norm/silu backward. Notable quirks (matched in v1):
- `dt_bias`, `d_param`, B/C norm weights, RoPE angle grads are all **zeroed** in current CPU impl (see `train.rs` 388–393). Parity still trains to >95%. v1 matches this; v3 fixes.

**Loss:** cross-entropy, softmax with max-subtraction for stability, mean over sequence length.

**Optimizer:** AdamW, `lr=1e-3, β1=0.9, β2=0.999, eps=1e-8, wd=0.1`. Decoupled weight decay: `p *= (1 - lr*wd)`, then standard Adam moment update.

---

## 1. Two architectures. Pick once, iterate from there.

### Option A — Per-op kernels (one PTX file per operation)
Direct analog of wgpu dispatches. 8 forward kernels + ~6 backward kernels. Each kernel independently testable.

- **Pros:** simple testing ladder, low risk, clear blast radius per kernel
- **Cons:** inherits wgpu's launch-overhead ceiling. ~8 launches × 3 layers = 24 launches per forward. Launch latency floor ~5 µs each = 120 µs per forward *just from launches*. That's the same class as GPU-fused wgpu (21 ms includes a bunch more).

### Option B — Persistent single-kernel forward
ONE launch does entire forward. All weights pre-loaded into shared memory (28 KB model fits easily in 228 KB SMEM). All activations stay in registers/SMEM across layers. HBM is touched for: (1) weight load on startup (once), (2) token IDs in, (3) logits out.

- **Pros:** sub-microsecond intra-layer transitions, total HBM round-trips ≈ 2. Theoretical ceiling ~ memory bandwidth of logits out.
- **Cons:** single big kernel to debug, harder to unit-test in isolation, register pressure (255/thread cap on H100)

### Decision: v1 = **Option A**, v2 = **Option B**.
Per-op kernels let us build the CPU-reference test ladder cleanly. Once every op is bit-exact, we fuse confidently.

---

## 2. v1 — Per-op PTX (correctness + baseline)

### 2.1 Forward kernels

| # | Kernel | Dispatch shape | Math |
|---|---|---|---|
| 1 | `embed_gather` | `(L,)` workgroups, 32 threads | `x[t,:] = embed_w[token[t],:]` |
| 2 | `layer_norm` | `(L,)` workgroups, 128 threads, warp-shuffle reduce | mean/var over d_model, affine |
| 3 | `matmul_f32` | `(M/16, N/16)` blocks × 16×16 threads | `C = A @ B.T` (weights row-major) |
| 4 | `ssm_prep` | `(L,)` workgroups, 128 threads | dt, a, decay, trap, RoPE phase (serial in t), B/C norm, outer product → inp |
| 5 | `ssm_scan` | `(n_heads,)` workgroups, headdim×d_state threads | sequential t, state in registers |
| 6 | `gate_silu` | elementwise, 256 threads per block | `z * sigmoid(z)` |
| 7 | `residual_add` | elementwise | `x += scale * y` |
| 8 | `final_head` | fused norm + matmul | `logits = LN(x) @ embed_w.T` |

**Kernel 3 detail:** v1 uses a **shared-memory-tiled** variant of the naive matmul in `ptx-bench`. 16×16 block tile, 16 iterations of K loaded cooperatively. Still FP32 fma.rn — same precision as CPU reference. Target ~20 TFLOPS (vs 3 TFLOPS naive). This is v1's only non-trivial kernel.

**Kernel 5 (SSM scan) detail:** dispatch is `(batch, n_heads)` workgroups, each with `headdim × d_state` threads (e.g., 16×16 = 256 for tiny config). Each thread owns one `state[p,n]`. Sequential loop over t:
```ptx
for t in 0..L:
    state = decay[t] * state + inp[t, p, n]
    // Reduce sum_n state[p,n] * cp[t,n] within warp row → warp-shuffle butterfly
    if threadIdx.n == 0: y[t, p] = reduced + D*x_raw[t,p], multiplied by z_silu[t,p]
```
State lives in **registers** (256 floats per thread, fits easily). Zero SMEM traffic in the hot loop.

### 2.2 Backward kernels

| Kernel | Notes |
|---|---|
| `cross_entropy_bwd` | softmax with max-sub, `d_logits = softmax - one_hot(target)`, scaled by `1/L` |
| `matmul_bwd` | reuses `matmul_f32` with transposed args — two launches per matmul |
| `layer_norm_bwd` | per-row, warp-shuffle reduce for sums |
| `gate_silu_bwd` | elementwise |
| `ssm_scan_bwd` | reverse scan, adjoint state in registers, same (h, p) parallelism as forward |
| `embed_scatter_bwd` | atomic adds into `d_embed` (collisions rare for small L) |

### 2.3 Optimizer kernel

`adamw_step`: 1 launch, 128 threads, grid-stride loop over all params. Reads `grad`, updates `m`, `v`, applies decoupled weight decay + Adam step. Takes `lr, β1, β2, eps, wd, step` as scalar params for bias correction.

---

## 3. v2+ — Creative iteration roadmap

Each bullet is a one-commit experiment. We measure, keep the wins.

### v2 — Single persistent forward kernel
Combine all 8 forward kernels into one. Warp specialization:
- Warp 0: embedding gather + norm
- Warps 1..n_heads: one per SSM head, owns state in registers
- Barriers via `bar.sync` at layer boundaries
- Weights resident in SMEM (load once at kernel start)

Expected: **100x reduction in launch overhead**. For the tiny d=64 model, inference becomes a single ~100 µs kernel instead of 24+ launches.

### v3 — Warp shuffles everywhere
Replace SMEM-based reductions in layer_norm, softmax, SSM output reduction with `shfl.sync.bfly` butterfly. Zero SMEM traffic, zero barriers, single warp owns the operation end to end.

### v4 — Tensor cores with precision guard
`mma.sync.aligned.m16n8k16.row.col.f32.tf32.tf32.f32` — TF32 inputs with FP32 accumulate. 10x throughput on matmul. Guarded by a `PTX_USE_TENSOR_CORES` flag, default OFF. Provides A/B tool to measure precision drift per-task.

### v5 — cp.async weight prefetch
While layer N computes, async-copy layer N+1 weights from HBM to SMEM. Hides weight-read latency entirely on multi-layer models.

### v6 — Persistent training kernel
Forward → loss → backward → optimizer, one launch. Single-example training step. For parity (L=7) this is *tiny* per step, all in one kernel.

### v7 — TMA for longer sequences
Hopper TMA for bulk HBM→SMEM of inputs when L grows beyond register budget.

### v8 — Cluster shared memory
Thread-block cluster (H100 SM-to-SM SMEM sharing). Enables parallel SSM scan across SMs for L ≫ register budget. Not needed for parity-sized tasks.

### v9 — `wgmma` (Hopper warpgroup matmul async)
Full Hopper-generation tensor core path for matmul. 4 warps produce, 4 warps consume. Only relevant for larger inner dims.

---

## 4. Correctness test ladder

Every kernel has a unit test. Pass criterion: `max |PTX - CPU| < 1e-5` at a fixed seed. Test binary reuses the existing CPU ref in `mamba3_engine::model`.

1. `test_embed_gather` — single token
2. `test_layer_norm` — random input, fixed weights
3. `test_matmul_f32` — 128×128×128 random
4. `test_ssm_prep` — tiny L=3, d_state=16
5. `test_ssm_scan` — L=7, checks states[t] per step
6. `test_gate_silu` — elementwise vs CPU
7. `test_residual_add` — trivial
8. `test_final_head` — full logits match
9. `test_full_forward` — load .bin, logits match CPU to 1e-5
10. `test_cross_entropy_bwd`
11. `test_ssm_scan_bwd`
12. `test_single_grad_step` — params after step match CPU
13. `test_training_trajectory` — 100 steps, loss curve matches CPU to 1e-3
14. `test_parity` — 5000 steps reach ≥95% acc

Only step 9 onwards is in the end-to-end benchmark. Everything earlier is in `cargo test`.

---

## 5. Benchmark targets

Reference: `run_length_next` model, d=64, 3 layers, 28K params, L=7 (from `GPU_PERFORMANCE.md`).

| Path | H100 now (wgpu) | v1 target | v2 target | Ceiling |
|------|---|---|---|---|
| Inference ms/call | 21.2 ms | ≤ 5 ms | ≤ 0.5 ms | ≤ 0.1 ms |
| Inference tok/s | 331 | ≥ 1,500 | ≥ 15,000 | ≥ 50,000 |
| Training ms/step | N/A on GPU | ≤ 10 ms | ≤ 2 ms | ≤ 0.5 ms |

We report into the same table format as `GPU_PERFORMANCE.md`.

---

## 6. cudarc 0.16 API (verified in ptx-bench)

```rust
use cudarc::driver::{CudaContext, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::Ptx;

let ctx = CudaContext::new(0)?;                            // pick GPU 0
let stream = ctx.default_stream();

// Compile/load PTX at startup
let module = ctx.load_module(Ptx::from_src(PTX_SRC))?;
let f_mm   = module.load_function("matmul_f32")?;
let f_ln   = module.load_function("layer_norm")?;
// ...

// Persistent device buffers (upload weights once)
let dev_embed: CudaSlice<f32> = stream.memcpy_stod(&host_embed)?;

// Launch
let cfg = LaunchConfig { grid_dim: (gx,gy,1), block_dim: (bx,by,1), shared_mem_bytes: smem };
let mut launch = stream.launch_builder(&f_mm);
launch.arg(&dev_a);
launch.arg(&dev_b);
launch.arg(&mut dev_c);
let m_u32 = m as u32;                 // scalars must outlive launch()
launch.arg(&m_u32);
unsafe { launch.launch(cfg)? };

stream.synchronize()?;                // before reading back
let host_out = stream.memcpy_dtov(&dev_c)?;
```

**Gotchas:**
- Scalars must be `&` to a local that lives through `launch.launch`. Can't pass `&(m as u32)` — that's a temporary.
- `CudaSlice<f32>::new(size)` doesn't exist; use `stream.alloc_zeros::<f32>(n)?` or `memcpy_stod`.
- PTX source must be ASCII. No em-dashes, no Unicode quotes. (Learned this in ptx-bench.)
- `module` and `CudaFunction` are cheap to clone (Arc internals). Safe to pass around.
- Stream ordering: launches on the same stream are serialized. Use a single default stream for v1.

---

## 7. Cargo layout

```
engine/ptx/
  Cargo.toml
  PLAN.md                           ← this file
  src/
    lib.rs                          ← public API: PtxModel, PtxTrainer
    runtime.rs                      ← CudaContext wrapper, CompiledKernels
    model.rs                        ← PtxModel struct, forward() dispatch
    training.rs                     ← PtxTrainer, train_step()
    kernels.rs                      ← LaunchConfig helpers per kernel
    weights.rs                      ← upload/mirror of Mamba3Model weights
    ptx/                            ← PTX source files (include_str!)
      embed.ptx
      layer_norm.ptx
      matmul.ptx
      ssm_prep.ptx
      ssm_scan.ptx
      gate_silu.ptx
      residual.ptx
      final_head.ptx
      cross_entropy.ptx
      ssm_scan_bwd.ptx
      layer_norm_bwd.ptx
      embed_scatter.ptx
      adamw.ptx
    bin/
      main.rs                       ← benchmark binary, matches wgpu format
  tests/
    kernels_vs_cpu.rs               ← the correctness ladder
```

**Dependencies (matches `engine/wgpu`):**
```toml
mamba3-engine = { path = "../wgpu" }          # reuse CPU reference + Mamba3Model
cudarc        = { version = "0.16", default-features = false, features = ["std", "driver", "cuda-version-from-build-system", "dynamic-linking"] }
bytemuck      = "1"
```

Crate-level gate: the build only works on Linux with CUDA installed. On macOS this crate is excluded. Add a `[target]` gate or let cudarc's link check fail loudly.

---

## 8. Execution order

**Phase 1 — v1 forward (target: 1 commit per numbered step):**

1. `engine/ptx/` skeleton (Cargo.toml, lib.rs, empty modules)
2. `runtime.rs` — CudaContext wrapper, kernel registry
3. Kernel `embed_gather` + test
4. Kernel `layer_norm` + test
5. Kernel `matmul_f32` (tiled variant) + test
6. Kernel `ssm_prep` + test
7. Kernel `ssm_scan` + test
8. Kernels `gate_silu`, `residual_add`, `final_head` + tests
9. `PtxModel::forward()` end-to-end + `test_full_forward`
10. Benchmark binary + PTX row in the table

**Phase 2 — v1 training:**

11. `cross_entropy` (loss + d_logits)
12. `layer_norm_bwd` + test
13. `matmul_bwd` (reuse matmul with transposed args) + test
14. `ssm_scan_bwd` + test
15. `embed_scatter_bwd` + `gate_silu_bwd`
16. `adamw` optimizer kernel + test
17. `PtxTrainer::train_step()` + `test_single_grad_step`
18. `run_parity_training` equivalent, verify ≥95%
19. Benchmark ms/step

**Phase 3 — creative iteration (v2+):**

20+. Pick highest-ROI item from §3, measure, commit. Expected order: v2 (persistent fwd) → v3 (shuffles) → v6 (persistent training) → v5 (prefetch) → v4 (tensor cores, guarded).

---

## 9. Open questions / needs-verification

- **RoPE backward:** current impl zeros RoPE angle grads. Match in v1, revisit later — may affect learning on longer sequences.
- **B/C layer-norm weight grads:** also zeroed in current impl. Match in v1.
- **dt_bias / d_param grads:** zeroed. Match in v1.
- **Batch dim in SSM scan:** reference impl processes batch=1. v1 matches; batching is a v2+ exploration (natural fit for cluster shmem).
- **Larger models:** this plan sizes everything for the parity-scale model (d=32-64, n_layers=1-3). When we scale to d=256+, weight-SMEM residency breaks — v5 (cp.async prefetch) covers that case.

---

## 10. What success looks like (short version)

1. `cargo test -p ptx-engine` on H100 → all kernels pass bit-exact
2. `cargo run --bin ptx-bench --release -- --model /tmp/run_length_next.bin` prints a PTX row in the benchmark table, with tok/s ≥ 5× wgpu's H100 numbers
3. `cargo run --bin ptx-bench --release -- --parity` reaches ≥95% accuracy on bitstring parity
4. PLAN.md gets an "iteration log" section where each creative commit appends its measurement delta

Then we iterate.
