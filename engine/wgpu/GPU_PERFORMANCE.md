# GPU Performance — Benchmarks & Plan

## Running Benchmarks

### Model inference benchmark (all GPU paths)
```bash
# Build
cd engine/wgpu
source ~/.cargo/env  # if cargo not in PATH
cargo build --release

# Run locally (M4 / any GPU)
./target/release/mamba3-bench --model /tmp/run_length_next.bin

# Run on H100
ssh -p 32783 root@ssh2.vast.ai \
  'cd /root/mamba3-hands-on/engine/wgpu && source ~/.cargo/env && \
   cargo build --release && ./target/release/mamba3-bench --model /tmp/run_length_next.bin'
```

This runs all GPU paths (GPU, GPU-resident, GPU-batched, GPU-pipeline, GPU-full, GPU-fused) and compares each against CPU for correctness (max diff < 1e-4 = PASS).

### Raw GPU throughput benchmark (WIP)
```bash
# Will be a separate binary, not --raw-gpu in main
# Shader ready: engine/wgpu/src/shaders/raw_bench.wgsl
# Measures: GFLOPS at various buffer sizes, dispatch overhead
```

### Exporting a model for benchmarking
```bash
# From Python, export any trained specialist:
python -c "
from specialist_trainer import export_rust_model
export_rust_model('checkpoints/specialists/run_length_next_best.pt', '/tmp/run_length_next.bin')
"
```

## Current Results (2026-04-24)

Model: run_length_next, d=64, 3 layers, 28K params, 7 tokens

| Path | M4 (Metal) | H100 (Vulkan) |
|------|-----------|---------------|
| CPU | **0.8ms** / 8,589 tok/s | **3.6ms** / 1,943 tok/s |
| GPU-fused | 11.6ms / 605 tok/s | 21.2ms / 331 tok/s |
| GPU-full | 33.9ms / 207 tok/s | 62.1ms / 113 tok/s |
| GPU-pipeline | 61.8ms / 113 tok/s | 79.0ms / 89 tok/s |

GPU-fused is our best GPU path. CPU wins on this tiny model because dispatch overhead dominates the actual compute.

## Architecture: GPU-Fused

2 dispatches per layer (down from 6+ in non-fused paths):

1. **Fused dispatch**: `n_heads` workgroups × 16 threads each. One workgroup per head: norm → in_proj → SSM_scan → out_proj. Results written to per-timestep scratch.
2. **Reduce dispatch**: sums across heads, applies residual `x += scale * sum_h(out_proj_h)`.

Key files:
- `src/shaders/fused_layer.wgsl` — fused shader
- `src/shaders/head_reduce.wgsl` — reduce shader  
- `src/gpu_full.rs` — `forward_fused()` Rust dispatch

## Why wgpu Is Not Enough

wgpu on H100 goes through 5 translation layers:

```
WGSL → naga → SPIR-V → Vulkan → NVIDIA driver → PTX
```

What we **cannot** access through wgpu:
- **Tensor cores**: 989 TFLOPS vs 67 TFLOPS without them (15x left on table)
- **Warp-level intrinsics**: `__shfl`, cooperative groups
- **Explicit shared memory**: bank conflict avoidance, tiling control
- **Async copy**: global → shared memory pipelining

## Plan: PTX Backend

**PTX** (Parallel Thread Execution) is the lowest practical abstraction for H100. It's the ISA-level representation that runs directly on NVIDIA streaming multiprocessors.

**Not CUDA.** CUDA (via PyTorch) has a known precision issue — CPU and CUDA diverge on the same ops. This was proven: raw CUDA without FMA shows 7.6e-6 diff, but PyTorch CUDA ops produce larger errors. PTX gives us full control over fp32 precision.

### Steps
1. Quantify the wgpu abstraction tax with raw throughput benchmark
2. Write PTX kernels for matmul + SSM scan (the critical path)
3. Load PTX from Rust via `cudarc` or raw CUDA driver API
4. Plug-in replacement: detect NVIDIA → use PTX, otherwise → use wgpu (Metal portability)

### Abstraction hierarchy (NVIDIA)
```
PTX / SASS          ← lowest, direct SM control, we control precision
CUDA C/C++          ← compiles to PTX, but CUDA runtime adds precision issues
Vulkan Compute      ← portable, no tensor cores, no warp intrinsics
wgpu (WebGPU)       ← WGSL→SPIR-V→Vulkan, 5 layers of translation ← WE ARE HERE
```
