# Mamba-3 Engine — Low-Level Implementations

PyTorch's CUDA runtime introduces precision differences that prevent
convergence on tasks requiring exact state tracking (parity, carry
propagation). The SSM scan kernel is correct (7.6e-06 diff vs CPU),
but PyTorch's linear layers, embeddings, and optimizer ops on CUDA
introduce enough noise to break gradient flow.

## Solution: vertical integration

Own the entire forward+backward pass. No PyTorch CUDA runtime.

### Implementations

| Implementation | Target | Status |
|---------------|--------|--------|
| `ssm_cuda_kernel.py` | CUDA (raw) | Scan works, rest is PyTorch |
| `engine_wgpu/` | Rust + wgpu | Planned — portable GPU compute |
| `engine_metal/` | Metal (Apple) | Planned — native Apple Silicon |

### Architecture

```
engine/
├── wgpu/
│   ├── Cargo.toml
│   ├── src/
│   │   ├── lib.rs          # Python bindings via PyO3
│   │   ├── scan.rs         # SSM scan compute shader
│   │   ├── linear.rs       # Matrix multiply (fp32, no FMA)
│   │   ├── embed.rs        # Embedding lookup
│   │   └── shaders/
│   │       ├── ssm_scan.wgsl    # WebGPU shader for scan
│   │       ├── matmul.wgsl      # WebGPU shader for matmul
│   │       └── silu.wgsl        # Activation functions
│   └── tests/
└── README.md
```

### Why wgpu?

- **Portable**: Vulkan (Linux/H100), Metal (Mac), DX12 (Windows), WebGPU (browser)
- **Explicit**: no hidden FMA, no implicit TF32, no runtime "optimizations"
- **Fast**: compute shaders are as fast as CUDA for this workload
- **Rust**: memory safe, no GIL, easy to distribute
- **PyO3**: seamless Python integration for training
