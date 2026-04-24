"""Low-level CUDA SSM scan via torch.utils.cpp_extension.

Inline CUDA kernel with explicit control over:
- FMA (disabled via __fmul_rn + __fadd_rn)
- Accumulation order (sequential, not tree)
- Precision (fp32 throughout, no implicit conversions)

This is the "lowest level possible" test to isolate the precision issue.
"""

import torch
from torch.utils.cpp_extension import load_inline

CUDA_SOURCE = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// SSM scan kernel — one thread per (batch, head) pair
// NO FMA: uses __fmul_rn and __fadd_rn for exact fp32 arithmetic
__global__ void ssm_scan_kernel(
    const float* __restrict__ inp,   // (B, L, H, hD, dS)
    const float* __restrict__ decay, // (B, L, H)
    const float* __restrict__ C,     // (B, L, H, dS)
    const float* __restrict__ x,     // (B, L, H, hD)
    const float* __restrict__ z_silu,// (B, L, H, hD) — precomputed
    const float* __restrict__ D,     // (H,)
    float* __restrict__ y,           // (B, L, H, hD)
    int B, int L, int H, int hD, int dS
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * H;
    if (idx >= total) return;

    int b = idx / H;
    int h = idx % H;

    // Local state — in registers
    float state[16 * 16];  // hD * dS, max 256 floats
    for (int i = 0; i < hD * dS; i++) state[i] = 0.0f;

    float D_h = D[h];

    for (int t = 0; t < L; t++) {
        // Load decay for this (b, t, h)
        float dec = decay[b * L * H + t * H + h];

        // State update: h = decay * h + inp
        // Using __fmul_rn + __fadd_rn to avoid FMA
        for (int p = 0; p < hD; p++) {
            for (int n = 0; n < dS; n++) {
                int si = p * dS + n;
                float inp_val = inp[((b * L + t) * H + h) * hD * dS + p * dS + n];
                state[si] = __fadd_rn(__fmul_rn(dec, state[si]), inp_val);
            }
        }

        // Output projection: y_t[p] = sum_n(state[p,n] * C[n])
        for (int p = 0; p < hD; p++) {
            float sum = 0.0f;
            for (int n = 0; n < dS; n++) {
                float c_val = C[((b * L + t) * H + h) * dS + n];
                sum = __fadd_rn(sum, __fmul_rn(state[p * dS + n], c_val));
            }

            // Skip connection
            float x_val = x[((b * L + t) * H + h) * hD + p];
            sum = __fadd_rn(sum, __fmul_rn(D_h, x_val));

            // Gate (z_silu precomputed)
            float gate = z_silu[((b * L + t) * H + h) * hD + p];
            y[((b * L + t) * H + h) * hD + p] = __fmul_rn(sum, gate);
        }
    }
}

torch::Tensor ssm_scan_cuda(
    torch::Tensor inp,    // (B, L, H, hD, dS)
    torch::Tensor decay,  // (B, L, H)
    torch::Tensor C,      // (B, L, H, dS)
    torch::Tensor x,      // (B, L, H, hD)
    torch::Tensor z,      // (B, L, H, hD)
    torch::Tensor D       // (H,)
) {
    int B = inp.size(0);
    int L = inp.size(1);
    int H = inp.size(2);
    int hD = inp.size(3);
    int dS = inp.size(4);

    // Ensure fp32
    auto inp_f = inp.to(torch::kFloat32).contiguous();
    auto decay_f = decay.to(torch::kFloat32).contiguous();
    auto C_f = C.to(torch::kFloat32).contiguous();
    auto x_f = x.to(torch::kFloat32).contiguous();
    auto z_f = z.to(torch::kFloat32).contiguous();
    auto D_f = D.to(torch::kFloat32).contiguous();

    // Precompute silu OUTSIDE the kernel (matches CPU behavior)
    auto z_silu = z_f * torch::sigmoid(z_f);

    auto y = torch::empty({B, L, H, hD}, inp.options().dtype(torch::kFloat32));

    int total_threads = B * H;
    int block_size = 256;
    int grid_size = (total_threads + block_size - 1) / block_size;

    ssm_scan_kernel<<<grid_size, block_size>>>(
        inp_f.data_ptr<float>(),
        decay_f.data_ptr<float>(),
        C_f.data_ptr<float>(),
        x_f.data_ptr<float>(),
        z_silu.data_ptr<float>(),
        D_f.data_ptr<float>(),
        y.data_ptr<float>(),
        B, L, H, hD, dS
    );

    return y.to(inp.dtype());
}
"""

CPP_SOURCE = """
torch::Tensor ssm_scan_cuda(
    torch::Tensor inp, torch::Tensor decay, torch::Tensor C,
    torch::Tensor x, torch::Tensor z, torch::Tensor D);
"""

_module = None

def get_cuda_scan():
    """Load the inline CUDA kernel. Cached after first call."""
    global _module
    if _module is None:
        _module = load_inline(
            name="ssm_scan_cuda",
            cpp_sources=CPP_SOURCE,
            cuda_sources=CUDA_SOURCE,
            functions=["ssm_scan_cuda"],
            verbose=False,
            extra_cuda_cflags=["-O3", "--fmad=false"],  # disable FMA globally
        )
    return _module


def ssm_scan_cuda_nofma(inp, decay, C, x, z, D):
    """CUDA SSM scan with FMA disabled — exact fp32 arithmetic."""
    mod = get_cuda_scan()
    return mod.ssm_scan_cuda(inp, decay, C, x, z, D)
