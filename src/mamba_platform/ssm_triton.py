"""
Triton kernel for Mamba-3 selective scan.

Replaces the Python for-loop with a GPU-native scan.
Each thread block handles one (batch, head) pair and loops
over the sequence length L entirely on-GPU.

The recurrence:
    h[t] = decay[t] * h[t-1] + inp[t]     (state update)
    y[t] = einsum("pn,n->p", h[t], C[t])  (output projection)
    y[t] = (y[t] + D * x[t]) * silu(z[t]) (skip + gate)

For short sequences (L < 128) this is bandwidth-bound, so we want
maximum parallelism over (batch * heads) dimension — exactly what
we get with batch=4096 and nheads=16 → 65K thread blocks.
"""
import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


# ── Fallback: torch.jit.script for non-Triton devices (MPS/CPU) ────

@torch.jit.script
def ssm_scan_jit(
    inp: torch.Tensor,     # (B, L, H, hD, dS)
    decay: torch.Tensor,   # (B, L, H)
    C: torch.Tensor,       # (B, L, H, dS)
    x: torch.Tensor,       # (B, L, H, hD)
    z: torch.Tensor,       # (B, L, H, hD)
    D: torch.Tensor,       # (H,)
) -> torch.Tensor:
    """JIT-compiled scan for MPS/CPU — no Python loop overhead."""
    B, L, H, hD, dS = inp.shape
    h = torch.zeros(B, H, hD, dS, device=inp.device, dtype=inp.dtype)
    y = torch.empty(B, L, H, hD, device=inp.device, dtype=inp.dtype)

    z_silu = z * torch.sigmoid(z)  # precompute silu

    for t in range(L):
        h = decay[:, t, :, None, None] * h + inp[:, t]
        # y_t = einsum("bhpn,bhn->bhp" = sum over n)
        y_t = (h * C[:, t, :, None, :]).sum(dim=-1)  # (B, H, hD)
        y_t = y_t + D[None, :, None] * x[:, t]
        y[:, t] = y_t * z_silu[:, t]

    return y


# Backend override: set to force a specific backend.
# Controlled by config["scan_backend"] via specialist_trainer.
# Options: None (auto), "native", "compiled", "jit", "triton"
FORCE_BACKEND = None


def ssm_scan(inp, decay, C, x, z, D):
    """
    Dispatch SSM scan to the best available backend.

    Default: native (correct on all hardware).
    Triton is available but has precision bugs — only used if explicitly forced.
    """
    if FORCE_BACKEND == "native":
        from .ssm_scan_native import ssm_scan_native
        return ssm_scan_native(inp, decay, C, x, z, D)
    if FORCE_BACKEND == "compiled":
        from .ssm_scan_native import ssm_scan_compiled
        return ssm_scan_compiled(inp, decay, C, x, z, D)
    if FORCE_BACKEND == "jit":
        return ssm_scan_jit(inp, decay, C, x, z, D)
    if FORCE_BACKEND == "triton" and inp.is_cuda and HAS_TRITON:
        return ssm_scan_triton(inp, decay, C, x, z, D)
    # Auto: use native (correct) on all devices.
    # Triton is NOT the default — it has precision bugs.
    from .ssm_scan_native import ssm_scan_native
    return ssm_scan_native(inp, decay, C, x, z, D)


# ── Triton kernel (only defined when triton is available) ───────────

if HAS_TRITON:
    from .ssm_triton_kernel import _ssm_scan_fwd_kernel

    def ssm_scan_triton(inp, decay, C, x, z, D):
        """Triton-accelerated SSM scan."""
        B, L, H, hD, dS = inp.shape
        y = torch.empty(B, L, H, hD, device=inp.device, dtype=inp.dtype)

        inp = inp.contiguous()
        decay = decay.contiguous()
        C = C.contiguous()
        x = x.contiguous()
        z = z.contiguous()

        grid = (B * H,)

        _ssm_scan_fwd_kernel[grid](
            inp, decay, C, x, z, D, y,
            B, L, H, hD, dS,
            *inp.stride(),
            *decay.stride(),
            *C.stride(),
            *x.stride(),
            *z.stride(),
            *y.stride(),
        )
        return y
