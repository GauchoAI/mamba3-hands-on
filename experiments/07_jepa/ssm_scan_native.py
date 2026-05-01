"""Custom SSM Scan — pure PyTorch, correct on all hardware.

Replaces the Triton kernel as the default scan backend. The Triton kernel
has a precision bug (CPU 100% vs CUDA 54% on parity) caused by:
1. SiLU computed per-step vs precomputed
2. Tree reduction order in tl.sum vs sequential sum
3. Compounding error over L timesteps

This implementation:
- Always accumulates state in fp32
- Precomputes silu outside the loop
- Uses PyTorch's deterministic sum
- Works on CPU, MPS, and CUDA
- Can be accelerated via torch.compile

Usage:
    from ssm_scan_native import ssm_scan_native, ssm_scan_compiled
    y = ssm_scan_native(inp, decay, C, x, z, D)      # eager
    y = ssm_scan_compiled(inp, decay, C, x, z, D)     # compiled
"""

import torch


def ssm_scan_native(inp, decay, C, x, z, D):
    """Pure PyTorch SSM scan — single source of truth.

    Args:
        inp:   (B, L, H, hD, dS) — precomputed B*x*dt
        decay: (B, L, H)         — per-step decay exp(A*dt)
        C:     (B, L, H, dS)     — output projection (RoPE-rotated)
        x:     (B, L, H, hD)     — skip connection value
        z:     (B, L, H, hD)     — gate signal
        D:     (H,)              — skip connection weight

    Returns:
        y:     (B, L, H, hD)     — scan output
    """
    B, L, H, hD, dS = inp.shape

    # State: always fp32 for precision
    h = torch.zeros(B, H, hD, dS, device=inp.device, dtype=torch.float32)
    y = torch.empty(B, L, H, hD, device=inp.device, dtype=inp.dtype)

    # Precompute silu OUTSIDE the loop — critical for matching JIT precision
    z_silu = (z.float() * torch.sigmoid(z.float()))

    # Cast inputs to fp32 for accumulation
    inp_f32 = inp.float()
    decay_f32 = decay.float()
    C_f32 = C.float()
    x_f32 = x.float()
    D_f32 = D.float()

    for t in range(L):
        # State update: h ← decay*h + inp
        h = decay_f32[:, t, :, None, None] * h + inp_f32[:, t]

        # Output projection: y_t = sum(h * C) over dS
        y_t = (h * C_f32[:, t, :, None, :]).sum(dim=-1)

        # Skip connection
        y_t = y_t + D_f32[None, :, None] * x_f32[:, t]

        # Gate and cast back to output dtype
        y[:, t] = (y_t * z_silu[:, t]).to(inp.dtype)

    return y


# ── torch.compile wrappers ────────────────────────────────────────

_compiled_cache = {}


def ssm_scan_compiled(inp, decay, C, x, z, D):
    """Compiled version of ssm_scan_native. Cached per device type.

    On CUDA: uses reduce-overhead mode (CUDA graphs)
    On CPU/MPS: uses default inductor mode
    """
    device_type = inp.device.type
    if device_type not in _compiled_cache:
        try:
            if device_type == "cuda":
                _compiled_cache[device_type] = torch.compile(
                    ssm_scan_native,
                    mode="reduce-overhead",
                    fullgraph=False,
                )
            elif device_type == "mps":
                # MPS has limited torch.compile support
                _compiled_cache[device_type] = torch.compile(
                    ssm_scan_native,
                    backend="aot_eager",
                )
            else:
                _compiled_cache[device_type] = torch.compile(
                    ssm_scan_native,
                    mode="default",
                )
        except Exception:
            # Fallback to eager if compile fails
            _compiled_cache[device_type] = ssm_scan_native
    return _compiled_cache[device_type](inp, decay, C, x, z, D)
