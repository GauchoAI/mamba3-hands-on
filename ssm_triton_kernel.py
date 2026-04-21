"""
Triton kernel for SSM scan — one thread block per (batch, head).

This file only imports when Triton is available (CUDA).
"""
import triton
import triton.language as tl


@triton.jit
def _ssm_scan_fwd_kernel(
    # Pointers
    inp_ptr,    # (B, L, H, hD, dS) — precomputed input
    decay_ptr,  # (B, L, H)         — e^{A*dt}
    C_ptr,      # (B, L, H, dS)     — output query (rotated)
    x_ptr,      # (B, L, H, hD)     — value for skip connection
    z_ptr,      # (B, L, H, hD)     — gating value
    D_ptr,      # (H,)              — skip parameter
    y_ptr,      # (B, L, H, hD)     — output
    # Dimensions
    B: tl.constexpr, L: tl.constexpr, H: tl.constexpr,
    hD: tl.constexpr, dS: tl.constexpr,
    # Strides for inp (B, L, H, hD, dS)
    inp_stride_b, inp_stride_l, inp_stride_h, inp_stride_p, inp_stride_n,
    # Strides for decay (B, L, H)
    dec_stride_b, dec_stride_l, dec_stride_h,
    # Strides for C (B, L, H, dS)
    C_stride_b, C_stride_l, C_stride_h, C_stride_n,
    # Strides for x (B, L, H, hD)
    x_stride_b, x_stride_l, x_stride_h, x_stride_p,
    # Strides for z (same as x)
    z_stride_b, z_stride_l, z_stride_h, z_stride_p,
    # Strides for y (same as x)
    y_stride_b, y_stride_l, y_stride_h, y_stride_p,
):
    """One thread block per (batch, head) pair."""
    pid = tl.program_id(0)
    b = pid // H
    head = pid % H

    D_val = tl.load(D_ptr + head)

    p_offs = tl.arange(0, hD)
    n_offs = tl.arange(0, dS)

    # State h[hD, dS] in registers — zero init
    h = tl.zeros((hD, dS), dtype=tl.float32)

    for t in range(L):
        # Load decay: scalar per (batch, step, head)
        dec = tl.load(decay_ptr + b * dec_stride_b + t * dec_stride_l + head * dec_stride_h)

        # Load inp[b, t, head, :, :] — (hD, dS)
        inp_base = b * inp_stride_b + t * inp_stride_l + head * inp_stride_h
        inp_block = tl.load(
            inp_ptr + inp_base +
            p_offs[:, None] * inp_stride_p +
            n_offs[None, :] * inp_stride_n
        )

        # State update
        h = dec * h + inp_block

        # Load C[b, t, head, :] — (dS,)
        C_base = b * C_stride_b + t * C_stride_l + head * C_stride_h
        C_vec = tl.load(C_ptr + C_base + n_offs * C_stride_n)

        # Output: y_t[p] = sum_n h[p, n] * C[n]
        y_t = tl.sum(h * C_vec[None, :], axis=1)  # (hD,)

        # Load x and z
        xz_base_x = b * x_stride_b + t * x_stride_l + head * x_stride_h
        xz_base_z = b * z_stride_b + t * z_stride_l + head * z_stride_h
        x_vec = tl.load(x_ptr + xz_base_x + p_offs * x_stride_p)
        z_vec = tl.load(z_ptr + xz_base_z + p_offs * z_stride_p)

        # Skip + gate: y = (y + D*x) * silu(z)
        y_t = y_t + D_val * x_vec
        z_sig = 1.0 / (1.0 + tl.exp(-z_vec))
        y_t = y_t * z_vec * z_sig

        # Store y[b, t, head, :]
        y_base = b * y_stride_b + t * y_stride_l + head * y_stride_h
        tl.store(y_ptr + y_base + p_offs * y_stride_p, y_t)
