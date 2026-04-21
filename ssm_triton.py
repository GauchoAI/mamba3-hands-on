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

if HAS_TRITON:
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

    # Process each (p, n) element of the hD x dS state matrix
    # We tile over p (hD) — each thread handles one p row
    p_offs = tl.arange(0, hD)
    n_offs = tl.arange(0, dS)

    # State: h[p, n] — stored in registers
    # Initialize to zero
    # We process one p-row at a time across all n
    # h shape: (hD, dS) but we flatten to process in tiles

    # For small hD and dS (≤32), we can keep full state in registers
    # h is (hD, dS) — flatten to hD*dS
    h = tl.zeros((hD, dS), dtype=tl.float32)

    for t in range(L):
        # Load decay for this timestep: scalar per head
        dec = tl.load(decay_ptr + b * dec_stride_b + t * dec_stride_l + head * dec_stride_h)

        # Load inp[b, t, head, :, :] — (hD, dS)
        inp_base = b * inp_stride_b + t * inp_stride_l + head * inp_stride_h
        inp_block = tl.load(
            inp_ptr + inp_base +
            p_offs[:, None] * inp_stride_p +
            n_offs[None, :] * inp_stride_n
        )  # (hD, dS)

        # State update: h = decay * h + inp
        h = dec * h + inp_block

        # Load C[b, t, head, :] — (dS,)
        C_base = b * C_stride_b + t * C_stride_l + head * C_stride_h
        C_vec = tl.load(C_ptr + C_base + n_offs * C_stride_n)  # (dS,)

        # Output: y_t[p] = sum_n h[p, n] * C[n]
        y_t = tl.sum(h * C_vec[None, :], axis=1)  # (hD,)

        # Load x[b, t, head, :] and z[b, t, head, :] — (hD,)
        xz_base_x = b * x_stride_b + t * x_stride_l + head * x_stride_h
        xz_base_z = b * z_stride_b + t * z_stride_l + head * z_stride_h
        x_vec = tl.load(x_ptr + xz_base_x + p_offs * x_stride_p)
        z_vec = tl.load(z_ptr + xz_base_z + p_offs * z_stride_p)

        # Skip + gate: y = (y + D*x) * silu(z)
        y_t = y_t + D_val * x_vec
        z_sigmoid = tl.sigmoid(z_vec)
        y_t = y_t * z_vec * z_sigmoid  # silu(z) = z * sigmoid(z)

        # Store y[b, t, head, :]
        y_base = b * y_stride_b + t * y_stride_l + head * y_stride_h
        tl.store(y_ptr + y_base + p_offs * y_stride_p, y_t)


def ssm_scan_triton(inp, decay, C, x, z, D):
    """
    Triton-accelerated SSM scan.

    Args:
        inp:   (B, L, H, hD, dS) — precomputed input (already scaled by dt)
        decay: (B, L, H)         — e^{A*dt} decay factors
        C:     (B, L, H, dS)     — output query vectors
        x:     (B, L, H, hD)     — value vectors (for skip connection)
        z:     (B, L, H, hD)     — gating vectors
        D:     (H,)              — skip parameter

    Returns:
        y:     (B, L, H, hD)     — output
    """
    B, L, H, hD, dS = inp.shape
    y = torch.empty(B, L, H, hD, device=inp.device, dtype=inp.dtype)

    # Make contiguous
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


def ssm_scan(inp, decay, C, x, z, D):
    """
    Dispatch to Triton (CUDA) or JIT (MPS/CPU).
    """
    if inp.is_cuda:
        try:
            return ssm_scan_triton(inp, decay, C, x, z, D)
        except Exception:
            pass
    return ssm_scan_jit(inp, decay, C, x, z, D)


# ── Test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    B, L, H, hD, dS = 4, 16, 8, 16, 16
    inp = torch.randn(B, L, H, hD, dS, device=device)
    decay = torch.sigmoid(torch.randn(B, L, H, device=device))
    C = torch.randn(B, L, H, dS, device=device)
    x = torch.randn(B, L, H, hD, device=device)
    z = torch.randn(B, L, H, hD, device=device)
    D = torch.ones(H, device=device)

    # Test JIT
    y_jit = ssm_scan_jit(inp, decay, C, x, z, D)
    print(f"JIT:    {y_jit.shape} — OK")

    if device == "cuda":
        y_tri = ssm_scan_triton(inp, decay, C, x, z, D)
        print(f"Triton: {y_tri.shape} — OK")

        # Compare
        diff = (y_jit - y_tri).abs().max().item()
        print(f"Max diff: {diff:.6f}")

        # Benchmark
        import time
        # Warmup
        for _ in range(10):
            ssm_scan_triton(inp, decay, C, x, z, D)
        torch.cuda.synchronize()

        # Large batch
        B_big = 4096
        inp_big = torch.randn(B_big, L, H, hD, dS, device=device)
        decay_big = torch.sigmoid(torch.randn(B_big, L, H, device=device))
        C_big = torch.randn(B_big, L, H, dS, device=device)
        x_big = torch.randn(B_big, L, H, hD, device=device)
        z_big = torch.randn(B_big, L, H, hD, device=device)

        # Warmup
        for _ in range(3):
            ssm_scan_triton(inp_big, decay_big, C_big, x_big, z_big, D)
        torch.cuda.synchronize()

        t0 = time.time()
        N = 100
        for _ in range(N):
            ssm_scan_triton(inp_big, decay_big, C_big, x_big, z_big, D)
        torch.cuda.synchronize()
        elapsed = (time.time() - t0) / N * 1000
        print(f"Triton B={B_big} L={L}: {elapsed:.2f}ms/call")
