"""
Minimal pure-PyTorch Mamba-3 block (SISO variant).

Implements the three Mamba-3 innovations from Lahoti et al. 2026:
  1. Trapezoidal (2nd-order) discretization via a data-dependent gate `trap`
  2. Complex-valued dynamics via data-dependent RoPE applied to B and C
  3. (MIMO omitted here; SISO rank=1 version for clarity)

This is a learning / research-grade implementation:
  - Sequential scan over the sequence (no chunked parallel scan)
  - Runs on CPU/MPS/CUDA (no custom kernels)
  - Matches the parameter layout of the official state-spaces/mamba3.py
    closely enough that the math is identifiable.

Shapes follow the official convention:
  u:     (B, L, d_model)
  After in_proj, split into:
    z        -> (B, L, d_inner)       gating
    x        -> (B, L, d_inner)       value
    Bproj    -> (B, L, d_state)       key   (will be rotated via RoPE)
    Cproj    -> (B, L, d_state)       query (will be rotated via RoPE)
    dd_dt    -> (B, L, nheads)        per-head delta-t
    dd_A     -> (B, L, nheads)        per-head A
    trap     -> (B, L, nheads)        per-head trapezoidal gate (sigmoid)
    angles   -> (B, L, num_angles)    RoPE angles (shared across heads)
"""
from __future__ import annotations
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Mamba3Config:
    d_model: int = 64
    d_state: int = 16        # N in the paper
    expand: int = 2
    headdim: int = 16
    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init_floor: float = 1e-4
    A_floor: float = 1e-4
    rope_fraction: float = 1.0   # fraction of d_state to rotate (1.0 = all)


def apply_rope_pairs(v: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    """
    Apply pairwise rotation to v.
    v:      (..., S)     where S is even
    angles: (..., S // 2)
    Returns v rotated so that pairs (v_{2k}, v_{2k+1}) are rotated by angles[..., k].
    """
    S = v.shape[-1]
    assert S % 2 == 0, "d_state must be even for pairwise RoPE"
    v_even = v[..., 0::2]   # (..., S/2)
    v_odd  = v[..., 1::2]   # (..., S/2)
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    rot_even = v_even * cos - v_odd * sin
    rot_odd  = v_even * sin + v_odd * cos
    out = torch.stack([rot_even, rot_odd], dim=-1)   # (..., S/2, 2)
    return out.flatten(-2)                            # (..., S)


class Mamba3Block(nn.Module):
    def __init__(self, cfg: Mamba3Config, use_rope: bool = True, use_trap: bool = True):
        super().__init__()
        self.cfg = cfg
        self.use_rope = use_rope
        self.use_trap = use_trap
        d_model  = cfg.d_model
        d_state  = cfg.d_state
        d_inner  = cfg.expand * d_model
        headdim  = cfg.headdim
        assert d_inner % headdim == 0, "d_inner must be divisible by headdim"
        nheads   = d_inner // headdim

        self.d_inner = d_inner
        self.nheads  = nheads
        self.d_state = d_state
        self.headdim = headdim

        # RoPE angles: pairs of (d_state // 2) angles, shared across heads
        assert d_state % 2 == 0
        self.num_rope_angles = d_state // 2

        # One big input projection: [z, x, B, C, dd_dt, dd_A, trap, angles]
        d_in_proj = (
            d_inner            # z
            + d_inner          # x
            + d_state          # B
            + d_state          # C
            + nheads           # dd_dt
            + nheads           # dd_A
            + nheads           # trap
            + self.num_rope_angles  # angles
        )
        self.in_proj  = nn.Linear(d_model, d_in_proj, bias=False)
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

        # dt_bias initialised so softplus(dt_bias) spans [dt_min, dt_max]
        _dt = torch.exp(
            torch.rand(nheads) * (math.log(cfg.dt_max) - math.log(cfg.dt_min))
            + math.log(cfg.dt_min)
        ).clamp(min=cfg.dt_init_floor)
        dt_bias = _dt + torch.log(-torch.expm1(-_dt))
        self.dt_bias = nn.Parameter(dt_bias)

        # D skip connection (per head)
        self.D = nn.Parameter(torch.ones(nheads))

        # Lightweight LayerNorms on B and C (the official uses RMSNormGated)
        self.B_norm = nn.LayerNorm(d_state)
        self.C_norm = nn.LayerNorm(d_state)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """
        u: (B, L, d_model) -> (B, L, d_model)
        """
        B_, L, _ = u.shape
        cfg = self.cfg
        nH, hD, dS = self.nheads, self.headdim, self.d_state

        proj = self.in_proj(u)
        splits = [
            self.d_inner, self.d_inner,
            dS, dS,
            nH, nH, nH,
            self.num_rope_angles,
        ]
        z, x, Bp, Cp, dd_dt, dd_A, trap_raw, angles = torch.split(proj, splits, dim=-1)

        # Reshape x and z to (B, L, H, hD)
        x = x.view(B_, L, nH, hD)
        z = z.view(B_, L, nH, hD)

        # Normalise B and C
        Bp = self.B_norm(Bp)   # (B, L, dS)
        Cp = self.C_norm(Cp)   # (B, L, dS)

        # ---- (Innovation 2) Data-dependent RoPE on B and C ----
        # angles: (B, L, dS/2). We integrate angles along time via dt to get
        # a cumulative phase, mimicking a complex eigenvalue e^{i * omega * dt}.
        # DT: (B, L, H), softplus(dd_dt + dt_bias). We use per-head DT to modulate
        # the phase — here we average across heads for a single phase schedule,
        # matching the SISO assumption that angles are shared across heads.
        DT = F.softplus(dd_dt + self.dt_bias)              # (B, L, H)
        DT_mean = DT.mean(dim=-1, keepdim=True)             # (B, L, 1) shared phase rate
        phase_step = angles * DT_mean                       # (B, L, dS/2)
        phase = torch.cumsum(phase_step, dim=1)             # (B, L, dS/2)

        if self.use_rope:
            Bp_rot = apply_rope_pairs(Bp, phase)            # (B, L, dS)
            Cp_rot = apply_rope_pairs(Cp, phase)            # (B, L, dS)
        else:
            Bp_rot, Cp_rot = Bp, Cp                          # ablate RoPE

        # Broadcast B, C across heads: (B, L, H, dS)
        Bp_rot = Bp_rot.unsqueeze(2).expand(B_, L, nH, dS)
        Cp_rot = Cp_rot.unsqueeze(2).expand(B_, L, nH, dS)

        # ---- Compute A * DT (continuous-time decay per head per step) ----
        # A is negative; parameterised as -softplus(dd_A), clamped so A <= -A_floor.
        # Parentheses matter: `-softplus(..).clamp(max=-f)` parses as -(clamp(..)), flipping sign.
        A = (-F.softplus(dd_A)).clamp(max=-cfg.A_floor)     # (B, L, H), <= -A_floor
        ADT = A * DT                                        # (B, L, H)
        decay = torch.exp(ADT)                              # (B, L, H) — e^{A dt}

        # ---- (Innovation 1) Trapezoidal gate ----
        # trap in [0, 1] mixes current and previous input contributions.
        # When trap == 1 we get Euler (Mamba-2); when trap < 1 we blend
        # previous x into the state, i.e. 2nd-order discretization.
        trap = torch.sigmoid(trap_raw)                      # (B, L, H)

        # ---- Precompute all inputs in one batched operation ----
        # Outer product B*x for all timesteps: (B, L, H, hD, dS)
        Bx_all = torch.einsum("blhp,blhn->blhpn", x, Bp_rot)

        if self.use_trap:
            # Trapezoidal: blend current and shifted-previous inputs
            Bx_prev = F.pad(Bx_all[:, :-1], (0,0, 0,0, 0,0, 1,0))  # shift right, zero-pad
            trap_exp = trap[..., None, None]  # (B, L, H, 1, 1)
            inp_all = trap_exp * Bx_all + (1 - trap_exp) * Bx_prev
        else:
            inp_all = Bx_all

        # Scale by dt
        inp_all = inp_all * DT[..., None, None]  # (B, L, H, hD, dS)

        # ---- Sequential scan (tight loop — only state recurrence) ----
        h = u.new_zeros(B_, nH, hD, dS)
        y = u.new_zeros(B_, L, nH, hD)

        # Pre-slice for less Python overhead in the loop
        decay_exp = decay[..., None, None]  # (B, L, H, 1, 1)
        z_silu = F.silu(z)  # (B, L, H, hD) — precompute all gating

        for t in range(L):
            h = decay_exp[:, t] * h + inp_all[:, t]
            y_t = torch.einsum("bhpn,bhn->bhp", h, Cp_rot[:, t])
            y_t = y_t + self.D[None, :, None] * x[:, t]
            y[:, t] = y_t * z_silu[:, t]

        y = y.reshape(B_, L, self.d_inner)

        out = self.out_proj(y)
        return out


# ---------------- A Mamba-2-style baseline (no RoPE, Euler) ----------------

class Mamba2LikeBlock(nn.Module):
    """Simplified real-scalar SSM baseline: no RoPE, Euler discretization.
    Shares the block structure of Mamba3Block but drops innovations 1 and 2.
    """
    def __init__(self, cfg: Mamba3Config):
        super().__init__()
        self.cfg = cfg
        d_model, d_state, headdim = cfg.d_model, cfg.d_state, cfg.headdim
        d_inner = cfg.expand * d_model
        nheads = d_inner // headdim

        self.d_inner, self.nheads, self.d_state, self.headdim = d_inner, nheads, d_state, headdim

        d_in_proj = d_inner + d_inner + d_state + d_state + nheads + nheads
        self.in_proj  = nn.Linear(d_model, d_in_proj, bias=False)
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

        _dt = torch.exp(
            torch.rand(nheads) * (math.log(cfg.dt_max) - math.log(cfg.dt_min))
            + math.log(cfg.dt_min)
        ).clamp(min=cfg.dt_init_floor)
        self.dt_bias = nn.Parameter(_dt + torch.log(-torch.expm1(-_dt)))
        self.D = nn.Parameter(torch.ones(nheads))
        self.B_norm = nn.LayerNorm(d_state)
        self.C_norm = nn.LayerNorm(d_state)

    def forward(self, u):
        B_, L, _ = u.shape
        nH, hD, dS = self.nheads, self.headdim, self.d_state

        proj = self.in_proj(u)
        z, x, Bp, Cp, dd_dt, dd_A = torch.split(
            proj,
            [self.d_inner, self.d_inner, dS, dS, nH, nH],
            dim=-1,
        )
        x = x.view(B_, L, nH, hD); z = z.view(B_, L, nH, hD)
        Bp = self.B_norm(Bp); Cp = self.C_norm(Cp)
        Bp = Bp.unsqueeze(2).expand(B_, L, nH, dS)
        Cp = Cp.unsqueeze(2).expand(B_, L, nH, dS)

        DT  = F.softplus(dd_dt + self.dt_bias)
        A   = (-F.softplus(dd_A)).clamp(max=-self.cfg.A_floor)
        dec = torch.exp(A * DT)

        h = u.new_zeros(B_, nH, hD, dS)
        ys = []
        for t in range(L):
            inp = torch.einsum("bhp,bhn->bhpn", x[:, t], Bp[:, t]) * DT[:, t][..., None, None]
            h = dec[:, t][..., None, None] * h + inp
            y_t = torch.einsum("bhpn,bhn->bhp", h, Cp[:, t]) + self.D[None, :, None] * x[:, t]
            y_t = y_t * F.silu(z[:, t])
            ys.append(y_t)
        y = torch.stack(ys, dim=1).reshape(B_, L, self.d_inner)
        return self.out_proj(y)


if __name__ == "__main__":
    # Smoke test: forward pass and backward pass
    torch.manual_seed(0)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    cfg = Mamba3Config(d_model=32, d_state=16, expand=2, headdim=16)
    blk = Mamba3Block(cfg).to(device)
    u = torch.randn(2, 12, cfg.d_model, device=device)
    y = blk(u)
    print(f"[Mamba3] in={tuple(u.shape)} out={tuple(y.shape)} device={device}")
    y.sum().backward()
    print(f"[Mamba3] backward OK. n_params={sum(p.numel() for p in blk.parameters()):,}")

    blk2 = Mamba2LikeBlock(cfg).to(device)
    y2 = blk2(u)
    print(f"[Mamba2-like] in={tuple(u.shape)} out={tuple(y2.shape)}")
    print(f"[Mamba2-like] n_params={sum(p.numel() for p in blk2.parameters()):,}")
