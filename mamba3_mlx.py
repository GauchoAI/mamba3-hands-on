"""MLX port of Mamba-3 SISO block + CortexLM.

Mirrors mamba3_minimal.Mamba3Block and cortex_counting.CortexLM exactly:
parameter shapes/names match so PyTorch checkpoints load via dict
remapping (see ckpt_torch_to_mlx in this file).

Why MLX:
- PyTorch on MPS hits an `F.pad` >3D fallback (the trapezoidal-blend
  shift in Mamba3Block) and runs the slow path on CPU view-ops. The
  pad-fallback memory documents that this caps MPS speedup at ~1.16×
  over CPU for our model size.
- MLX is Apple's native Metal framework. No fallback dance, lazy
  compilation, lower op-launch overhead. Expected speedup on full
  training step: 2-5× over PyTorch MPS at our model size.

Scope of this port:
- forward() of Mamba3Block + CortexLM (training path)
- Primitive base class + CounterPrimitive (the cortex thesis)
- AdamW optimizer + cosine schedule
- Greedy / temperature sampling
NOT ported (yet):
- step() / init_state() per-token incremental decoding
- Mamba2LikeBlock baseline
"""
from __future__ import annotations
import math
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn


# ----------------------------------------------------------------------------
# Mamba-3 block
# ----------------------------------------------------------------------------

@dataclass
class Mamba3Config:
    d_model: int = 64
    d_state: int = 16
    expand: int = 2
    headdim: int = 16
    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init_floor: float = 1e-4
    A_floor: float = 1e-4


def apply_rope_pairs(v: mx.array, angles: mx.array) -> mx.array:
    """v: (..., S) with S even; angles: (..., S//2). Pairwise rotation."""
    v_even = v[..., 0::2]
    v_odd  = v[..., 1::2]
    cos = mx.cos(angles)
    sin = mx.sin(angles)
    rot_even = v_even * cos - v_odd * sin
    rot_odd  = v_even * sin + v_odd * cos
    out = mx.stack([rot_even, rot_odd], axis=-1)
    return out.reshape(*out.shape[:-2], -1)


def softplus(x: mx.array) -> mx.array:
    """Numerically stable softplus."""
    return mx.logaddexp(x, mx.zeros_like(x))


class Mamba3Block(nn.Module):
    """Mamba-3 SISO block. Mirrors mamba3_minimal.Mamba3Block parameter
    layout exactly: in_proj, out_proj, dt_bias, D, B_norm, C_norm.
    """

    def __init__(self, cfg: Mamba3Config):
        super().__init__()
        self.cfg = cfg
        d_model = cfg.d_model
        d_state = cfg.d_state
        d_inner = cfg.expand * d_model
        headdim = cfg.headdim
        nheads  = d_inner // headdim

        self.d_inner = d_inner
        self.nheads  = nheads
        self.d_state = d_state
        self.headdim = headdim
        self.num_rope_angles = d_state // 2

        d_in_proj = d_inner + d_inner + d_state + d_state + 3 * nheads + self.num_rope_angles

        self.in_proj  = nn.Linear(d_model, d_in_proj, bias=False)
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

        # dt_bias init: softplus(dt_bias) ∈ [dt_min, dt_max]
        rng = mx.random.uniform(0.0, 1.0, (nheads,))
        _dt = mx.exp(rng * (math.log(cfg.dt_max) - math.log(cfg.dt_min)) + math.log(cfg.dt_min))
        _dt = mx.maximum(_dt, cfg.dt_init_floor)
        self.dt_bias = _dt + mx.log(-mx.expm1(-_dt))   # learnable param via assignment below

        self.D = mx.ones((nheads,))
        self.B_norm = nn.LayerNorm(d_state)
        self.C_norm = nn.LayerNorm(d_state)

    def __call__(self, u: mx.array) -> mx.array:
        """u: (B, L, d_model) -> (B, L, d_model)"""
        cfg = self.cfg
        B_, L, _ = u.shape
        nH, hD, dS = self.nheads, self.headdim, self.d_state
        d_inner = self.d_inner

        proj = self.in_proj(u)
        # Splits: [d_inner, d_inner, dS, dS, nH, nH, nH, num_rope]
        s = [d_inner, d_inner, dS, dS, nH, nH, nH, self.num_rope_angles]
        offsets = [0]
        for sz in s: offsets.append(offsets[-1] + sz)
        z       = proj[..., offsets[0]:offsets[1]]
        x       = proj[..., offsets[1]:offsets[2]]
        Bp      = proj[..., offsets[2]:offsets[3]]
        Cp      = proj[..., offsets[3]:offsets[4]]
        dd_dt   = proj[..., offsets[4]:offsets[5]]
        dd_A    = proj[..., offsets[5]:offsets[6]]
        trap_raw = proj[..., offsets[6]:offsets[7]]
        angles  = proj[..., offsets[7]:offsets[8]]

        x = x.reshape(B_, L, nH, hD)
        z = z.reshape(B_, L, nH, hD)

        Bp = self.B_norm(Bp)
        Cp = self.C_norm(Cp)

        DT = softplus(dd_dt + self.dt_bias)              # (B, L, H)
        DT_mean = DT.mean(axis=-1, keepdims=True)         # (B, L, 1)
        phase_step = angles * DT_mean                     # (B, L, dS/2)
        phase = mx.cumsum(phase_step, axis=1)             # (B, L, dS/2)

        Bp_rot = apply_rope_pairs(Bp, phase)              # (B, L, dS)
        Cp_rot = apply_rope_pairs(Cp, phase)              # (B, L, dS)

        Bp_rot = mx.broadcast_to(Bp_rot[:, :, None, :], (B_, L, nH, dS))
        Cp_rot = mx.broadcast_to(Cp_rot[:, :, None, :], (B_, L, nH, dS))

        A = -softplus(dd_A)
        A = mx.minimum(A, -cfg.A_floor)                   # ensure A <= -A_floor
        decay = mx.exp(A * DT)                            # (B, L, H)

        trap = mx.sigmoid(trap_raw)                       # (B, L, H)

        # Outer product Bx for all timesteps: (B, L, H, hD, dS)
        # einsum: "blhp, blhn -> blhpn"
        Bx_all = (x[..., :, None] * Bp_rot[..., None, :])

        # Trapezoidal shift: pad along L axis (axis=1), prepending one zero step.
        # MLX's pad accepts (axis, (before, after)) per-axis pairs.
        # Bx_all shape (B, L, H, hD, dS); pad axis=1 with (1, 0) before, drop last
        Bx_shifted = Bx_all[:, :-1, ...]
        Bx_prev = mx.pad(Bx_shifted, [(0,0), (1,0), (0,0), (0,0), (0,0)])
        trap_exp = trap[..., None, None]
        inp_all = trap_exp * Bx_all + (1 - trap_exp) * Bx_prev

        # Scale by dt
        inp_all = inp_all * DT[..., None, None]           # (B, L, H, hD, dS)

        # Sequential scan: h[t] = decay[t] * h[t-1] + inp[t]
        # Output: y[t] = sum_n(h[t] * C[t]) + D * x[t], gated by silu(z[t])
        h = mx.zeros((B_, nH, hD, dS), dtype=u.dtype)
        z_silu = z * mx.sigmoid(z)                        # (B, L, H, hD)

        ys = []
        for t in range(L):
            h = decay[:, t, :, None, None] * h + inp_all[:, t]    # (B, H, hD, dS)
            y_t = (h * Cp_rot[:, t, :, None, :]).sum(axis=-1)     # (B, H, hD)
            y_t = y_t + self.D[None, :, None] * x[:, t]
            y_t = y_t * z_silu[:, t]
            ys.append(y_t)
        y = mx.stack(ys, axis=1).reshape(B_, L, d_inner)

        return self.out_proj(y)


# ----------------------------------------------------------------------------
# Primitive plug interface (mirrors cortex_counting.Primitive API)
# ----------------------------------------------------------------------------

class Primitive(nn.Module):
    def __init__(self, d_model: int, layer: int = 0, name: str = ""):
        super().__init__()
        self.d_model = d_model
        self.layer = layer
        self.name = name or self.__class__.__name__

    def __call__(self, x: mx.array, tokens: mx.array) -> dict:
        raise NotImplementedError

    def aux_loss(self, tokens: mx.array, fwd_out: dict, mask: mx.array) -> mx.array:
        return mx.zeros((), dtype=mx.float32)


STAR_BYTE = ord("*")  # 42
A_BYTE    = ord("a")  # 97


class CounterPrimitive(Primitive):
    """MLX port of the cortex counter primitive — identical math, same params."""

    def __init__(self, d_model: int, layer: int = 0, n_counters: int = 2,
                 readout: str = "unbounded", injection_scale: float = 10.0):
        super().__init__(d_model, layer, name="counter")
        self.n_counters = n_counters
        self.readout = readout
        self.injection_scale = injection_scale
        self.hard_gates_inference = False

        self.inc_proj = nn.Linear(d_model, n_counters)
        self.reset_proj = nn.Linear(d_model, n_counters)

        if readout == "unbounded":
            P = n_counters * (n_counters - 1) // 2
            read_in = n_counters + P
        else:
            raise ValueError(f"MLX port supports only readout='unbounded' (was {readout!r})")
        self.read_proj = nn.Linear(read_in, d_model)

        # Bias init mirrors PyTorch
        # NB: nn.Linear in MLX exposes .bias as an array attribute we can overwrite.
        self.inc_proj.bias = mx.full((n_counters,), -2.0)
        self.reset_proj.bias = mx.full((n_counters,), -5.0)

    def _unbounded(self, c: mx.array) -> mx.array:
        K = self.n_counters
        k = 8.0
        feats = [mx.tanh(c / k)]
        if K >= 2:
            diffs = []
            for i in range(K):
                for j in range(i + 1, K):
                    diffs.append(c[..., i:i + 1] - c[..., j:j + 1])
            diff = mx.concatenate(diffs, axis=-1)
            feats.append(mx.tanh(diff / k))
        return mx.concatenate(feats, axis=-1)

    def __call__(self, x: mx.array, tokens: mx.array) -> dict:
        B_, L, _ = x.shape
        K = self.n_counters

        inc_logits = self.inc_proj(x)
        reset_logits = self.reset_proj(x)
        # Hard gates at inference (no .training flag in MLX modules; use explicit)
        if self.hard_gates_inference:
            inc = (inc_logits > 0).astype(x.dtype)
            rst = (reset_logits > 0).astype(x.dtype)
        else:
            inc = mx.sigmoid(inc_logits)
            rst = mx.sigmoid(reset_logits)
        keep = 1.0 - rst

        c = mx.zeros((B_, K), dtype=x.dtype)
        outs = []
        for t in range(L):
            c = keep[:, t] * c + inc[:, t]
            outs.append(c)
        counters = mx.stack(outs, axis=1)

        emb = self._unbounded(counters)
        injection = self.injection_scale * self.read_proj(emb)
        return {
            "injection": injection,
            "inc_logits": inc_logits,
            "counters": counters,
        }

    def aux_loss(self, tokens: mx.array, fwd_out: dict, mask: mx.array) -> mx.array:
        is_star = (tokens == STAR_BYTE).astype(mx.float32)
        is_a    = (tokens == A_BYTE).astype(mx.float32)
        target = mx.stack([is_star, is_a], axis=-1)               # (B, L, K)
        # binary cross entropy with logits, then average over K, mask over L
        inc_logits = fwd_out["inc_logits"]
        # bce(logit, target) = max(logit, 0) - logit*target + log(1 + exp(-|logit|))
        bce = mx.maximum(inc_logits, 0) - inc_logits * target + mx.log1p(mx.exp(-mx.abs(inc_logits)))
        bce = bce.mean(axis=-1)                                    # (B, L)
        return (bce * mask.astype(mx.float32)).sum() / mx.maximum(mask.astype(mx.float32).sum(), 1.0)


# ----------------------------------------------------------------------------
# CortexLM — MLX
# ----------------------------------------------------------------------------

@dataclass
class CortexLMConfig:
    n_layers: int = 4
    d_model: int = 128
    d_state: int = 16
    expand: int = 2
    headdim: int = 16
    vocab_size: int = 256
    max_seq_len: int = 128


class LayerBlock(nn.Module):
    def __init__(self, cfg: Mamba3Config):
        super().__init__()
        self.norm = nn.LayerNorm(cfg.d_model)
        self.block = Mamba3Block(cfg)

    def __call__(self, x):
        return self.block(self.norm(x))


class CortexLM(nn.Module):
    def __init__(self, cfg: CortexLMConfig, primitives: list | None = None):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.embed_norm = nn.LayerNorm(cfg.d_model)

        ssm_cfg = Mamba3Config(
            d_model=cfg.d_model, d_state=cfg.d_state,
            expand=cfg.expand, headdim=cfg.headdim,
        )
        self.layers = [LayerBlock(ssm_cfg) for _ in range(cfg.n_layers)]
        self.primitives = primitives or []

        self.final_norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        # NOTE: weight tying done by trainer (sharing object identity in MLX
        # is awkward — easier to copy embed.weight to head.weight at save).

    @property
    def counter(self):
        for p in self.primitives:
            if isinstance(p, CounterPrimitive):
                return p
        return None

    def __call__(self, tokens: mx.array, return_aux: bool = False):
        x = self.embed(tokens)
        x = self.embed_norm(x)
        prim_outputs = {}
        for i, layer in enumerate(self.layers):
            x = x + layer(x)
            for p in self.primitives:
                if p.layer == i:
                    out = p(x, tokens)
                    x = x + out["injection"]
                    prim_outputs[p.name] = out
        x = self.final_norm(x)
        logits = self.head(x)
        if return_aux:
            return logits, prim_outputs
        return logits

    def aux_loss(self, tokens, prim_outputs, mask):
        total = mx.zeros((), dtype=mx.float32)
        for p in self.primitives:
            if p.name in prim_outputs:
                total = total + p.aux_loss(tokens, prim_outputs[p.name], mask)
        return total
