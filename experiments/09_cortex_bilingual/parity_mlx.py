"""Numerical parity check: MLX vs PyTorch on identical inputs.

Builds a small CortexLM in PyTorch, copies its weights into the MLX
port, runs forward on the same input, asserts max-abs-diff is below
tolerance. This is the gate before any benchmarking — if outputs
diverge, the speedup numbers are meaningless.

Run:
    python parity_mlx.py
"""
from __future__ import annotations
import math
import os
import sys
import numpy as np
import torch

# Path: prepend script dir (so `mamba3_mlx` resolves to the local file)
# and repo root (so `cortex_counting` resolves to the file at root).
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

import mlx.core as mx
import mlx.nn as mlx_nn

# PyTorch reference
from mamba_platform.cortex_counting import (
    CortexLM as PTCortexLM,
    CortexLMConfig as PTCortexLMConfig,
    CounterPrimitive as PTCounterPrimitive,
)
# MLX port (sibling file in the same folder)
from mamba3_mlx import (
    CortexLM as MXCortexLM,
    CortexLMConfig as MXCortexLMConfig,
    CounterPrimitive as MXCounterPrimitive,
    Mamba3Config as MXMamba3Config,
)


def torch_to_mlx(t: torch.Tensor) -> mx.array:
    return mx.array(t.detach().cpu().float().numpy())


def transfer_weights(pt_model: PTCortexLM, mx_model: MXCortexLM) -> None:
    """Copy PyTorch parameters into the MLX module by walking both trees."""
    pt_sd = pt_model.state_dict()

    # Embeddings
    mx_model.embed.weight = torch_to_mlx(pt_sd["embed.weight"])
    mx_model.embed_norm.weight = torch_to_mlx(pt_sd["embed_norm.weight"])
    mx_model.embed_norm.bias   = torch_to_mlx(pt_sd["embed_norm.bias"])

    # Layers
    for i, layer_module in enumerate(mx_model.layers):
        layer_module.norm.weight = torch_to_mlx(pt_sd[f"layers.{i}.norm.weight"])
        layer_module.norm.bias   = torch_to_mlx(pt_sd[f"layers.{i}.norm.bias"])

        # Mamba3Block params
        blk = layer_module.block
        blk.in_proj.weight  = torch_to_mlx(pt_sd[f"layers.{i}.block.in_proj.weight"])
        blk.out_proj.weight = torch_to_mlx(pt_sd[f"layers.{i}.block.out_proj.weight"])
        blk.dt_bias = torch_to_mlx(pt_sd[f"layers.{i}.block.dt_bias"])
        blk.D       = torch_to_mlx(pt_sd[f"layers.{i}.block.D"])
        blk.B_norm.weight = torch_to_mlx(pt_sd[f"layers.{i}.block.B_norm.weight"])
        blk.B_norm.bias   = torch_to_mlx(pt_sd[f"layers.{i}.block.B_norm.bias"])
        blk.C_norm.weight = torch_to_mlx(pt_sd[f"layers.{i}.block.C_norm.weight"])
        blk.C_norm.bias   = torch_to_mlx(pt_sd[f"layers.{i}.block.C_norm.bias"])

    # Final norm + head (head is tied to embed in PT; in MLX we copy explicitly)
    mx_model.final_norm.weight = torch_to_mlx(pt_sd["final_norm.weight"])
    mx_model.final_norm.bias   = torch_to_mlx(pt_sd["final_norm.bias"])
    mx_model.head.weight = torch_to_mlx(pt_sd["embed.weight"])  # tied

    # Primitives — match by name and order
    for i, mx_prim in enumerate(mx_model.primitives):
        pt_prim = pt_model.primitives[i]
        if isinstance(mx_prim, MXCounterPrimitive):
            mx_prim.inc_proj.weight = torch_to_mlx(pt_sd[f"primitives.{i}.inc_proj.weight"])
            mx_prim.inc_proj.bias   = torch_to_mlx(pt_sd[f"primitives.{i}.inc_proj.bias"])
            mx_prim.reset_proj.weight = torch_to_mlx(pt_sd[f"primitives.{i}.reset_proj.weight"])
            mx_prim.reset_proj.bias   = torch_to_mlx(pt_sd[f"primitives.{i}.reset_proj.bias"])
            mx_prim.read_proj.weight = torch_to_mlx(pt_sd[f"primitives.{i}.read_proj.weight"])
            mx_prim.read_proj.bias   = torch_to_mlx(pt_sd[f"primitives.{i}.read_proj.bias"])


def parity_block_only():
    """Test 1: just the LM (no primitives), small config, fixed random input."""
    print("\n=== test 1: CortexLM forward, no primitives ===")
    torch.manual_seed(0)

    pt_cfg = PTCortexLMConfig(
        n_layers=2, d_model=64, d_state=16, expand=2, headdim=16,
        vocab_size=256, max_seq_len=32, use_counter=False,
    )
    mx_cfg = MXCortexLMConfig(
        n_layers=2, d_model=64, d_state=16, expand=2, headdim=16,
        vocab_size=256, max_seq_len=32,
    )
    pt_m = PTCortexLM(pt_cfg).eval()
    mx_m = MXCortexLM(mx_cfg)
    transfer_weights(pt_m, mx_m)

    rng = torch.randint(0, 256, (2, 24), generator=torch.Generator().manual_seed(7))
    pt_in = rng
    mx_in = mx.array(rng.numpy())

    with torch.no_grad():
        pt_out = pt_m(pt_in).numpy()
    mx_out_arr = mx_m(mx_in)
    mx.eval(mx_out_arr)
    mx_out = np.array(mx_out_arr)

    diff = np.abs(pt_out - mx_out)
    print(f"  output shape:  PT={pt_out.shape}  MX={mx_out.shape}")
    print(f"  max-abs-diff:  {diff.max():.2e}")
    print(f"  mean-abs-diff: {diff.mean():.2e}")
    print(f"  PT range:      [{pt_out.min():.3f}, {pt_out.max():.3f}]")
    return float(diff.max())


def parity_with_counter():
    """Test 2: CortexLM + CounterPrimitive."""
    print("\n=== test 2: CortexLM + CounterPrimitive ===")
    torch.manual_seed(1)

    pt_cfg = PTCortexLMConfig(
        n_layers=2, d_model=64, d_state=16, expand=2, headdim=16,
        vocab_size=256, max_seq_len=32,
        use_counter=True, n_counters=2, counter_layer=0,
        counter_readout="unbounded", counter_injection_scale=10.0,
    )
    mx_cfg = MXCortexLMConfig(
        n_layers=2, d_model=64, d_state=16, expand=2, headdim=16,
        vocab_size=256, max_seq_len=32,
    )
    pt_m = PTCortexLM(pt_cfg).eval()
    mx_ctr = MXCounterPrimitive(
        mx_cfg.d_model, layer=0, n_counters=2,
        readout="unbounded", injection_scale=10.0,
    )
    mx_m = MXCortexLM(mx_cfg, primitives=[mx_ctr])
    transfer_weights(pt_m, mx_m)

    # Build an input with realistic byte distribution: stars + colon + a's
    s = "*" * 5 + ":" + "a" * 5 + "\n"
    seq = [b for b in s.encode("utf-8")]
    seq = seq + [0] * (24 - len(seq))   # pad to 24
    rng = torch.tensor([seq, seq], dtype=torch.long)
    pt_in = rng
    mx_in = mx.array(rng.numpy())

    with torch.no_grad():
        pt_out = pt_m(pt_in).numpy()
    mx_out_arr = mx_m(mx_in)
    mx.eval(mx_out_arr)
    mx_out = np.array(mx_out_arr)

    diff = np.abs(pt_out - mx_out)
    print(f"  output shape:  PT={pt_out.shape}  MX={mx_out.shape}")
    print(f"  max-abs-diff:  {diff.max():.2e}")
    print(f"  mean-abs-diff: {diff.mean():.2e}")
    return float(diff.max())


def main():
    tol = 1e-3
    d1 = parity_block_only()
    d2 = parity_with_counter()
    print()
    print("=" * 60)
    print(f"max-abs-diff (no primitives):    {d1:.2e}  ({'PASS' if d1 < tol else 'FAIL'})")
    print(f"max-abs-diff (with counter):     {d2:.2e}  ({'PASS' if d2 < tol else 'FAIL'})")
    print(f"tolerance: {tol}")
    print("=" * 60)
    if d1 < tol and d2 < tol:
        print("ALL PASS — MLX port is numerically equivalent to PyTorch.")
        return 0
    else:
        print("PARITY FAIL — investigate before benchmarking.")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
