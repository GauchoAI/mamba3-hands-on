"""
Bridge between PyTorch `.pt` checkpoints (specialist_trainer.py format) and
the binary `from_bin` format the PTX engine reads / writes.

The PyTorch format (state_dict under canonical ProgressiveModel keys):
    embed.weight                                  [V, d]
    embed_norm.weight, embed_norm.bias            [d], [d]
    final_norm.weight, final_norm.bias            [d], [d]
    head.weight (== embed.weight via tying)       [V, d]
    kernel_layers.{i}.block.in_proj.weight        [dip, d]
    kernel_layers.{i}.block.out_proj.weight       [d, di]
    kernel_layers.{i}.block.dt_bias               [H]
    kernel_layers.{i}.block.D                     [H]
    kernel_layers.{i}.block.B_norm.weight/.bias   [dS], [dS]
    kernel_layers.{i}.block.C_norm.weight/.bias   [dS], [dS]
    kernel_layers.{i}.norm.weight, .bias          [d], [d]
    kernel_layers.{i}.scale.0                     []   (scalar)

The from_bin layout (used by Mamba3Model::from_bin / forward-parity):
  header:  d_model, d_state, headdim, n_layers, vocab, 0, 0   (7 u32, 28 bytes)
  embed_w (V*d), embed_norm_w (d), embed_norm_b (d)
  per layer: in_proj_w, out_proj_w, dt_bias, d_param,
             b_norm_w, b_norm_b, c_norm_w, c_norm_b,
             layer_norm_w, layer_norm_b, scale (1 f32)
  final_norm_w (d), final_norm_b (d)
  — all f32 little-endian

Round trip:
  pt_to_bin(pt_path, bin_path)
  bin_to_pt(bin_path, pt_path, task=..., config=..., accuracy=..., cycles=...)
"""
import os, struct, sys
from pathlib import Path

import numpy as np


def _state_dict_layer_keys(prefix, i):
    """All state_dict keys for layer i, in from_bin order."""
    return [
        f"{prefix}{i}.block.in_proj.weight",
        f"{prefix}{i}.block.out_proj.weight",
        f"{prefix}{i}.block.dt_bias",
        f"{prefix}{i}.block.D",
        f"{prefix}{i}.block.B_norm.weight",
        f"{prefix}{i}.block.B_norm.bias",
        f"{prefix}{i}.block.C_norm.weight",
        f"{prefix}{i}.block.C_norm.bias",
        f"{prefix}{i}.norm.weight",
        f"{prefix}{i}.norm.bias",
    ]


def _detect_layer_prefix(state_dict):
    """ProgressiveModel uses kernel_layers.{i}. and cortex_layers.{i}. — for
    now we assume only kernel_layers (per the existing checkpoints we
    inspected).  If a checkpoint has cortex layers we'd need to concatenate.
    """
    if any(k.startswith("kernel_layers.") for k in state_dict):
        return "kernel_layers."
    if any(k.startswith("layers.") for k in state_dict):
        return "layers."
    raise RuntimeError(
        "ckpt_bridge: couldn't find layer keys (kernel_layers.* or layers.*) in state_dict; "
        f"top-level keys: {sorted(set(k.split('.')[0] for k in state_dict))}"
    )


def pt_to_bin(pt_path: str, bin_path: str) -> dict:
    """Convert a PyTorch specialist_trainer checkpoint to from_bin format.

    Returns the metadata dict (config, accuracy, cycles, task, n_params)
    so the caller can pass these along separately.  The state_dict itself
    is NOT returned — only its on-disk binary form.
    """
    import torch
    ck = torch.load(pt_path, map_location="cpu", weights_only=False)
    sd = ck["model"]
    config = ck.get("config", {})

    # Detect dimensions from the embedding shape.
    embed_w = sd["embed.weight"]   # (V, d)
    V, d = embed_w.shape
    d_state  = config.get("d_state", 16)
    headdim  = config.get("headdim", 16)
    n_layers = config.get("n_kernel_layers", config.get("n_layers"))
    if n_layers is None:
        # Count layer keys.
        prefix = _detect_layer_prefix(sd)
        n_layers = 1 + max(int(k.split(".")[1]) for k in sd if k.startswith(prefix))
    prefix = _detect_layer_prefix(sd)

    # Sanity checks
    di = 2 * d
    nh = di // headdim
    num_rope_angles = d_state // 2
    dip = 2 * di + 2 * d_state + 3 * nh + num_rope_angles
    in_proj_w_shape = sd[f"{prefix}0.block.in_proj.weight"].shape  # (dip, d)
    if in_proj_w_shape != (dip, d):
        raise RuntimeError(
            f"ckpt_bridge: in_proj.weight shape {tuple(in_proj_w_shape)} != expected ({dip}, {d}). "
            f"Check d_state ({d_state}) / headdim ({headdim}) match the trained config."
        )

    def t2np(t):
        return t.detach().cpu().numpy().astype(np.float32, copy=False)

    with open(bin_path, "wb") as f:
        f.write(struct.pack("<7I", d, d_state, headdim, n_layers, V, 0, 0))
        # embed
        f.write(t2np(sd["embed.weight"]).tobytes())
        f.write(t2np(sd["embed_norm.weight"]).tobytes())
        f.write(t2np(sd["embed_norm.bias"]).tobytes())
        # per layer
        for i in range(n_layers):
            for key in _state_dict_layer_keys(prefix, i):
                f.write(t2np(sd[key]).tobytes())
            # scale: ParameterList stored as kernel_layers.{i}.scale.0
            scale_key = f"{prefix}{i}.scale.0"
            scale_val = float(sd[scale_key]) if scale_key in sd else 0.01
            f.write(struct.pack("<f", scale_val))
        # final_norm
        f.write(t2np(sd["final_norm.weight"]).tobytes())
        f.write(t2np(sd["final_norm.bias"]).tobytes())

    return {
        "task": ck.get("task"),
        "config": config,
        "accuracy": ck.get("accuracy"),
        "cycles": ck.get("cycles"),
        "n_params": ck.get("n_params"),
        "n_layers_loaded": n_layers,
        "d_model": d, "vocab": V, "d_state": d_state, "headdim": headdim,
        "n_heads": nh, "d_in_proj": dip,
    }


def bin_to_pt(bin_path: str, pt_path: str,
              task: str, config: dict,
              accuracy: float, cycles: int,
              optimizer_state=None) -> None:
    """Inverse of pt_to_bin: read a from_bin file and write a PyTorch
    `.pt` checkpoint that specialist_trainer.py can resume from.

    This lets the GA continue evolving against PTX-trained specialists
    using the existing checkpoint pipeline."""
    import torch
    with open(bin_path, "rb") as f:
        header = struct.unpack("<7I", f.read(28))
        d, d_state, headdim, n_layers, V = header[:5]
        rest = f.read()

    floats = np.frombuffer(rest, dtype=np.float32)
    off = 0
    def take(*shape):
        nonlocal off
        n = int(np.prod(shape))
        v = floats[off:off + n].reshape(shape).copy()
        off += n
        return torch.from_numpy(v)

    di = 2 * d
    nh = di // headdim
    num_rope_angles = d_state // 2
    dip = 2 * di + 2 * d_state + 3 * nh + num_rope_angles

    sd = {}
    sd["embed.weight"]      = take(V, d)
    sd["embed_norm.weight"] = take(d)
    sd["embed_norm.bias"]   = take(d)
    for i in range(n_layers):
        sd[f"kernel_layers.{i}.block.in_proj.weight"]   = take(dip, d)
        sd[f"kernel_layers.{i}.block.out_proj.weight"]  = take(d, di)
        sd[f"kernel_layers.{i}.block.dt_bias"]          = take(nh)
        sd[f"kernel_layers.{i}.block.D"]                = take(nh)
        sd[f"kernel_layers.{i}.block.B_norm.weight"]    = take(d_state)
        sd[f"kernel_layers.{i}.block.B_norm.bias"]      = take(d_state)
        sd[f"kernel_layers.{i}.block.C_norm.weight"]    = take(d_state)
        sd[f"kernel_layers.{i}.block.C_norm.bias"]      = take(d_state)
        sd[f"kernel_layers.{i}.norm.weight"]            = take(d)
        sd[f"kernel_layers.{i}.norm.bias"]              = take(d)
        scale_v = float(floats[off]); off += 1
        sd[f"kernel_layers.{i}.scale.0"]                = torch.tensor(scale_v)
    sd["final_norm.weight"] = take(d)
    sd["final_norm.bias"]   = take(d)
    # Weight tying: head.weight aliases embed.weight in the live module.
    # In a state_dict we store both (PyTorch's load_state_dict happily
    # accepts the duplicate as long as values are consistent).
    sd["head.weight"] = sd["embed.weight"].clone()

    out = {
        "model": sd,
        "task": task,
        "config": config,
        "accuracy": accuracy,
        "cycles": cycles,
        "n_params": sum(t.numel() for t in sd.values()),
    }
    if optimizer_state is not None:
        out["optimizer"] = optimizer_state
    Path(pt_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, pt_path)


if __name__ == "__main__":
    # Smoke test: round-trip an existing checkpoint and verify state_dict
    # values are preserved exactly.
    if len(sys.argv) < 2:
        print("usage: python3 ckpt_bridge.py <pt-path>")
        sys.exit(1)
    src = sys.argv[1]
    bin_tmp = src + ".bin.tmp"
    pt_tmp  = src + ".roundtrip.pt"
    meta = pt_to_bin(src, bin_tmp)
    print(f"pt_to_bin: {bin_tmp}  ({os.path.getsize(bin_tmp):,} bytes)  meta={meta}")
    bin_to_pt(bin_tmp, pt_tmp, meta["task"], meta["config"],
              meta.get("accuracy", 0.0), meta.get("cycles", 0))
    # Verify match
    import torch
    src_sd = torch.load(src, map_location="cpu", weights_only=False)["model"]
    rt_sd  = torch.load(pt_tmp, map_location="cpu", weights_only=False)["model"]
    bad = []
    for k, v in src_sd.items():
        if k not in rt_sd:
            bad.append(f"missing in roundtrip: {k}")
            continue
        if not torch.allclose(v.float(), rt_sd[k].float(), atol=1e-6):
            bad.append(f"{k}: max_diff={float((v - rt_sd[k]).abs().max()):.2e}")
    if bad:
        print("FAIL — round-trip mismatches:")
        for b in bad[:10]: print("  ", b)
        sys.exit(1)
    print(f"PASS — round-trip preserved all {len(src_sd)} state_dict tensors")
    os.unlink(bin_tmp); os.unlink(pt_tmp)
