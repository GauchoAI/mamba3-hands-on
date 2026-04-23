"""GGUF Export — convert Mamba-3 specialist checkpoints to GGUF format.

GGUF (GPT-Generated Unified Format) is the standard format for llama.cpp
inference. Since Mamba-3 is not a transformer, we define a custom tensor
layout that maps SSM blocks to GGUF tensors.

Tensor naming convention:
    token_embd.weight                    — embedding table
    blk.{i}.ssm_in_proj.weight          — Mamba3Block in_proj
    blk.{i}.ssm_conv1d.weight           — conv1d kernel
    blk.{i}.ssm_conv1d.bias             — conv1d bias
    blk.{i}.ssm_x_proj.weight           — x_proj (dt, B, C projection)
    blk.{i}.ssm_dt_proj.weight          — dt linear
    blk.{i}.ssm_dt_proj.bias            — dt bias
    blk.{i}.ssm_A_log                   — A parameter (log space)
    blk.{i}.ssm_D                       — D parameter
    blk.{i}.ssm_out_proj.weight         — output projection
    blk.{i}.ssm_norm.weight             — layer norm
    output_norm.weight                   — final layer norm
    output.weight                        — language model head

Usage:
    from export.gguf_export import export_to_gguf
    export_to_gguf("checkpoints/specialists/parity.pt", "checkpoints/gguf/parity.gguf")
"""

import struct
import numpy as np
from pathlib import Path


# GGUF constants
GGUF_MAGIC = 0x46475547  # "GGUF" in little-endian
GGUF_VERSION = 3

# GGUF metadata value types
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32 = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9

# GGUF tensor types
GGUF_TENSOR_F32 = 0
GGUF_TENSOR_F16 = 1


def export_to_gguf(checkpoint_path: str, output_path: str,
                   model_type: str = "mamba3") -> str:
    """Convert a PyTorch specialist checkpoint to GGUF format.

    Args:
        checkpoint_path: Path to .pt checkpoint file
        output_path: Path for output .gguf file
        model_type: Model architecture identifier

    Returns:
        Path to the created .gguf file
    """
    import torch

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["model"]
    config = ckpt.get("config", {})
    task = ckpt.get("task", "unknown")
    accuracy = ckpt.get("accuracy", 0)

    # Extract model dimensions from config or state dict
    d_model = config.get("d_model", 64)
    n_layers = config.get("n_kernel_layers", 1)
    d_state = config.get("d_state", 16)
    headdim = config.get("headdim", 16)

    # Map PyTorch state dict keys to GGUF tensor names
    tensor_map = _build_tensor_map(state_dict, n_layers)

    # Prepare metadata
    metadata = {
        "general.architecture": ("string", model_type),
        "general.name": ("string", f"mamba3-specialist-{task}"),
        "general.file_type": ("uint32", 0),  # F32
        f"{model_type}.context_length": ("uint32", 512),
        f"{model_type}.embedding_length": ("uint32", d_model),
        f"{model_type}.block_count": ("uint32", n_layers),
        f"{model_type}.ssm.d_state": ("uint32", d_state),
        f"{model_type}.ssm.headdim": ("uint32", headdim),
        "mamba.task": ("string", task),
        "mamba.accuracy": ("float32", accuracy),
        "mamba.device": ("string", config.get("device", "unknown")),
        "mamba.scan_backend": ("string", config.get("scan_backend", "unknown")),
        "tokenizer.ggml.model": ("string", "byte"),
    }

    # Convert tensors to numpy
    tensors = []
    for gguf_name, pt_key in tensor_map.items():
        if pt_key in state_dict:
            tensor = state_dict[pt_key].float().numpy()
            tensors.append((gguf_name, tensor))

    # Write GGUF file
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with open(output, "wb") as f:
        _write_gguf(f, metadata, tensors)

    n_params = sum(t.size for _, t in tensors)
    size_mb = output.stat().st_size / (1024 * 1024)
    print(f"  GGUF exported: {output} ({n_params:,} params, {size_mb:.1f} MB)", flush=True)

    return str(output)


def _build_tensor_map(state_dict: dict, n_layers: int) -> dict:
    """Map PyTorch state dict keys to GGUF tensor names."""
    tensor_map = {}

    # Embedding
    for key in ("embed.weight", "embedding.weight"):
        if key in state_dict:
            tensor_map["token_embd.weight"] = key
            break

    # Per-layer SSM tensors
    for i in range(n_layers):
        # Try different naming patterns from ProgressiveModel
        prefixes = [
            f"kernel_layers.{i}.block.",
            f"layers.{i}.block.",
            f"kernel.{i}.",
        ]

        for prefix in prefixes:
            mappings = {
                f"blk.{i}.ssm_in_proj.weight": f"{prefix}in_proj.weight",
                f"blk.{i}.ssm_conv1d.weight": f"{prefix}conv1d.weight",
                f"blk.{i}.ssm_conv1d.bias": f"{prefix}conv1d.bias",
                f"blk.{i}.ssm_out_proj.weight": f"{prefix}out_proj.weight",
                f"blk.{i}.ssm_A_log": f"{prefix}A_log",
                f"blk.{i}.ssm_D": f"{prefix}D",
                f"blk.{i}.ssm_dt_bias": f"{prefix}dt_bias",
                f"blk.{i}.ssm_norm.weight": f"{prefix}norm.weight",
            }

            for gguf_name, pt_key in mappings.items():
                if pt_key in state_dict:
                    tensor_map[gguf_name] = pt_key

        # Layer norm (outside block)
        for norm_key in (f"kernel_layers.{i}.norm.weight", f"layers.{i}.norm.weight"):
            if norm_key in state_dict:
                tensor_map[f"blk.{i}.attn_norm.weight"] = norm_key
                break

    # Final norm
    for key in ("final_norm.weight", "norm.weight"):
        if key in state_dict:
            tensor_map["output_norm.weight"] = key
            break

    # LM head
    for key in ("head.weight", "lm_head.weight", "output.weight"):
        if key in state_dict:
            tensor_map["output.weight"] = key
            break

    return tensor_map


def _write_gguf(f, metadata: dict, tensors: list):
    """Write a complete GGUF file."""
    n_tensors = len(tensors)
    n_metadata = len(metadata)

    # Header
    f.write(struct.pack("<I", GGUF_MAGIC))
    f.write(struct.pack("<I", GGUF_VERSION))
    f.write(struct.pack("<Q", n_tensors))
    f.write(struct.pack("<Q", n_metadata))

    # Metadata KV pairs
    for key, (vtype, value) in metadata.items():
        _write_string(f, key)
        if vtype == "string":
            f.write(struct.pack("<I", GGUF_TYPE_STRING))
            _write_string(f, str(value))
        elif vtype == "uint32":
            f.write(struct.pack("<I", GGUF_TYPE_UINT32))
            f.write(struct.pack("<I", int(value)))
        elif vtype == "int32":
            f.write(struct.pack("<I", GGUF_TYPE_INT32))
            f.write(struct.pack("<i", int(value)))
        elif vtype == "float32":
            f.write(struct.pack("<I", GGUF_TYPE_FLOAT32))
            f.write(struct.pack("<f", float(value)))

    # Tensor info (name, n_dims, dims, type, offset)
    # Calculate offsets
    data_offset = 0
    tensor_infos = []
    for name, arr in tensors:
        # Align to 32 bytes
        if data_offset % 32 != 0:
            data_offset += 32 - (data_offset % 32)
        tensor_infos.append((name, arr, data_offset))
        data_offset += arr.nbytes

    for name, arr, offset in tensor_infos:
        _write_string(f, name)
        n_dims = len(arr.shape)
        f.write(struct.pack("<I", n_dims))
        for dim in arr.shape:
            f.write(struct.pack("<Q", dim))
        f.write(struct.pack("<I", GGUF_TENSOR_F32))  # always f32 for now
        f.write(struct.pack("<Q", offset))

    # Alignment padding before tensor data
    current = f.tell()
    if current % 32 != 0:
        f.write(b"\x00" * (32 - (current % 32)))

    # Tensor data
    data_start = f.tell()
    for name, arr, offset in tensor_infos:
        # Pad to alignment
        current = f.tell() - data_start
        if current < offset:
            f.write(b"\x00" * (offset - current))
        f.write(arr.tobytes())


def _write_string(f, s: str):
    """Write a GGUF string (length-prefixed)."""
    b = s.encode("utf-8")
    f.write(struct.pack("<Q", len(b)))
    f.write(b)
