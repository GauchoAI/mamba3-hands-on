#!/usr/bin/env python3
"""Export PyTorch checkpoint to Rust binary format.

Format:
  Header: 7 × uint32 (d_model, d_state, headdim, n_layers, vocab_size, 0, 0)
  Weights: flat f32 arrays in order:
    embed_w, embed_norm_w, embed_norm_b,
    per-layer: in_proj_w, out_proj_w, dt_bias, D, B_norm_w, B_norm_b, C_norm_w, C_norm_b, layer_norm_w, scale
    final_norm_w, final_norm_b

Usage:
    python export/rust_export.py --checkpoint checkpoints/specialists/parity.pt --output parity.bin
"""

import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def export_to_bin(checkpoint_path: str, output_path: str) -> str:
    import torch

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = ckpt.get("config", {})
    state_dict = ckpt["model"]
    task = ckpt.get("task", "unknown")

    d_model = config.get("d_model", 64)
    d_state = config.get("d_state", 16)
    headdim = config.get("headdim", 16)
    n_layers = config.get("n_kernel_layers", 1)
    vocab_size = 260

    print(f"Exporting {task}: d={d_model} L={n_layers} dS={d_state} hd={headdim}")
    total_floats = 0

    with open(output_path, "wb") as f:
        # Header
        for v in [d_model, d_state, headdim, n_layers, vocab_size, 0, 0]:
            f.write(struct.pack("<I", v))

        def write(key, required=True):
            nonlocal total_floats
            if key in state_dict:
                data = state_dict[key].float().contiguous().numpy()
                f.write(data.tobytes())
                total_floats += data.size
                return True
            elif required:
                print(f"  WARNING: missing {key}")
            return False

        # Embedding
        write("embed.weight")
        write("embed_norm.weight")
        write("embed_norm.bias")

        # Layers
        for i in range(n_layers):
            bp = f"kernel_layers.{i}.block."
            write(f"{bp}in_proj.weight")
            write(f"{bp}out_proj.weight")
            write(f"{bp}dt_bias")
            write(f"{bp}D")
            write(f"{bp}B_norm.weight")
            write(f"{bp}B_norm.bias")
            write(f"{bp}C_norm.weight")
            write(f"{bp}C_norm.bias")

            # Layer norm (outside block in progressive_model)
            if not write(f"kernel_layers.{i}.norm.weight", required=False):
                f.write(struct.pack(f"<{d_model}f", *([1.0] * d_model)))
                total_floats += d_model
            if not write(f"kernel_layers.{i}.norm.bias", required=False):
                f.write(struct.pack(f"<{d_model}f", *([0.0] * d_model)))
                total_floats += d_model

            # Scale (stored as ParameterList: scale.0)
            scale_key = f"kernel_layers.{i}.scale.0"
            if scale_key not in state_dict:
                scale_key = f"kernel_layers.{i}.scale"
            if scale_key in state_dict:
                f.write(struct.pack("<f", state_dict[scale_key].float().item()))
            else:
                f.write(struct.pack("<f", 0.01))
            total_floats += 1

        # Final norm
        write("final_norm.weight")
        write("final_norm.bias")

    size_kb = Path(output_path).stat().st_size / 1024
    print(f"  {total_floats:,} floats → {output_path} ({size_kb:.1f} KB)")
    return output_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    export_to_bin(args.checkpoint, args.output)
