#!/usr/bin/env python3
"""Export PyTorch checkpoint to Rust binary format for the wgpu engine.

The format is simple:
  - 7 × uint32 header: d_model, d_state, headdim, n_layers, vocab_size, 0, 0
  - All weights concatenated as flat f32 arrays in order:
    embed, embed_norm_w, embed_norm_b,
    per-layer: in_proj_w, conv1d_w, conv1d_b, out_proj_w, dt_bias, A_log, D, norm_w, scale
    final_norm_w, final_norm_b

Usage:
    python export/rust_export.py --checkpoint checkpoints/specialists/parity.pt --output parity.bin
"""

import struct
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def export_to_bin(checkpoint_path: str, output_path: str) -> str:
    import torch
    from progressive_model import ProgressiveModel

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

    with open(output_path, "wb") as f:
        # Header
        for v in [d_model, d_state, headdim, n_layers, vocab_size, 0, 0]:
            f.write(struct.pack("<I", v))

        def write_tensor(key):
            if key in state_dict:
                data = state_dict[key].float().contiguous().numpy()
                f.write(data.tobytes())
                return data.size
            return 0

        # Embedding
        n = write_tensor("embed.weight")
        print(f"  embed: {n} floats")
        write_tensor("embed_norm.weight")
        write_tensor("embed_norm.bias")

        # Layers
        for i in range(n_layers):
            prefix = f"kernel_layers.{i}.block."
            n = write_tensor(f"{prefix}in_proj.weight")
            print(f"  layer {i} in_proj: {n} floats")
            write_tensor(f"{prefix}conv1d.weight")
            write_tensor(f"{prefix}conv1d.bias")
            write_tensor(f"{prefix}out_proj.weight")
            write_tensor(f"{prefix}dt_bias")
            write_tensor(f"{prefix}A_log")
            write_tensor(f"{prefix}D")

            # Layer norm (outside block)
            write_tensor(f"kernel_layers.{i}.norm.weight")

            # Scale parameter
            scale_key = f"kernel_layers.{i}.scale"
            if scale_key in state_dict:
                val = state_dict[scale_key].float().item()
                f.write(struct.pack("<f", val))
            else:
                f.write(struct.pack("<f", 0.01))

        # Final norm
        write_tensor("final_norm.weight")
        write_tensor("final_norm.bias")

    size_kb = Path(output_path).stat().st_size / 1024
    print(f"  Exported: {output_path} ({size_kb:.1f} KB)")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    export_to_bin(args.checkpoint, args.output)
