"""mlx_to_torch_ckpt.py — convert an MLX `.npz` checkpoint to a PyTorch
`.pt` file in the format `train_counter_attach.py` expects.

The MLX trainer (`train_bilingual_mlx.py`) saves model parameters as
`np.savez(...)` with keys prefixed `model/<param-path>`. The PyTorch
trainer (`train_bilingual_cortex_lm.py`) saves `torch.save({"model":
state_dict, "cfg": CortexLMConfig, "step": int})` — same module-path
naming convention, different container.

This converter:
  1. Loads the .npz
  2. Strips the `model/` prefix from each key
  3. Casts bf16 → fp32 (PyTorch CortexLM is fp32 by default)
  4. Builds a CortexLMConfig from CLI args
  5. Saves as a .pt file ready for `train_counter_attach.py --lm-ckpt …`

Run:
    python cortex_bilingual/mlx_to_torch_ckpt.py \\
        --in checkpoints/lm_mlx_widerN/step_FINAL.npz \\
        --out checkpoints/lm_mlx_widerN/step_FINAL.pt \\
        --n-layers 4 --d-model 128

CLI args mirror train_bilingual_mlx.py defaults so most of the time
no overrides are needed.
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cortex_counting import CortexLMConfig


def convert(npz_path: Path, pt_path: Path, cfg: CortexLMConfig) -> None:
    """MLX .npz → PyTorch .pt in counter-attach format."""
    print(f"loading {npz_path} ...", flush=True)
    data = np.load(npz_path, allow_pickle=False)
    print(f"  {len(data.files)} arrays in npz", flush=True)

    # Extract model state — keys prefixed `model/`, strip + cast.
    state_dict: dict[str, torch.Tensor] = {}
    skipped = []
    for key in data.files:
        if not key.startswith("model/"):
            # Optimizer state (`opt/...`) and bookkeeping (`step`) ignored.
            skipped.append(key)
            continue
        torch_key = key[len("model/"):]
        arr = data[key]
        # MLX bf16 arrays come back as uint16 (since numpy doesn't have
        # native bf16). Detect and cast properly.
        if arr.dtype == np.uint16:
            # bf16 stored as raw bits in uint16. Reinterpret + upcast.
            arr_bf16 = arr.view(np.uint16)
            # bf16 → fp32: shift left by 16 bits, treat as float32.
            arr_fp32 = (arr_bf16.astype(np.uint32) << 16).view(np.float32)
            tensor = torch.from_numpy(arr_fp32.copy())
        else:
            tensor = torch.from_numpy(arr.astype(np.float32, copy=False))
        state_dict[torch_key] = tensor

    print(f"  {len(state_dict)} model tensors extracted "
          f"({len(skipped)} skipped: opt/step bookkeeping)",
          flush=True)

    step = int(data["step"]) if "step" in data.files else -1

    payload = {"model": state_dict, "cfg": cfg, "step": step}
    pt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, pt_path)
    print(f"wrote {pt_path}  step={step}  size={pt_path.stat().st_size:,} bytes",
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True,
                    help="input MLX .npz checkpoint")
    ap.add_argument("--out", dest="out_path", default=None,
                    help="output .pt path (default: same dir, .pt suffix)")
    # CortexLMConfig fields — defaults match train_bilingual_mlx.py
    ap.add_argument("--n-layers", type=int, default=4)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--d-state", type=int, default=16)
    ap.add_argument("--expand", type=int, default=2)
    ap.add_argument("--headdim", type=int, default=16)
    ap.add_argument("--vocab-size", type=int, default=256)
    ap.add_argument("--max-seq-len", type=int, default=128)
    args = ap.parse_args()

    in_path = Path(args.in_path)
    if not in_path.exists():
        print(f"ERROR: {in_path} does not exist", file=sys.stderr)
        sys.exit(1)
    out_path = Path(args.out_path) if args.out_path \
        else in_path.with_suffix(".pt")

    cfg = CortexLMConfig(
        n_layers=args.n_layers,
        d_model=args.d_model,
        d_state=args.d_state,
        expand=args.expand,
        headdim=args.headdim,
        vocab_size=args.vocab_size,
        max_seq_len=args.max_seq_len,
        use_counter=False,
    )
    convert(in_path, out_path, cfg)


if __name__ == "__main__":
    main()
