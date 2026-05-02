from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
CHECKPOINT_DIR = HERE / "checkpoints" / "chess_full_game_trace_arena"

sys.path.insert(0, str(HERE))
import chess_jepa_bridge as jepa  # noqa: E402
import chess_motif_generalization as motif  # noqa: E402
from chess_policy_arena import JepaPolicy  # noqa: E402


def load_motif(checkpoint_dir: Path) -> motif.MateMLP:
    ckpt = torch.load(checkpoint_dir / "motif_full_trace.pt", map_location="cpu", weights_only=False)
    model = motif.MateMLP(12 * 64 + 5, int(ckpt["config"]["policy_width"]))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def load_jepa(checkpoint_dir: Path) -> JepaPolicy:
    ckpt = torch.load(checkpoint_dir / "jepa_full_trace.pt", map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    encoder = jepa.Encoder(int(cfg["jepa_width"]), int(cfg["latent_dim"]))
    model = JepaPolicy(encoder, int(cfg["latent_dim"]), int(cfg["policy_width"]))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def export_model(model: torch.nn.Module, sample: torch.Tensor, path: Path, input_name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        sample,
        path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        external_data=False,
        input_names=[input_name],
        output_names=["logits"],
        dynamic_axes={
            input_name: {0: "batch"},
            "logits": {0: "batch"},
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path, default=CHECKPOINT_DIR)
    parser.add_argument("--out-dir", type=Path, default=CHECKPOINT_DIR)
    args = parser.parse_args()

    motif_model = load_motif(args.checkpoint_dir)
    jepa_model = load_jepa(args.checkpoint_dir)

    export_model(
        motif_model,
        torch.zeros(1, 12 * 64 + 5, dtype=torch.float32),
        args.out_dir / "motif_full_trace.onnx",
        "board_features",
    )
    export_model(
        jepa_model,
        torch.zeros(1, jepa.BOARD_DIM, dtype=torch.float32),
        args.out_dir / "jepa_full_trace.onnx",
        "board_features",
    )

    try:
        import onnx

        for name in ("motif_full_trace.onnx", "jepa_full_trace.onnx"):
            model = onnx.load(args.out_dir / name)
            onnx.checker.check_model(model)
    except ImportError:
        pass

    manifest = {
        "format": "chess_expert_onnx_manifest/v1",
        "runtime": "onnxruntime-web",
        "opset": 18,
        "input": "board_features",
        "output": "logits",
        "move_class": "from_square * 64 + to_square",
        "models": [
            {
                "name": "motif_full_trace",
                "file": "motif_full_trace.onnx",
                "input_dim": 12 * 64 + 5,
                "feature_schema": "motif_board_features/v1",
                "bytes": (args.out_dir / "motif_full_trace.onnx").stat().st_size,
            },
            {
                "name": "jepa_full_trace",
                "file": "jepa_full_trace.onnx",
                "input_dim": jepa.BOARD_DIM,
                "feature_schema": "jepa_board_features/v1",
                "bytes": (args.out_dir / "jepa_full_trace.onnx").stat().st_size,
            },
        ],
    }
    (args.out_dir / "onnx_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
