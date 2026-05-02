from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
FULL_TRACE_DIR = HERE / "checkpoints" / "chess_full_game_trace_arena"
ONLINE_DIR = HERE / "checkpoints" / "chess_online_world_model"
PLAYABLE_DIR = HERE / "checkpoints" / "chess_playable_assets"
HF_REPO_ID = "miguelemosreverte/mamba3-chess-experts"
HF_PATH = "chess_playable_assets"

sys.path.insert(0, str(HERE))
from chess_online_world_model import FEATURE_DIM, OnlineValueModel  # noqa: E402


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def copy_asset(src: Path, dst: Path) -> dict:
    if not src.exists():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return {
        "file": str(dst.relative_to(PLAYABLE_DIR)),
        "bytes": dst.stat().st_size,
        "sha256": sha256(dst),
    }


def bounded_kpi(namespace: str, name: str, value: float, source: str) -> dict:
    return {
        "namespace": namespace,
        "name": name,
        "value": round(max(0.0, min(1.0, float(value))), 6),
        "range": [0.0, 1.0],
        "higher_is_better": True,
        "source": source,
    }


def full_trace_kpi(policy: str) -> dict:
    result_path = FULL_TRACE_DIR / "chess_full_game_trace_arena_result.json"
    if not result_path.exists():
        return bounded_kpi(
            f"12_chess_experts/chess_full_game_trace_arena/{policy}",
            "full_game_score_rate",
            0.0,
            "missing_result_artifact",
        )
    result = json.loads(result_path.read_text())
    summary = result.get("summary") or {}
    games = max(int(summary.get("games") or 0), 1)
    wins = int(summary.get(f"{policy}_wins") or 0)
    draws = int(summary.get("draws") or 0)
    score = (wins + 0.5 * draws) / games
    return bounded_kpi(
        f"12_chess_experts/chess_full_game_trace_arena/{policy}",
        "full_game_score_rate",
        score,
        f"summary.{policy}_wins_plus_half_draws_over_games",
    )


def tactical_risk(summary: dict) -> float | None:
    audit = (summary.get("tactical_audit") or {}).get("adaptive_world_model") or {}
    if not audit:
        return None
    return round(
        2.0 * float(audit.get("tactical_blunder_rate") or 0.0)
        + float(audit.get("queen_hang_rate") or 0.0)
        + 0.75 * float(audit.get("major_piece_hang_rate") or 0.0)
        + 0.1 * float(audit.get("avg_opponent_best_capture") or 0.0),
        6,
    )


def export_online_checkpoint(checkpoint_path: Path, out_path: Path) -> tuple[dict, dict]:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    width = int(ckpt["model_width"])
    model = OnlineValueModel(width)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        torch.zeros(1, FEATURE_DIM, dtype=torch.float32),
        out_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        external_data=False,
        input_names=["candidate_features"],
        output_names=["value"],
        dynamic_axes={
            "candidate_features": {0: "batch"},
            "value": {0: "batch"},
        },
    )
    try:
        import onnx

        exported = onnx.load(out_path)
        onnx.checker.check_model(exported)
    except ImportError:
        pass
    return ckpt, {
        "file": str(out_path.relative_to(PLAYABLE_DIR)),
        "bytes": out_path.stat().st_size,
        "sha256": sha256(out_path),
    }


def online_policy_from_checkpoint(checkpoint_path: Path, publish_rank: int) -> dict:
    ckpt, online_asset = export_online_checkpoint(
        checkpoint_path,
        PLAYABLE_DIR / "online_strategies" / f"{checkpoint_path.stem}.onnx",
    )
    ckpt_asset = copy_asset(
        checkpoint_path,
        PLAYABLE_DIR / "online_strategies" / checkpoint_path.name,
    )
    strategy_id = ckpt.get("strategy_id") or checkpoint_path.stem.removesuffix("_value")
    strategy_label = ckpt.get("strategy_label") or strategy_id.replace("_", " ").title()
    champion_summary = ckpt.get("heldout_vs_safety", {}).get("summary", {})
    alpha_summary = ckpt.get("heldout_vs_alpha", {}).get("summary", {})
    online_kpi = ckpt.get("kpi") or bounded_kpi(
        f"12_chess_experts/chess_online_world_model/{strategy_id}",
        "heldout_vs_static_safety_alpha_score",
        champion_summary.get("adaptive_score_rate") or 0.0,
        "heldout_vs_safety.summary.adaptive_score_rate",
    )
    return {
        "id": strategy_id,
        "label": strategy_label,
        "kind": "safety_candidate_reranker_value_head",
        "default": False,
        "playable": True,
        "kpi": online_kpi,
        "model": online_asset,
        "checkpoint": ckpt_asset,
        "input": "candidate_features",
        "output": "value",
        "feature_schema": "online_world_model_candidate_features/v1",
        "candidate_source": "safety_aware_teacher_score_top_k",
        "input_dim": FEATURE_DIM,
        "top_k": int(ckpt["config"]["top_k"]),
        "value_weight": float(ckpt["config"]["value_weight"]),
        "adaptive_safety_weight": float(ckpt["config"]["adaptive_safety_weight"]),
        "champion_iteration": ckpt.get("champion_iteration"),
        "champion_promotions": ckpt.get("champion_promotions"),
        "metrics": {
            "heldout_vs_safety_score": champion_summary.get("adaptive_score_rate"),
            "heldout_vs_safety_wld": [
                champion_summary.get("adaptive_wins"),
                champion_summary.get("opponent_wins"),
                champion_summary.get("draws"),
            ],
            "heldout_vs_alpha_score": alpha_summary.get("adaptive_score_rate"),
            "tactical_risk": tactical_risk(champion_summary),
            "tactical_audit": champion_summary.get("tactical_audit", {}).get("adaptive_world_model", {}),
        },
        "publish_rank": publish_rank,
    }


def build_manifest(args) -> dict:
    if PLAYABLE_DIR.exists():
        shutil.rmtree(PLAYABLE_DIR)
    PLAYABLE_DIR.mkdir(parents=True, exist_ok=True)
    legacy_dir = PLAYABLE_DIR / "legacy_full_trace"

    motif_asset = copy_asset(FULL_TRACE_DIR / "motif_full_trace.onnx", legacy_dir / "motif_full_trace.onnx")
    jepa_asset = copy_asset(FULL_TRACE_DIR / "jepa_full_trace.onnx", legacy_dir / "jepa_full_trace.onnx")
    full_trace_manifest = FULL_TRACE_DIR / "onnx_manifest.json"
    if full_trace_manifest.exists():
        copy_asset(full_trace_manifest, legacy_dir / "onnx_manifest.json")

    online_policies = [
        online_policy_from_checkpoint(path, idx + 1)
        for idx, path in enumerate(sorted(ONLINE_DIR.glob("*_value.pt")))
    ]
    motif_kpi = full_trace_kpi("motif_full_trace")
    jepa_kpi = full_trace_kpi("jepa_full_trace")
    legacy_rank = len(online_policies) + 1
    policies = online_policies + [
        {
            "id": "motif_full_trace",
            "label": "Motif full trace",
            "kind": "legacy_onnx_policy",
            "default": False,
            "playable": True,
            "kpi": motif_kpi,
            "model": motif_asset,
            "input": "board_features",
            "output": "logits",
            "feature_schema": "motif_board_features/v1",
            "input_dim": 773,
            "move_class": "from_square * 64 + to_square",
            "publish_rank": legacy_rank,
        },
        {
            "id": "jepa_full_trace",
            "label": "JEPA full trace",
            "kind": "legacy_onnx_policy",
            "default": False,
            "playable": True,
            "kpi": jepa_kpi,
            "model": jepa_asset,
            "input": "board_features",
            "output": "logits",
            "feature_schema": "jepa_board_features/v1",
            "input_dim": 783,
            "move_class": "from_square * 64 + to_square",
            "publish_rank": legacy_rank + 1,
        },
    ]
    policies.sort(
        key=lambda row: (
            -row["kpi"]["value"],
            row.get("metrics", {}).get("tactical_risk", float("inf")),
            row["publish_rank"],
        )
    )
    if policies:
        policies[0]["default"] = True
    default_policy = policies[0]["id"] if policies else ""
    manifest = {
        "format": "mamba3_chess_playable_manifest/v1",
        "created_at": time.time(),
        "hf_repo": HF_REPO_ID,
        "hf_path": HF_PATH,
        "kpi_policy": {
            "required": True,
            "description": "Every pushed playable checkpoint carries a bounded 0-1 KPI used for default selection and future pruning.",
            "default_selection": "highest kpi.value, then lowest tactical_risk, then lowest publish_rank",
        },
        "default_policy": default_policy,
        "selection_reason": "highest KPI value with tactical-risk tie break",
        "policies": policies,
    }
    manifest_path = PLAYABLE_DIR / "playable_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def upload_to_hf(args) -> dict:
    if args.no_upload:
        return {"uploaded": False, "reason": "no_upload"}
    if not os.environ.get("HF_TOKEN"):
        return {"uploaded": False, "reason": "HF_TOKEN_missing"}
    from huggingface_hub import HfApi

    api = HfApi(token=os.environ["HF_TOKEN"])
    api.upload_folder(
        repo_id=args.repo_id,
        repo_type="model",
        folder_path=str(PLAYABLE_DIR),
        path_in_repo=args.path_in_repo,
        commit_message="Update playable chess expert assets",
    )
    return {
        "uploaded": True,
        "repo": args.repo_id,
        "path_in_repo": args.path_in_repo,
        "manifest_url": f"https://huggingface.co/{args.repo_id}/resolve/main/{args.path_in_repo}/playable_manifest.json",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default=HF_REPO_ID)
    parser.add_argument("--path-in-repo", default=HF_PATH)
    parser.add_argument("--no-upload", action="store_true")
    args = parser.parse_args()

    manifest = build_manifest(args)
    upload = upload_to_hf(args)
    payload = {"manifest": manifest, "upload": upload}
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
