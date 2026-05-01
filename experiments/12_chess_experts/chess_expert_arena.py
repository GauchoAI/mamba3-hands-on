from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import chess
import torch
import torch.nn.functional as F

import chess_jepa_bridge as jepa
import chess_motif_generalization as motif


HERE = Path(__file__).resolve().parent
ARTIFACT = HERE / "artifacts" / "chess_expert_arena_result.json"


def train_motif_model(args, dev: str) -> motif.MateMLP:
    train_cases, _ = motif.generate_cases(args.motif_train_per_family, args.motif_val_per_family, args.seed)
    x_train, y_train = motif.batch(train_cases, dev)
    model = motif.MateMLP(x_train.shape[-1], args.motif_width).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    for _ in range(args.motif_epochs):
        logits = model(x_train)
        loss = F.cross_entropy(logits, y_train)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    model.eval()
    return model


def train_jepa_model(args, dev: str) -> jepa.ChessJEPA:
    cfg = jepa.Config(
        seed=args.seed,
        pairs=args.jepa_pairs,
        val_pairs=args.jepa_val_pairs,
        epochs=args.jepa_epochs,
        width=args.jepa_width,
        latent_dim=args.jepa_latent_dim,
    )
    pairs = jepa.generate_pairs(cfg.pairs, cfg.seed)
    train_pairs = pairs[cfg.val_pairs :]
    x_train, move_train, y_train, _ = jepa.tensorize(train_pairs)
    model = jepa.ChessJEPA(cfg).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    n = x_train.shape[0]
    for _ in range(cfg.epochs):
        order = torch.randperm(n)
        for start in range(0, n, cfg.batch_size):
            idx = order[start : start + cfg.batch_size]
            x_b = x_train[idx].to(dev)
            move_b = move_train[idx].to(dev)
            y_b = y_train[idx].to(dev)
            z_t = model.encoder(x_b)
            pred = model.predictor(z_t, move_b)
            target = model.encoder(y_b)
            loss, _ = jepa.batch_loss(pred, target)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    model.eval()
    return model


@torch.no_grad()
def jepa_transition_metrics(model: jepa.ChessJEPA, board: chess.Board, move: chess.Move, dev: str) -> dict:
    before = jepa.board_to_tensor(board).unsqueeze(0).to(dev)
    move_x = jepa.move_to_tensor(board, move).unsqueeze(0).to(dev)
    after = board.copy(stack=False)
    after.push(move)
    after_x = jepa.board_to_tensor(after).unsqueeze(0).to(dev)
    z_t = model.encoder(before)
    pred = model.predictor(z_t, move_x)
    target = model.encoder(after_x)
    return {
        "cosine": float(F.cosine_similarity(pred, target, dim=-1).item()),
        "normalized_mse": float(F.mse_loss(F.normalize(pred, dim=-1), F.normalize(target, dim=-1)).item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--motif-train-per-family", type=int, default=256)
    parser.add_argument("--motif-val-per-family", type=int, default=16)
    parser.add_argument("--motif-width", type=int, default=1024)
    parser.add_argument("--motif-epochs", type=int, default=100)
    parser.add_argument("--jepa-pairs", type=int, default=1400)
    parser.add_argument("--jepa-val-pairs", type=int, default=240)
    parser.add_argument("--jepa-width", type=int, default=256)
    parser.add_argument("--jepa-latent-dim", type=int, default=64)
    parser.add_argument("--jepa-epochs", type=int, default=12)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--sample-rows", type=int, default=12)
    args = parser.parse_args()

    t0 = time.time()
    dev = motif.device()
    motif_model = train_motif_model(args, dev)
    jepa_model = train_jepa_model(args, dev)
    _, val_cases = motif.generate_cases(args.motif_train_per_family, args.motif_val_per_family, args.seed)

    rows = []
    motif_legal_mates = 0
    jepa_cos = []
    jepa_mse = []
    by_family: dict[str, dict[str, int]] = {}
    for case in val_cases:
        board = chess.Board(case.fen)
        pred = motif.legal_masked_prediction(motif_model, board, dev)
        after = board.copy(stack=False)
        after.push(pred)
        legal_mate = after.is_checkmate()
        motif_legal_mates += int(legal_mate)
        metrics = jepa_transition_metrics(jepa_model, board, pred, dev)
        jepa_cos.append(metrics["cosine"])
        jepa_mse.append(metrics["normalized_mse"])
        fam = by_family.setdefault(case.motif, {"n": 0, "legal_mate": 0})
        fam["n"] += 1
        fam["legal_mate"] += int(legal_mate)
        if len(rows) < args.sample_rows:
            rows.append({
                "motif": case.motif,
                "fen": case.fen,
                "expected": case.move_uci,
                "motif_prediction": pred.uci(),
                "motif_prediction_legal_mate": legal_mate,
                "jepa_transition_cosine_on_motif_move": round(metrics["cosine"], 6),
                "jepa_transition_mse_on_motif_move": round(metrics["normalized_mse"], 8),
            })

    payload = {
        "status": "diagnostic_arena_not_full_game",
        "reason": "motif model is a move policy for mate-in-one; JEPA is a transition model, not a value/policy model",
        "device": dev,
        "elapsed_s": round(time.time() - t0, 3),
        "positions": len(val_cases),
        "motif_policy": {
            "legal_mate_pass": motif_legal_mates,
            "legal_mate_rate": round(motif_legal_mates / len(val_cases), 4),
            "by_family": {
                name: {
                    "n": stats["n"],
                    "legal_mate_pass": stats["legal_mate"],
                    "legal_mate_rate": round(stats["legal_mate"] / max(stats["n"], 1), 4),
                }
                for name, stats in by_family.items()
            },
        },
        "jepa_world_model_on_motif_moves": {
            "mean_cosine": round(sum(jepa_cos) / len(jepa_cos), 6),
            "mean_normalized_mse": round(sum(jepa_mse) / len(jepa_mse), 8),
        },
        "sample_rows": rows,
    }
    ARTIFACT.parent.mkdir(exist_ok=True)
    ARTIFACT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
