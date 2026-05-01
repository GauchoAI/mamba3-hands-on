from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import chess
import torch
import torch.nn as nn
import torch.nn.functional as F

import chess_jepa_bridge as jepa
import chess_motif_generalization as motif


HERE = Path(__file__).resolve().parent
ARTIFACT = HERE / "artifacts" / "chess_policy_arena_result.json"
N_MOVE_CLASSES = 64 * 64


class JepaPolicy(nn.Module):
    def __init__(self, encoder: jepa.Encoder, latent_dim: int, width: int):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(
            nn.Linear(latent_dim, width),
            nn.GELU(),
            nn.LayerNorm(width),
            nn.Linear(width, N_MOVE_CLASSES),
        )

    def forward(self, board_x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(board_x))


def train_motif_policy(args, dev: str) -> motif.MateMLP:
    train_cases, _ = motif.generate_cases(args.train_per_family, args.val_per_family, args.seed)
    x_train, y_train = motif.batch(train_cases, dev)
    model = motif.MateMLP(x_train.shape[-1], args.motif_width).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    for _ in range(args.motif_epochs):
        loss = F.cross_entropy(model(x_train), y_train)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    model.eval()
    return model


def train_jepa_encoder(args, dev: str) -> tuple[jepa.Encoder, dict]:
    cfg = jepa.Config(
        seed=args.seed,
        pairs=args.jepa_pairs,
        val_pairs=args.jepa_val_pairs,
        epochs=args.jepa_epochs,
        width=args.jepa_width,
        latent_dim=args.latent_dim,
    )
    pairs = jepa.generate_pairs(cfg.pairs, cfg.seed)
    train_pairs = pairs[cfg.val_pairs :]
    val_pairs = pairs[: cfg.val_pairs]
    x_train, move_train, y_train, _ = jepa.tensorize(train_pairs)
    x_val, move_val, y_val, val_fens = jepa.tensorize(val_pairs)
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
    metrics = jepa.evaluate(model, x_val, move_val, y_val, val_fens, dev)
    return model.encoder, metrics


def train_jepa_policy(args, dev: str) -> tuple[JepaPolicy, dict]:
    train_cases, _ = motif.generate_cases(args.train_per_family, args.val_per_family, args.seed)
    x_train = torch.stack([jepa.board_to_tensor(chess.Board(case.fen)) for case in train_cases]).to(dev)
    y_train = torch.tensor(
        [motif.move_class(chess.Move.from_uci(case.move_uci)) for case in train_cases],
        dtype=torch.long,
        device=dev,
    )
    encoder, bridge_metrics = train_jepa_encoder(args, dev)
    policy = JepaPolicy(encoder, args.latent_dim, args.policy_width).to(dev)
    if args.freeze_encoder:
        for p in policy.encoder.parameters():
            p.requires_grad_(False)
    opt = torch.optim.AdamW((p for p in policy.parameters() if p.requires_grad), lr=args.lr, weight_decay=1e-4)
    for _ in range(args.policy_epochs):
        loss = F.cross_entropy(policy(x_train), y_train)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    policy.eval()
    return policy, bridge_metrics


@torch.no_grad()
def legal_masked_prediction_from_logits(logits: torch.Tensor, board: chess.Board) -> chess.Move:
    scores = logits.detach().cpu()
    legal = list(board.legal_moves)
    legal.sort(key=lambda move: float(scores[motif.move_class(move)]), reverse=True)
    return legal[0]


@torch.no_grad()
def jepa_policy_move(policy: JepaPolicy, board: chess.Board, dev: str) -> chess.Move:
    x = jepa.board_to_tensor(board).unsqueeze(0).to(dev)
    logits = policy(x)[0]
    return legal_masked_prediction_from_logits(logits, board)


def is_mate_after(board: chess.Board, move: chess.Move) -> bool:
    after = board.copy(stack=False)
    after.push(move)
    return after.is_checkmate()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--train-per-family", type=int, default=384)
    parser.add_argument("--val-per-family", type=int, default=24)
    parser.add_argument("--motif-width", type=int, default=1024)
    parser.add_argument("--motif-epochs", type=int, default=120)
    parser.add_argument("--jepa-pairs", type=int, default=1800)
    parser.add_argument("--jepa-val-pairs", type=int, default=300)
    parser.add_argument("--jepa-width", type=int, default=256)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--jepa-epochs", type=int, default=14)
    parser.add_argument("--policy-width", type=int, default=512)
    parser.add_argument("--policy-epochs", type=int, default=140)
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--sample-rows", type=int, default=16)
    args = parser.parse_args()

    t0 = time.time()
    dev = motif.device()
    motif_policy = train_motif_policy(args, dev)
    jepa_policy, bridge_metrics = train_jepa_policy(args, dev)
    _, val_cases = motif.generate_cases(args.train_per_family, args.val_per_family, args.seed)

    rows = []
    motif_wins = 0
    jepa_wins = 0
    ties_both_mate = 0
    ties_both_fail = 0
    by_family: dict[str, dict[str, int]] = {}
    for case in val_cases:
        board = chess.Board(case.fen)
        motif_move = motif.legal_masked_prediction(motif_policy, board, dev)
        jepa_move = jepa_policy_move(jepa_policy, board, dev)
        motif_mate = is_mate_after(board, motif_move)
        jepa_mate = is_mate_after(board, jepa_move)
        fam = by_family.setdefault(case.motif, {"n": 0, "motif_mate": 0, "jepa_mate": 0, "motif_wins": 0, "jepa_wins": 0})
        fam["n"] += 1
        fam["motif_mate"] += int(motif_mate)
        fam["jepa_mate"] += int(jepa_mate)
        if motif_mate and not jepa_mate:
            motif_wins += 1
            fam["motif_wins"] += 1
            outcome = "motif_win"
        elif jepa_mate and not motif_mate:
            jepa_wins += 1
            fam["jepa_wins"] += 1
            outcome = "jepa_win"
        elif motif_mate and jepa_mate:
            ties_both_mate += 1
            outcome = "tie_both_mate"
        else:
            ties_both_fail += 1
            outcome = "tie_both_fail"
        if len(rows) < args.sample_rows:
            rows.append({
                "motif": case.motif,
                "fen": case.fen,
                "expected": case.move_uci,
                "all_mates": list(case.mate_moves),
                "motif_move": motif_move.uci(),
                "motif_mate": motif_mate,
                "jepa_policy_move": jepa_move.uci(),
                "jepa_policy_mate": jepa_mate,
                "outcome": outcome,
            })

    n = len(val_cases)
    payload = {
        "status": "competitive_tactical_policy_arena",
        "scope": "held-out mate-in-one positions, not full opening-to-endgame chess",
        "device": dev,
        "elapsed_s": round(time.time() - t0, 3),
        "positions": n,
        "config": {
            "freeze_encoder": args.freeze_encoder,
            "train_per_family": args.train_per_family,
            "val_per_family": args.val_per_family,
            "jepa_pairs": args.jepa_pairs,
            "jepa_epochs": args.jepa_epochs,
            "policy_epochs": args.policy_epochs,
        },
        "jepa_bridge_pretrain": {
            "cosine_mean": round(float(bridge_metrics["cosine_mean"]), 6),
            "nearest_neighbor_top1": round(float(bridge_metrics["nearest_neighbor_top1"]), 4),
            "nearest_neighbor_top5": round(float(bridge_metrics["nearest_neighbor_top5"]), 4),
        },
        "score": {
            "motif_wins": motif_wins,
            "jepa_wins": jepa_wins,
            "ties_both_mate": ties_both_mate,
            "ties_both_fail": ties_both_fail,
            "motif_mate_rate": round((motif_wins + ties_both_mate) / n, 4),
            "jepa_policy_mate_rate": round((jepa_wins + ties_both_mate) / n, 4),
        },
        "by_family": {
            name: {
                "n": stats["n"],
                "motif_mate_rate": round(stats["motif_mate"] / stats["n"], 4),
                "jepa_policy_mate_rate": round(stats["jepa_mate"] / stats["n"], 4),
                "motif_wins": stats["motif_wins"],
                "jepa_wins": stats["jepa_wins"],
            }
            for name, stats in by_family.items()
        },
        "sample_rows": rows,
    }
    ARTIFACT.parent.mkdir(exist_ok=True)
    ARTIFACT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
