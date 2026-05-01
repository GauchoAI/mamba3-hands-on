from __future__ import annotations

import argparse
import json
import random
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import chess
import chess.engine
import torch
import torch.nn as nn
import torch.nn.functional as F


HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE / "artifacts"
N_MOVE_CLASSES = 64 * 64
PIECE_TO_PLANE = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
    chess.KING: 5,
}


@dataclass(frozen=True)
class LabeledPosition:
    fen: str
    move_uci: str


class MoveMLP(nn.Module):
    def __init__(self, d_in: int, width: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, width),
            nn.GELU(),
            nn.LayerNorm(width),
            nn.Linear(width, width),
            nn.GELU(),
            nn.LayerNorm(width),
            nn.Linear(width, N_MOVE_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def board_features(board: chess.Board) -> torch.Tensor:
    x = torch.zeros(12 * 64 + 5, dtype=torch.float32)
    for sq, piece in board.piece_map().items():
        color_offset = 0 if piece.color == chess.WHITE else 6
        plane = color_offset + PIECE_TO_PLANE[piece.piece_type]
        x[plane * 64 + sq] = 1.0
    base = 12 * 64
    x[base] = 1.0 if board.turn == chess.WHITE else 0.0
    x[base + 1] = float(board.has_kingside_castling_rights(chess.WHITE))
    x[base + 2] = float(board.has_queenside_castling_rights(chess.WHITE))
    x[base + 3] = float(board.has_kingside_castling_rights(chess.BLACK))
    x[base + 4] = float(board.has_queenside_castling_rights(chess.BLACK))
    return x


def move_class(move: chess.Move) -> int:
    return move.from_square * 64 + move.to_square


def random_positions(n: int, seed: int, min_ply: int, max_ply: int) -> list[chess.Board]:
    rng = random.Random(seed)
    boards: list[chess.Board] = []
    seen: set[str] = set()
    attempts = 0
    while len(boards) < n and attempts < n * 50:
        attempts += 1
        board = chess.Board()
        for _ in range(rng.randint(min_ply, max_ply)):
            if board.is_game_over():
                break
            move = rng.choice(list(board.legal_moves))
            board.push(move)
        if not board.is_game_over() and board.fen() not in seen:
            seen.add(board.fen())
            boards.append(board)
    if len(boards) < n:
        raise RuntimeError(f"only generated {len(boards)} positions")
    return boards


def label_with_engine(engine_path: str, boards: list[chess.Board], depth: int, time_limit: float) -> list[LabeledPosition]:
    labels: list[LabeledPosition] = []
    with chess.engine.SimpleEngine.popen_uci(engine_path) as engine:
        limit = chess.engine.Limit(depth=depth, time=time_limit)
        for board in boards:
            result = engine.play(board, limit)
            if result.move is not None:
                labels.append(LabeledPosition(board.fen(), result.move.uci()))
    return labels


def batch(rows: list[LabeledPosition], dev: str) -> tuple[torch.Tensor, torch.Tensor]:
    xs = []
    ys = []
    for row in rows:
        board = chess.Board(row.fen)
        xs.append(board_features(board))
        ys.append(move_class(chess.Move.from_uci(row.move_uci)))
    return torch.stack(xs).to(dev), torch.tensor(ys, dtype=torch.long, device=dev)


def legal_masked_prediction(model: MoveMLP, board: chess.Board, dev: str) -> chess.Move:
    x = board_features(board).unsqueeze(0).to(dev)
    with torch.no_grad():
        logits = model(x)[0].detach().cpu()
    legal = list(board.legal_moves)
    legal.sort(key=lambda move: float(logits[move_class(move)]), reverse=True)
    return legal[0]


def train_student(rows: list[LabeledPosition], args, dev: str) -> dict:
    split = int(len(rows) * 0.8)
    train_rows = rows[:split]
    val_rows = rows[split:]
    x_train, y_train = batch(train_rows, dev)
    x_val, y_val = batch(val_rows, dev)
    model = MoveMLP(x_train.shape[-1], args.width).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    t0 = time.time()
    for _ in range(args.epochs):
        logits = model(x_train)
        loss = F.cross_entropy(logits, y_train)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    raw_ok = 0
    legal_ok = 0
    eval_rows = []
    with torch.no_grad():
        raw_acc = (model(x_val).argmax(-1) == y_val).float().mean().item()
    for row in val_rows:
        board = chess.Board(row.fen)
        pred = legal_masked_prediction(model, board, dev)
        raw_pred_idx = int(model(board_features(board).unsqueeze(0).to(dev)).argmax(-1).item())
        raw_pred = chess.Move(raw_pred_idx // 64, raw_pred_idx % 64)
        raw_ok += int(raw_pred.uci() == row.move_uci)
        legal_ok += int(pred.uci() == row.move_uci)
        eval_rows.append({
            "fen": row.fen,
            "teacher": row.move_uci,
            "raw_argmax": raw_pred.uci(),
            "legal_masked": pred.uci(),
            "legal_masked_matches_teacher": pred.uci() == row.move_uci,
        })
    return {
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "params": sum(p.numel() for p in model.parameters()),
        "train_elapsed_s": round(time.time() - t0, 3),
        "raw_val_acc": round(raw_acc, 4),
        "raw_argmax_pass": raw_ok,
        "legal_masked_pass": legal_ok,
        "eval_rows": eval_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", default="stockfish")
    parser.add_argument("--positions", type=int, default=256)
    parser.add_argument("--seed", type=int, default=12)
    parser.add_argument("--min-ply", type=int, default=6)
    parser.add_argument("--max-ply", type=int, default=28)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--time-limit", type=float, default=0.02)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--lr", type=float, default=2e-3)
    args = parser.parse_args()

    t0 = time.time()
    engine_path = shutil.which(args.engine) or (args.engine if Path(args.engine).exists() else None)
    payload: dict = {
        "teacher": args.engine,
        "engine_path": engine_path,
        "positions_requested": args.positions,
    }
    if engine_path is None:
        payload.update({
            "status": "blocked",
            "reason": "UCI engine not found. Install Stockfish or pass --engine /path/to/stockfish.",
        })
    else:
        dev = device()
        boards = random_positions(args.positions, args.seed, args.min_ply, args.max_ply)
        labels = label_with_engine(engine_path, boards, args.depth, args.time_limit)
        payload.update({
            "status": "ok",
            "device": dev,
            "labels": len(labels),
            "student": train_student(labels, args, dev),
        })
    payload["elapsed_s"] = round(time.time() - t0, 3)
    ARTIFACTS.mkdir(exist_ok=True)
    (ARTIFACTS / "chess_teacher_distill_result.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
