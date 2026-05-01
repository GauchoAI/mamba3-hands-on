from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import chess
import torch
import torch.nn as nn
import torch.nn.functional as F


HERE = Path(__file__).resolve().parent
ARTIFACT = HERE / "artifacts" / "chess_motif_generalization_result.json"
N_MOVE_CLASSES = 64 * 64
PIECE_TO_PLANE = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
    chess.KING: 5,
}
MOTIFS = (
    "back_rank_rook_queen",
    "queen_side_file",
    "rook_side_file",
    "knight_corner",
)


@dataclass(frozen=True)
class ChessCase:
    motif: str
    fen: str
    move_uci: str
    mate_moves: tuple[str, ...]


class MateMLP(nn.Module):
    def __init__(self, d_in: int, width: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, width),
            nn.GELU(),
            nn.LayerNorm(width),
            nn.Dropout(0.05),
            nn.Linear(width, width),
            nn.GELU(),
            nn.LayerNorm(width),
            nn.Dropout(0.05),
            nn.Linear(width, width // 2),
            nn.GELU(),
            nn.LayerNorm(width // 2),
            nn.Linear(width // 2, N_MOVE_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def sq(file: int, rank: int) -> int:
    return chess.square(file, rank)


def empty_board(turn: chess.Color = chess.WHITE) -> chess.Board:
    board = chess.Board(None)
    board.turn = turn
    board.castling_rights = chess.BB_EMPTY
    board.ep_square = None
    board.halfmove_clock = 0
    board.fullmove_number = 1
    board.clear_stack()
    return board


def move_class(move: chess.Move) -> int:
    return move.from_square * 64 + move.to_square


def class_move(idx: int) -> chess.Move:
    return chess.Move(idx // 64, idx % 64)


def mate_moves(board: chess.Board) -> list[chess.Move]:
    mates = []
    for move in board.legal_moves:
        nxt = board.copy(stack=False)
        nxt.push(move)
        if nxt.is_checkmate():
            mates.append(move)
    return mates


def board_features(board: chess.Board) -> torch.Tensor:
    x = torch.zeros(12 * 64 + 5, dtype=torch.float32)
    for square, piece in board.piece_map().items():
        color_offset = 0 if piece.color == chess.WHITE else 6
        plane = color_offset + PIECE_TO_PLANE[piece.piece_type]
        x[plane * 64 + square] = 1.0
    base = 12 * 64
    x[base] = 1.0 if board.turn == chess.WHITE else 0.0
    x[base + 1] = float(board.has_kingside_castling_rights(chess.WHITE))
    x[base + 2] = float(board.has_queenside_castling_rights(chess.WHITE))
    x[base + 3] = float(board.has_kingside_castling_rights(chess.BLACK))
    x[base + 4] = float(board.has_queenside_castling_rights(chess.BLACK))
    return x


def validate_case(motif: str, board: chess.Board, intended: chess.Move) -> ChessCase | None:
    if not board.is_valid() or board.is_check():
        return None
    mates = mate_moves(board)
    if intended not in mates:
        return None
    mate_uci = tuple(sorted(move.uci() for move in mates))
    return ChessCase(motif=motif, fen=board.fen(), move_uci=intended.uci(), mate_moves=mate_uci)


def add_distractors(board: chess.Board, rng: random.Random, protected: set[int], max_pieces: int = 4) -> None:
    candidates = [
        (chess.WHITE, chess.PAWN),
        (chess.WHITE, chess.KNIGHT),
        (chess.WHITE, chess.BISHOP),
        (chess.BLACK, chess.PAWN),
        (chess.BLACK, chess.KNIGHT),
        (chess.BLACK, chess.BISHOP),
    ]
    rng.shuffle(candidates)
    occupied = protected | set(board.piece_map())
    for color, piece_type in candidates[: rng.randint(0, max_pieces)]:
        free = [
            square
            for square in chess.SQUARES
            if square not in occupied and 1 <= chess.square_rank(square) <= 6
        ]
        if not free:
            return
        square = rng.choice(free)
        board.set_piece_at(square, chess.Piece(piece_type, color))
        if board.is_valid() and not board.is_check():
            occupied.add(square)
            continue
        board.remove_piece_at(square)


def add_quiet_knight_distractors(board: chess.Board, rng: random.Random, protected: set[int]) -> None:
    pieces = [
        (chess.WHITE, chess.BISHOP),
        (chess.WHITE, chess.PAWN),
        (chess.BLACK, chess.BISHOP),
        (chess.BLACK, chess.PAWN),
    ]
    rng.shuffle(pieces)
    occupied = protected | set(board.piece_map())
    for color, piece_type in pieces[: rng.randint(0, 2)]:
        free = [
            square
            for square in chess.SQUARES
            if square not in occupied and 1 <= chess.square_rank(square) <= 6
        ]
        if not free:
            return
        square = rng.choice(free)
        board.set_piece_at(square, chess.Piece(piece_type, color))
        if board.is_valid() and not board.is_check():
            occupied.add(square)
            continue
        board.remove_piece_at(square)


def build_back_rank_rook_queen(rng: random.Random) -> ChessCase | None:
    top = rng.choice([True, False])
    king_rank = 7 if top else 0
    pawn_rank = 6 if top else 1
    origin_ranks = list(range(0, 6)) if top else list(range(2, 8))
    king_file = rng.randint(1, 6)
    direction = rng.choice([-1, 1])
    target_file = king_file + 2 * direction
    if not 0 <= target_file <= 7:
        return None

    board = empty_board()
    black_king = sq(king_file, king_rank)
    target = sq(target_file, king_rank)
    origin = sq(target_file, rng.choice(origin_ranks))
    white_king = chess.H1 if top else chess.H8
    if chess.square_file(white_king) == target_file:
        white_king = chess.A1 if top else chess.A8

    protected = {black_king, target, origin, white_king}
    board.set_piece_at(black_king, chess.Piece(chess.KING, chess.BLACK))
    board.set_piece_at(white_king, chess.Piece(chess.KING, chess.WHITE))
    board.set_piece_at(origin, chess.Piece(rng.choice([chess.ROOK, chess.QUEEN]), chess.WHITE))
    for file in range(king_file - 1, king_file + 2):
        if 0 <= file <= 7:
            pawn_square = sq(file, pawn_rank)
            board.set_piece_at(pawn_square, chess.Piece(chess.PAWN, chess.BLACK))
            protected.add(pawn_square)
    add_distractors(board, rng, protected)
    return validate_case("back_rank_rook_queen", board, chess.Move(origin, target))


def build_side_file(rng: random.Random, piece_type: chess.PieceType, motif: str) -> ChessCase | None:
    side_file = rng.choice([0, 7])
    inward = 1 if side_file == 0 else -1
    king_rank = rng.randint(2, 5)
    target_rank = 0 if king_rank >= 4 else 7
    target = sq(side_file, target_rank)
    origin_options = []
    for file in range(8):
        if file != side_file:
            origin_options.append(sq(file, target_rank))
    if piece_type == chess.QUEEN:
        # Extra queen-line variants: enter the mating file from a diagonal.
        for delta in (-3, -2, 2, 3):
            rank = target_rank + delta
            file = side_file + abs(delta) * inward
            if 0 <= file <= 7 and 0 <= rank <= 7:
                origin_options.append(sq(file, rank))
    origin = rng.choice(origin_options)

    board = empty_board()
    black_king = sq(side_file, king_rank)
    white_king = chess.H1 if side_file == 0 else chess.A1
    if abs(chess.square_rank(white_king) - king_rank) <= 1:
        white_king = chess.H8 if side_file == 0 else chess.A8

    protected = {black_king, white_king, target, origin}
    board.set_piece_at(black_king, chess.Piece(chess.KING, chess.BLACK))
    board.set_piece_at(white_king, chess.Piece(chess.KING, chess.WHITE))
    board.set_piece_at(origin, chess.Piece(piece_type, chess.WHITE))
    for rank in (king_rank - 1, king_rank, king_rank + 1):
        blocker = sq(side_file + inward, rank)
        board.set_piece_at(blocker, chess.Piece(chess.PAWN, chess.BLACK))
        protected.add(blocker)
    add_distractors(board, rng, protected)
    return validate_case(motif, board, chess.Move(origin, target))


def build_rook_side_file(rng: random.Random) -> ChessCase | None:
    return build_side_file(rng, chess.ROOK, "rook_side_file")


def build_queen_side_file(rng: random.Random) -> ChessCase | None:
    return build_side_file(rng, chess.QUEEN, "queen_side_file")


def build_knight_corner(rng: random.Random) -> ChessCase | None:
    corner_file = rng.choice([0, 7])
    corner_rank = rng.choice([0, 7])
    df = 1 if corner_file == 0 else -1
    dr = 1 if corner_rank == 0 else -1
    black_king = sq(corner_file, corner_rank)
    target = rng.choice([
        sq(corner_file + 2 * df, corner_rank + dr),
        sq(corner_file + df, corner_rank + 2 * dr),
    ])

    board = empty_board()
    white_king = sq(7 - corner_file, 7 - corner_rank)
    protected = {black_king, target, white_king}
    board.set_piece_at(black_king, chess.Piece(chess.KING, chess.BLACK))
    board.set_piece_at(white_king, chess.Piece(chess.KING, chess.WHITE))
    for file, rank in (
        (corner_file + df, corner_rank),
        (corner_file, corner_rank + dr),
        (corner_file + df, corner_rank + dr),
    ):
        blocker = sq(file, rank)
        if blocker != target:
            piece_type = chess.KNIGHT if rank in {0, 7} else chess.PAWN
            board.set_piece_at(blocker, chess.Piece(piece_type, chess.BLACK))
            protected.add(blocker)

    origins = []
    target_file = chess.square_file(target)
    target_rank = chess.square_rank(target)
    for file_delta, rank_delta in ((1, 2), (2, 1), (-1, 2), (-2, 1), (1, -2), (2, -1), (-1, -2), (-2, -1)):
        file = target_file + file_delta
        rank = target_rank + rank_delta
        origin = sq(file, rank) if 0 <= file <= 7 and 0 <= rank <= 7 else None
        if origin is not None and origin not in protected and origin != black_king:
            origins.append(origin)
    if not origins:
        return None
    origin = rng.choice(origins)
    board.set_piece_at(origin, chess.Piece(chess.KNIGHT, chess.WHITE))
    add_quiet_knight_distractors(board, rng, protected | {origin})
    return validate_case("knight_corner", board, chess.Move(origin, target))


BUILDERS = {
    "back_rank_rook_queen": build_back_rank_rook_queen,
    "queen_side_file": build_queen_side_file,
    "rook_side_file": build_rook_side_file,
    "knight_corner": build_knight_corner,
}


def generate_family(motif: str, n: int, rng: random.Random, seen: set[str]) -> list[ChessCase]:
    cases: list[ChessCase] = []
    attempts = 0
    max_attempts = n * 1200
    while len(cases) < n and attempts < max_attempts:
        attempts += 1
        case = BUILDERS[motif](rng)
        if case and case.fen not in seen:
            seen.add(case.fen)
            cases.append(case)
    if len(cases) < n:
        raise RuntimeError(f"generated {len(cases)}/{n} cases for {motif} after {attempts} attempts")
    return cases


def generate_cases(per_family_train: int, per_family_val: int, seed: int) -> tuple[list[ChessCase], list[ChessCase]]:
    rng = random.Random(seed)
    train: list[ChessCase] = []
    val: list[ChessCase] = []
    seen: set[str] = set()
    for motif in MOTIFS:
        cases = generate_family(motif, per_family_train + per_family_val, rng, seen)
        rng.shuffle(cases)
        train.extend(cases[:per_family_train])
        val.extend(cases[per_family_train:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def batch(cases: list[ChessCase], dev: str) -> tuple[torch.Tensor, torch.Tensor]:
    xs = []
    ys = []
    for case in cases:
        board = chess.Board(case.fen)
        xs.append(board_features(board))
        ys.append(move_class(chess.Move.from_uci(case.move_uci)))
    return torch.stack(xs).to(dev), torch.tensor(ys, dtype=torch.long, device=dev)


@torch.no_grad()
def legal_masked_prediction(model: MateMLP, board: chess.Board, dev: str) -> chess.Move:
    logits = model(board_features(board).unsqueeze(0).to(dev))[0].detach().cpu()
    legal = list(board.legal_moves)
    legal.sort(key=lambda move: float(logits[move_class(move)]), reverse=True)
    return legal[0]


def evaluate(model: MateMLP, cases: list[ChessCase], dev: str, max_rows: int) -> dict:
    by_family: dict[str, dict[str, int]] = defaultdict(lambda: {"n": 0, "exact": 0, "legal_mate": 0})
    rows = []
    for case in cases:
        board = chess.Board(case.fen)
        pred = legal_masked_prediction(model, board, dev)
        after = board.copy(stack=False)
        after.push(pred)
        exact = pred.uci() == case.move_uci
        legal_mate = after.is_checkmate()
        fam = by_family[case.motif]
        fam["n"] += 1
        fam["exact"] += int(exact)
        fam["legal_mate"] += int(legal_mate)
        if len(rows) < max_rows:
            rows.append({
                "motif": case.motif,
                "fen": case.fen,
                "expected": case.move_uci,
                "all_mates": list(case.mate_moves),
                "prediction": pred.uci(),
                "exact": exact,
                "legal_mate": legal_mate,
            })

    family_rates = {}
    for motif in MOTIFS:
        stats = by_family[motif]
        n = max(stats["n"], 1)
        family_rates[motif] = {
            "n": stats["n"],
            "exact_pass": stats["exact"],
            "exact_rate": round(stats["exact"] / n, 4),
            "legal_mate_pass": stats["legal_mate"],
            "legal_mate_rate": round(stats["legal_mate"] / n, 4),
        }
    total_n = sum(stats["n"] for stats in by_family.values())
    total_exact = sum(stats["exact"] for stats in by_family.values())
    total_mate = sum(stats["legal_mate"] for stats in by_family.values())
    return {
        "n": total_n,
        "exact_pass": total_exact,
        "exact_rate": round(total_exact / max(total_n, 1), 4),
        "legal_mate_pass": total_mate,
        "legal_mate_rate": round(total_mate / max(total_n, 1), 4),
        "by_family": family_rates,
        "sample_rows": rows,
    }


def train(args: argparse.Namespace, dev: str) -> dict:
    torch.manual_seed(args.seed)
    train_cases, val_cases = generate_cases(args.train_per_family, args.val_per_family, args.seed)
    x_train, y_train = batch(train_cases, dev)
    x_val, y_val = batch(val_cases, dev)
    model = MateMLP(x_train.shape[-1], args.width).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    t0 = time.time()
    history = []
    best_val = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        logits = model(x_train)
        loss = F.cross_entropy(logits, y_train)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if epoch == 1 or epoch % args.report_every == 0 or epoch == args.epochs:
            model.eval()
            with torch.no_grad():
                train_acc = (model(x_train).argmax(-1) == y_train).float().mean().item()
                val_acc = (model(x_val).argmax(-1) == y_val).float().mean().item()
            best_val = max(best_val, val_acc)
            history.append({
                "epoch": epoch,
                "loss": round(float(loss.detach().cpu()), 4),
                "train_raw_acc": round(train_acc, 4),
                "val_raw_acc": round(val_acc, 4),
            })

    model.eval()
    evaluation = evaluate(model, val_cases, dev, args.sample_rows)
    return {
        "seed": args.seed,
        "device": dev,
        "elapsed_s": round(time.time() - t0, 3),
        "train_cases": len(train_cases),
        "val_cases": len(val_cases),
        "motifs": list(MOTIFS),
        "model": {
            "type": "torch MLP over 12 piece planes plus side/castling bits; no Phi or LLM",
            "width": args.width,
            "epochs": args.epochs,
            "params": sum(p.numel() for p in model.parameters()),
        },
        "best_val_raw_acc": round(best_val, 4),
        "history": history,
        "evaluation": evaluation,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--train-per-family", type=int, default=512)
    parser.add_argument("--val-per-family", type=int, default=32)
    parser.add_argument("--width", type=int, default=1536)
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--report-every", type=int, default=20)
    parser.add_argument("--sample-rows", type=int, default=32)
    args = parser.parse_args()

    t0 = time.time()
    dev = device()
    payload = train(args, dev)
    payload["total_elapsed_s"] = round(time.time() - t0, 3)
    ARTIFACT.parent.mkdir(exist_ok=True)
    ARTIFACT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    if payload["evaluation"]["legal_mate_pass"] != payload["evaluation"]["n"]:
        raise SystemExit("held-out legal mate evaluation was not perfect")


if __name__ == "__main__":
    main()
