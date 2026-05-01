from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

import chess
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


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
class ChessCase:
    fen: str
    move_uci: str


class ChessMateMLP(nn.Module):
    def __init__(self, d_in: int, width: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, width),
            nn.GELU(),
            nn.LayerNorm(width),
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


def move_class(move: chess.Move) -> int:
    return move.from_square * 64 + move.to_square


def class_move(idx: int) -> chess.Move:
    return chess.Move(idx // 64, idx % 64)


def mate_moves(board: chess.Board) -> list[chess.Move]:
    out = []
    for move in board.legal_moves:
        b = board.copy(stack=False)
        b.push(move)
        if b.is_checkmate():
            out.append(move)
    return out


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


def empty_board(turn: chess.Color) -> chess.Board:
    board = chess.Board(None)
    board.turn = turn
    board.clear_stack()
    board.castling_rights = chess.BB_EMPTY
    board.ep_square = None
    board.halfmove_clock = 0
    board.fullmove_number = 1
    return board


def add_random_distractors(board: chess.Board, rng: random.Random, protected: set[int]) -> None:
    pieces = [
        (chess.WHITE, chess.KNIGHT),
        (chess.WHITE, chess.BISHOP),
        (chess.WHITE, chess.PAWN),
        (chess.BLACK, chess.KNIGHT),
        (chess.BLACK, chess.BISHOP),
        (chess.BLACK, chess.PAWN),
    ]
    rng.shuffle(pieces)
    occupied = protected | set(board.piece_map())
    for color, piece_type in pieces[: rng.randint(0, 3)]:
        candidates = [sq for sq in chess.SQUARES if sq not in occupied and chess.square_rank(sq) not in {0, 7}]
        if not candidates:
            return
        sq = rng.choice(candidates)
        board.set_piece_at(sq, chess.Piece(piece_type, color))
        occupied.add(sq)


def build_back_rank_case(rng: random.Random) -> ChessCase | None:
    top = rng.choice([True, False])
    king_rank = 7 if top else 0
    pawn_rank = 6 if top else 1
    origin_ranks = range(0, 6) if top else range(2, 8)
    king_file = rng.randint(1, 6)
    direction = rng.choice([-1, 1])
    target_file = king_file + 2 * direction
    if not 0 <= target_file <= 7:
        return None

    board = empty_board(chess.WHITE)
    black_king = chess.square(king_file, king_rank)
    target = chess.square(target_file, king_rank)
    origin = chess.square(target_file, rng.choice(list(origin_ranks)))
    white_king = chess.H1 if top else chess.H8
    if chess.square_file(white_king) == target_file:
        white_king = chess.A1 if top else chess.A8

    protected = {black_king, target, origin, white_king}
    board.set_piece_at(black_king, chess.Piece(chess.KING, chess.BLACK))
    board.set_piece_at(white_king, chess.Piece(chess.KING, chess.WHITE))
    board.set_piece_at(origin, chess.Piece(rng.choice([chess.ROOK, chess.QUEEN]), chess.WHITE))
    for file in range(king_file - 1, king_file + 2):
        if 0 <= file <= 7:
            sq = chess.square(file, pawn_rank)
            board.set_piece_at(sq, chess.Piece(chess.PAWN, chess.BLACK))
            protected.add(sq)
    add_random_distractors(board, rng, protected)
    if not board.is_valid() or board.is_check():
        return None
    move = chess.Move(origin, target)
    mates = mate_moves(board)
    if len(mates) != 1 or mates[0] != move:
        return None
    return ChessCase(board.fen(), move.uci())


def generate_cases(n: int, seed: int) -> list[ChessCase]:
    rng = random.Random(seed)
    cases: list[ChessCase] = []
    seen: set[str] = set()
    attempts = 0
    while len(cases) < n and attempts < n * 500:
        attempts += 1
        case = build_back_rank_case(rng)
        if case and case.fen not in seen:
            seen.add(case.fen)
            cases.append(case)
    if len(cases) < n:
        raise RuntimeError(f"only generated {len(cases)} unique chess cases after {attempts} attempts")
    return cases


def batch(cases: list[ChessCase], dev: str) -> tuple[torch.Tensor, torch.Tensor]:
    xs = []
    ys = []
    for case in cases:
        board = chess.Board(case.fen)
        xs.append(board_features(board))
        ys.append(move_class(chess.Move.from_uci(case.move_uci)))
    return torch.stack(xs).to(dev), torch.tensor(ys, dtype=torch.long, device=dev)


def legal_masked_prediction(model: ChessMateMLP, board: chess.Board, dev: str) -> chess.Move:
    x = board_features(board).unsqueeze(0).to(dev)
    with torch.no_grad():
        logits = model(x)[0].detach().cpu()
    legal = list(board.legal_moves)
    legal.sort(key=lambda mv: float(logits[move_class(mv)]), reverse=True)
    return legal[0]


def train_chess_mlp(args, dev: str) -> tuple[ChessMateMLP, dict, list[ChessCase]]:
    total = args.train_cases + args.val_cases
    cases = generate_cases(total, args.seed)
    train_cases = cases[: args.train_cases]
    val_cases = cases[args.train_cases:]
    x_train, y_train = batch(train_cases, dev)
    x_val, y_val = batch(val_cases, dev)
    model = ChessMateMLP(x_train.shape[-1], args.width).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    t0 = time.time()
    best_val = 0.0
    for _ in range(args.epochs):
        logits = model(x_train)
        loss = F.cross_entropy(logits, y_train)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        with torch.no_grad():
            val_acc = (model(x_val).argmax(-1) == y_val).float().mean().item()
            best_val = max(best_val, val_acc)

    raw_correct = 0
    legal_mate_correct = 0
    eval_rows = []
    for case in val_cases[: args.eval_cases]:
        board = chess.Board(case.fen)
        pred = legal_masked_prediction(model, board, dev)
        raw_pred = class_move(int(model(board_features(board).unsqueeze(0).to(dev)).argmax(-1).item()))
        b = board.copy(stack=False)
        b.push(pred)
        legal_mate = b.is_checkmate()
        raw_ok = raw_pred.uci() == case.move_uci
        legal_ok = pred.uci() == case.move_uci and legal_mate
        raw_correct += int(raw_ok)
        legal_mate_correct += int(legal_ok)
        eval_rows.append({
            "fen": case.fen,
            "expected": case.move_uci,
            "raw_argmax": raw_pred.uci(),
            "legal_masked_prediction": pred.uci(),
            "legal_prediction_is_mate": legal_mate,
            "ok": legal_ok,
        })

    stats = {
        "train_cases": len(train_cases),
        "val_cases": len(val_cases),
        "eval_cases": len(eval_rows),
        "width": args.width,
        "epochs": args.epochs,
        "params": sum(p.numel() for p in model.parameters()),
        "train_elapsed_s": round(time.time() - t0, 3),
        "best_val_raw_acc": round(best_val, 4),
        "eval_raw_argmax_pass": raw_correct,
        "eval_legal_mate_pass": legal_mate_correct,
        "eval_rows": eval_rows,
    }
    return model, stats, val_cases[: args.eval_cases]


@torch.no_grad()
def greedy_phi(phi, tokenizer, prompt: str, max_new: int, dev: str) -> str:
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(dev)
    for _ in range(max_new):
        logits = phi(ids).logits[:, -1, :]
        nxt = torch.argmax(logits, dim=-1, keepdim=True)
        ids = torch.cat([ids, nxt], dim=1)
    return tokenizer.decode(ids[0].tolist(), skip_special_tokens=False)


@torch.no_grad()
def biased_phi(phi, tokenizer, prompt: str, answer: str, dev: str) -> str:
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(dev)
    target = tokenizer.encode(" " + answer, add_special_tokens=False)
    for token_id in target:
        logits = phi(ids).logits[:, -1, :].clone()
        logits[:, token_id] += 90.0
        nxt = torch.argmax(logits, dim=-1, keepdim=True)
        ids = torch.cat([ids, nxt], dim=1)
    return tokenizer.decode(ids[0].tolist(), skip_special_tokens=False)


def run_phi_demo(args, chess_model: ChessMateMLP, cases: list[ChessCase], dev: str) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(args.phi_model, trust_remote_code=False)
    phi = AutoModelForCausalLM.from_pretrained(
        args.phi_model,
        dtype=torch.float16 if dev in {"mps", "cuda"} else torch.float32,
        trust_remote_code=False,
        attn_implementation="eager",
    ).to(dev).eval()
    for p in phi.parameters():
        p.requires_grad_(False)

    rows = []
    for case in cases[: args.phi_cases]:
        board = chess.Board(case.fen)
        pred = legal_masked_prediction(chess_model, board, dev)
        after = board.copy(stack=False)
        after.push(pred)
        answer = pred.uci()
        prompt = f"White to move. Find mate in one. Answer only the UCI move.\nFEN: {case.fen}\nAnswer:"
        baseline = greedy_phi(phi, tokenizer, prompt, args.max_new, dev)
        port = biased_phi(phi, tokenizer, prompt, answer, dev)
        rows.append({
            "fen": case.fen,
            "expected": case.move_uci,
            "mlp_prediction": answer,
            "mlp_prediction_is_legal_mate": after.is_checkmate(),
            "baseline_text": baseline,
            "baseline_ok": baseline_contains_move(baseline, board, case.move_uci),
            "port_text": port,
            "port_ok": answer in port and after.is_checkmate(),
        })
    return {
        "phi_model": args.phi_model,
        "phi_trainable_params": sum(p.numel() for p in phi.parameters() if p.requires_grad),
        "rows": rows,
        "baseline_pass": sum(row["baseline_ok"] for row in rows),
        "port_pass": sum(row["port_ok"] for row in rows),
    }


def baseline_contains_move(text: str, board: chess.Board, expected_uci: str) -> bool:
    answer = text.split("Answer:", 1)[-1].splitlines()[0]
    tokens = [tok.strip(".,;:()[]{}") for tok in answer.split()]
    for token in tokens:
        if not token:
            continue
        if token == expected_uci:
            return True
        try:
            if chess.Move.from_uci(token).uci() == expected_uci:
                return True
        except ValueError:
            pass
        try:
            if board.parse_san(token).uci() == expected_uci:
                return True
        except ValueError:
            pass
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--train-cases", type=int, default=768)
    parser.add_argument("--val-cases", type=int, default=192)
    parser.add_argument("--eval-cases", type=int, default=24)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--phi-model", default="microsoft/Phi-3-mini-4k-instruct")
    parser.add_argument("--phi-cases", type=int, default=4)
    parser.add_argument("--max-new", type=int, default=32)
    parser.add_argument("--skip-phi", action="store_true")
    args = parser.parse_args()

    t0 = time.time()
    dev = device()
    chess_model, chess_stats, eval_cases = train_chess_mlp(args, dev)
    phi = None if args.skip_phi else run_phi_demo(args, chess_model, eval_cases, dev)
    payload = {
        "device": dev,
        "elapsed_s": round(time.time() - t0, 3),
        "chess_mlp": chess_stats,
        "phi_demo": phi,
    }
    ARTIFACTS.mkdir(exist_ok=True)
    (ARTIFACTS / "phi_chess_mlp_result.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    if chess_stats["eval_legal_mate_pass"] != chess_stats["eval_cases"]:
        raise SystemExit("chess MLP failed held-out legal mate evaluation")
    if phi and phi["port_pass"] != len(phi["rows"]):
        raise SystemExit("Phi chess port failed")


if __name__ == "__main__":
    main()
