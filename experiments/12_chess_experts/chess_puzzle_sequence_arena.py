from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

import chess
import torch

import chess_motif_generalization as motif
from chess_policy_arena import train_jepa_policy, train_motif_policy, jepa_policy_move


HERE = Path(__file__).resolve().parent
ARTIFACT = HERE / "artifacts" / "chess_puzzle_sequence_arena_result.json"


@dataclass(frozen=True)
class SequencePuzzle:
    motif: str
    fen: str
    line: tuple[str, ...]


def mate_in_one_moves(board: chess.Board) -> list[chess.Move]:
    out = []
    for move in board.legal_moves:
        nxt = board.copy(stack=False)
        nxt.push(move)
        if nxt.is_checkmate():
            out.append(move)
    return out


def best_defense(board: chess.Board, rng: random.Random) -> chess.Move | None:
    legal = list(board.legal_moves)
    if not legal:
        return None
    non_mate_next = []
    for move in legal:
        nxt = board.copy(stack=False)
        nxt.push(move)
        if not mate_in_one_moves(nxt):
            non_mate_next.append(move)
    pool = non_mate_next or legal
    captures = [move for move in pool if board.is_capture(move)]
    checks = [move for move in pool if board.gives_check(move)]
    if checks:
        return sorted(checks, key=lambda mv: mv.uci())[0]
    if captures:
        return sorted(captures, key=lambda mv: mv.uci())[0]
    return sorted(pool, key=lambda mv: mv.uci())[0] if rng.random() < 0.85 else rng.choice(pool)


def build_two_step_puzzles(args) -> list[SequencePuzzle]:
    rng = random.Random(args.seed + 900)
    _, base_cases = motif.generate_cases(args.train_per_family, args.val_per_family * 16, args.seed + 91)
    puzzles: list[SequencePuzzle] = []
    seen: set[str] = set()
    attempts = 0
    for case in base_cases:
        if len(puzzles) >= args.puzzles:
            break
        attempts += 1
        mate_board = chess.Board(case.fen)
        mating_move = chess.Move.from_uci(case.move_uci)
        piece = mate_board.piece_at(mating_move.from_square)
        if piece is None:
            continue
        black_pieces = [
            (sq, p)
            for sq, p in mate_board.piece_map().items()
            if p.color == chess.BLACK and p.piece_type != chess.KING
        ]
        rng.shuffle(black_pieces)
        built = None
        for reply_from, reply_piece in black_pieces[:8]:
            for reply_to in chess.SQUARES:
                if reply_to == reply_from or mate_board.piece_at(reply_to) is not None:
                    continue
                before = mate_board.copy(stack=False)
                before.turn = chess.BLACK
                before.remove_piece_at(reply_from)
                before.set_piece_at(reply_to, reply_piece)
                if not before.is_valid() or before.is_check():
                    continue
                reply = chess.Move(reply_to, reply_from)
                if reply not in before.legal_moves:
                    continue
                after_reply = before.copy(stack=False)
                after_reply.push(reply)
                if after_reply.board_fen() != mate_board.board_fen():
                    continue
                if mating_move not in mate_in_one_moves(after_reply):
                    continue
                built = (before, reply, mating_move)
                break
            if built:
                break
        if built is None:
            continue
        before_board, reply, mate = built
        key = before_board.fen()
        if key in seen:
            continue
        seen.add(key)
        puzzles.append(SequencePuzzle(case.motif, key, (reply.uci(), mate.uci())))
    if len(puzzles) < args.puzzles:
        raise RuntimeError(f"only generated {len(puzzles)}/{args.puzzles} sequence puzzles after {attempts} attempts")
    return puzzles


@torch.no_grad()
def motif_move(model: motif.MateMLP, board: chess.Board, dev: str) -> chess.Move:
    return motif.legal_masked_prediction(model, board, dev)


def solve_sequence(name: str, model, puzzle: SequencePuzzle, dev: str) -> dict:
    board = chess.Board(puzzle.fen)
    moves = []
    ok = True
    for ply, expected in enumerate(puzzle.line):
        if board.is_game_over(claim_draw=True):
            ok = False
            break
        if board.turn == chess.WHITE:
            move = motif_move(model, board, dev) if name == "motif" else jepa_policy_move(model, board, dev)
            if move not in board.legal_moves:
                ok = False
                break
            moves.append(move.uci())
            board.push(move)
        else:
            move = chess.Move.from_uci(expected)
            if move not in board.legal_moves:
                ok = False
                break
            moves.append(move.uci())
            board.push(move)
    return {
        "moves": moves,
        "solved": ok and board.is_checkmate(),
        "terminal": "checkmate" if board.is_checkmate() else ("game_over" if board.is_game_over(claim_draw=True) else "incomplete"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=53)
    parser.add_argument("--train-per-family", type=int, default=384)
    parser.add_argument("--val-per-family", type=int, default=24)
    parser.add_argument("--puzzles", type=int, default=48)
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
    parser.add_argument("--sample-rows", type=int, default=12)
    args = parser.parse_args()

    t0 = time.time()
    dev = motif.device()
    motif_policy = train_motif_policy(args, dev)
    jepa_policy, _ = train_jepa_policy(args, dev)
    puzzles = build_two_step_puzzles(args)

    motif_solved = 0
    jepa_solved = 0
    motif_wins = 0
    jepa_wins = 0
    ties_both = 0
    ties_fail = 0
    rows = []
    by_family: dict[str, dict[str, int]] = {}
    for puzzle in puzzles:
        motif_result = solve_sequence("motif", motif_policy, puzzle, dev)
        jepa_result = solve_sequence("jepa", jepa_policy, puzzle, dev)
        motif_ok = motif_result["solved"]
        jepa_ok = jepa_result["solved"]
        motif_solved += int(motif_ok)
        jepa_solved += int(jepa_ok)
        fam = by_family.setdefault(puzzle.motif, {"n": 0, "motif": 0, "jepa": 0})
        fam["n"] += 1
        fam["motif"] += int(motif_ok)
        fam["jepa"] += int(jepa_ok)
        if motif_ok and not jepa_ok:
            motif_wins += 1
            outcome = "motif_win"
        elif jepa_ok and not motif_ok:
            jepa_wins += 1
            outcome = "jepa_win"
        elif motif_ok and jepa_ok:
            ties_both += 1
            outcome = "tie_both_solve"
        else:
            ties_fail += 1
            outcome = "tie_both_fail"
        if len(rows) < args.sample_rows:
            rows.append({
                "motif": puzzle.motif,
                "fen": puzzle.fen,
                "target_line": list(puzzle.line),
                "motif_moves": motif_result["moves"],
                "motif_solved": motif_ok,
                "jepa_moves": jepa_result["moves"],
                "jepa_solved": jepa_ok,
                "outcome": outcome,
            })

    n = len(puzzles)
    payload = {
        "status": "multi_ply_puzzle_arena",
        "scope": "constructed two-move attacking sequences with one defender reply",
        "device": dev,
        "elapsed_s": round(time.time() - t0, 3),
        "puzzles": n,
        "score": {
            "motif_solved": motif_solved,
            "jepa_solved": jepa_solved,
            "motif_solve_rate": round(motif_solved / n, 4),
            "jepa_solve_rate": round(jepa_solved / n, 4),
            "motif_wins": motif_wins,
            "jepa_wins": jepa_wins,
            "ties_both_solve": ties_both,
            "ties_both_fail": ties_fail,
        },
        "by_family": {
            name: {
                "n": stats["n"],
                "motif_solve_rate": round(stats["motif"] / stats["n"], 4),
                "jepa_solve_rate": round(stats["jepa"] / stats["n"], 4),
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
