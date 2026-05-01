from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

import chess
import torch

import chess_jepa_bridge as jepa
import chess_motif_generalization as motif
from chess_competition_sweep import train_jepa_encoder, train_jepa_policy, train_motif_policy
from chess_policy_arena import JepaPolicy


HERE = Path(__file__).resolve().parent
ARTIFACT = HERE / "artifacts" / "chess_full_game_arena_result.json"
PIECE_VALUES = {
    chess.PAWN: 1.0,
    chess.KNIGHT: 3.0,
    chess.BISHOP: 3.0,
    chess.ROOK: 5.0,
    chess.QUEEN: 9.0,
    chess.KING: 0.0,
}


@dataclass
class Player:
    name: str
    model: motif.MateMLP | JepaPolicy
    kind: str


def material_score(board: chess.Board) -> float:
    score = 0.0
    for piece in board.piece_map().values():
        value = PIECE_VALUES[piece.piece_type]
        score += value if piece.color == chess.WHITE else -value
    return score


def center_bonus(move: chess.Move) -> float:
    file = chess.square_file(move.to_square)
    rank = chess.square_rank(move.to_square)
    return (3.5 - abs(file - 3.5) + 3.5 - abs(rank - 3.5)) / 14.0


def position_key(board: chess.Board) -> str:
    return " ".join(board.fen().split(" ")[:4])


def model_scores(player: Player, board: chess.Board, dev: str) -> torch.Tensor:
    if player.kind == "motif":
        x = motif.board_features(board).unsqueeze(0).to(dev)
        return player.model(x)[0].detach().cpu()
    x = jepa.board_to_tensor(board).unsqueeze(0).to(dev)
    return player.model(x)[0].detach().cpu()


@torch.no_grad()
def choose_move(
    player: Player,
    board: chess.Board,
    dev: str,
    blend: float,
    temperature: float,
    anti_repetition: float,
    seen: dict[str, int],
    rng: random.Random,
) -> tuple[chess.Move, dict]:
    logits = model_scores(player, board, dev)
    candidates = []
    before_material = material_score(board)
    for move in board.legal_moves:
        after = board.copy(stack=False)
        after.push(move)
        if after.is_checkmate():
            return move, {"reason": "immediate_checkmate", "model_rank": 1}

        raw = float(logits[motif.move_class(move)])
        material_delta = material_score(after) - before_material
        if board.turn == chess.BLACK:
            material_delta = -material_delta
        prior = 0.0
        prior += 1.2 * material_delta
        prior += 0.5 if board.gives_check(move) else 0.0
        prior += 0.4 if move.promotion is not None else 0.0
        prior += 0.15 * center_bonus(move)
        if after.is_check():
            prior -= 0.2
        prior -= anti_repetition * seen.get(position_key(after), 0)
        if after.can_claim_threefold_repetition():
            prior -= 2.0 * anti_repetition
        score = blend * raw + (1.0 - blend) * prior
        candidates.append((score, raw, prior, move))

    candidates.sort(key=lambda item: item[0], reverse=True)
    top_k = candidates[: min(5, len(candidates))]
    if temperature <= 0:
        score, raw, prior, move = top_k[0]
        return move, {"reason": "argmax", "model_score": round(raw, 4), "prior": round(prior, 4)}

    weights = [pow(2.718281828, item[0] / temperature) for item in top_k]
    total = sum(weights)
    pick = rng.random() * total
    running = 0.0
    for weight, item in zip(weights, top_k):
        running += weight
        if running >= pick:
            score, raw, prior, move = item
            return move, {"reason": "sampled_top5", "model_score": round(raw, 4), "prior": round(prior, 4)}
    score, raw, prior, move = top_k[0]
    return move, {"reason": "fallback_argmax", "model_score": round(raw, 4), "prior": round(prior, 4)}


def warmup_board(seed: int, plies: int) -> chess.Board:
    rng = random.Random(seed)
    board = chess.Board()
    for _ in range(plies):
        if board.is_game_over(claim_draw=True):
            return chess.Board()
        legal = list(board.legal_moves)
        captures = [move for move in legal if board.is_capture(move)]
        checks = [move for move in legal if board.gives_check(move)]
        pool = checks or captures or legal
        board.push(rng.choice(pool))
    return board


def adjudicate(board: chess.Board) -> tuple[str, str, float]:
    if board.is_checkmate():
        winner = "black" if board.turn == chess.WHITE else "white"
        return winner, "checkmate", material_score(board)
    if board.is_stalemate():
        return "draw", "stalemate", material_score(board)
    if board.is_insufficient_material():
        return "draw", "insufficient_material", material_score(board)
    if board.can_claim_draw():
        return "draw", "claimable_draw", material_score(board)
    material = material_score(board)
    if material > 1.5:
        return "white", "material_adjudication", material
    if material < -1.5:
        return "black", "material_adjudication", material
    return "draw", "material_adjudication", material


def play_game(
    white: Player,
    black: Player,
    start_board: chess.Board,
    game_seed: int,
    max_plies: int,
    dev: str,
    blend: float,
    temperature: float,
    anti_repetition: float,
) -> dict:
    rng = random.Random(game_seed)
    board = start_board.copy(stack=False)
    seen = {position_key(board): 1}
    moves = []
    reasons: dict[str, int] = {}
    for ply in range(max_plies):
        if board.is_game_over(claim_draw=True):
            break
        player = white if board.turn == chess.WHITE else black
        move, info = choose_move(player, board, dev, blend, temperature, anti_repetition, seen, rng)
        reasons[info["reason"]] = reasons.get(info["reason"], 0) + 1
        san = board.san(move)
        board.push(move)
        key = position_key(board)
        seen[key] = seen.get(key, 0) + 1
        moves.append(san)

    winner, reason, material = adjudicate(board)
    if winner == "white":
        winner_name = white.name
    elif winner == "black":
        winner_name = black.name
    else:
        winner_name = "draw"
    return {
        "white": white.name,
        "black": black.name,
        "start_fen": start_board.fen(),
        "plies": len(moves),
        "result": board.result(claim_draw=True) if board.is_game_over(claim_draw=True) else "*",
        "winner_color": winner,
        "winner": winner_name,
        "termination": reason,
        "final_material_white_minus_black": round(material, 2),
        "final_fen": board.fen(),
        "selection_reasons": reasons,
        "moves_san": moves[:120],
    }


def summarize(games: list[dict]) -> dict:
    wins = {"motif": 0, "jepa": 0, "draw": 0}
    terminations: dict[str, int] = {}
    plies = 0
    for game in games:
        if game["winner"] == "motif":
            wins["motif"] += 1
        elif game["winner"] == "jepa":
            wins["jepa"] += 1
        else:
            wins["draw"] += 1
        terminations[game["termination"]] = terminations.get(game["termination"], 0) + 1
        plies += game["plies"]
    return {
        "games": len(games),
        "motif_wins": wins["motif"],
        "jepa_wins": wins["jepa"],
        "draws": wins["draw"],
        "avg_plies": round(plies / max(len(games), 1), 2),
        "terminations": terminations,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--games", type=int, default=24)
    parser.add_argument("--max-plies", type=int, default=180)
    parser.add_argument("--opening-plies", type=int, default=6)
    parser.add_argument("--train-per-family", type=int, default=1024)
    parser.add_argument("--max-train-per-family", type=int, default=1024)
    parser.add_argument("--val-per-family", type=int, default=32)
    parser.add_argument("--motif-width", type=int, default=1024)
    parser.add_argument("--jepa-pairs", type=int, default=8000)
    parser.add_argument("--jepa-val-pairs", type=int, default=800)
    parser.add_argument("--jepa-width", type=int, default=384)
    parser.add_argument("--latent-dim", type=int, default=96)
    parser.add_argument("--jepa-epochs", type=int, default=28)
    parser.add_argument("--policy-width", type=int, default=768)
    parser.add_argument("--policy-epochs", type=int, default=220)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--blend", type=float, default=0.78)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--anti-repetition", type=float, default=1.5)
    args = parser.parse_args()

    t0 = time.time()
    dev = motif.device()
    train_pool, _ = motif.generate_cases(args.max_train_per_family, args.val_per_family, args.seed)
    train_cases = []
    grouped = {name: [] for name in motif.MOTIFS}
    for case in train_pool:
        grouped[case.motif].append(case)
    for name in motif.MOTIFS:
        train_cases.extend(grouped[name][: args.train_per_family])

    motif_player = Player("motif", train_motif_policy(train_cases, args, dev), "motif")
    encoder, bridge_metrics = train_jepa_encoder(args, dev, args.seed)
    jepa_player = Player("jepa", train_jepa_policy(encoder, train_cases, args, dev), "jepa")

    games = []
    for game_idx in range(args.games):
        pair_idx = game_idx // 2
        start = warmup_board(args.seed + pair_idx, args.opening_plies)
        if game_idx % 2 == 0:
            white, black = motif_player, jepa_player
        else:
            white, black = jepa_player, motif_player
        games.append(
            play_game(
                white,
                black,
                start,
                args.seed + 10_000 + game_idx,
                args.max_plies,
                dev,
                args.blend,
                args.temperature,
                args.anti_repetition,
            )
        )

    payload = {
        "status": "full_game_arena",
        "scope": "paired full legal chess games with color swaps; policies are still trained from tactical generated data",
        "device": dev,
        "elapsed_s": round(time.time() - t0, 3),
        "config": vars(args),
        "jepa_bridge_pretrain": {
            "cosine_mean": round(float(bridge_metrics["cosine_mean"]), 6),
            "nearest_neighbor_top1": round(float(bridge_metrics["nearest_neighbor_top1"]), 4),
            "nearest_neighbor_top5": round(float(bridge_metrics["nearest_neighbor_top5"]), 4),
        },
        "summary": summarize(games),
        "games": games,
    }
    ARTIFACT.parent.mkdir(exist_ok=True)
    ARTIFACT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
