from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

import chess
import torch
import torch.nn as nn
import torch.nn.functional as F

import chess_motif_generalization as motif
from chess_full_game_arena import material_score, position_key, warmup_board
from chess_full_game_trace_arena import teacher_score


HERE = Path(__file__).resolve().parent
ARTIFACT = HERE / "artifacts" / "chess_online_world_model_result.json"
MOVE_DIM = 128
TACTICAL_DIM = 4
FEATURE_DIM = 12 * 64 + 5 + MOVE_DIM + 4 + TACTICAL_DIM
PIECE_VALUES = {
    chess.PAWN: 1.0,
    chess.KNIGHT: 3.0,
    chess.BISHOP: 3.0,
    chess.ROOK: 5.0,
    chess.QUEEN: 9.0,
    chess.KING: 0.0,
}


@dataclass(frozen=True)
class MoveSample:
    fen: str
    move_uci: str
    mover: chess.Color
    final_value: float
    material_after: float
    terminal: str


class OnlineValueModel(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(FEATURE_DIM, width),
            nn.GELU(),
            nn.LayerNorm(width),
            nn.Linear(width, width),
            nn.GELU(),
            nn.LayerNorm(width),
            nn.Linear(width, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def encode_move(move: chess.Move) -> torch.Tensor:
    x = torch.zeros(MOVE_DIM, dtype=torch.float32)
    x[move.from_square] = 1.0
    x[64 + move.to_square] = 1.0
    return x


def material_for_turn(board: chess.Board, color: chess.Color) -> float:
    value = material_score(board)
    return value if color == chess.WHITE else -value


def capture_value(board: chess.Board, move: chess.Move) -> float:
    if not board.is_capture(move):
        return 0.0
    victim = board.piece_at(move.to_square)
    if victim is None and board.is_en_passant(move):
        victim = chess.Piece(chess.PAWN, not board.turn)
    return PIECE_VALUES.get(victim.piece_type, 0.0) if victim else 0.0


def capture_victim(board: chess.Board, move: chess.Move) -> chess.Piece | None:
    victim = board.piece_at(move.to_square)
    if victim is None and board.is_en_passant(move):
        return chess.Piece(chess.PAWN, not board.turn)
    return victim


def tactical_audit(board: chess.Board, move: chess.Move) -> dict:
    mover = board.turn
    after = board.copy(stack=False)
    before_material = material_for_turn(board, mover)
    after.push(move)
    if after.is_checkmate():
        return {
            "opponent_best_capture": 0.0,
            "opponent_best_reply_delta": 0.0,
            "queen_hang": False,
            "major_piece_hang": False,
            "tactical_blunder": False,
        }

    best_capture = 0.0
    best_reply_delta = 0.0
    queen_hang = False
    major_piece_hang = False
    for reply in after.legal_moves:
        reply_after = after.copy(stack=False)
        victim = capture_victim(after, reply)
        reply_after.push(reply)
        reply_delta = material_for_turn(reply_after, mover) - before_material
        best_reply_delta = min(best_reply_delta, reply_delta)
        if victim is None or victim.color != mover:
            continue
        value = PIECE_VALUES.get(victim.piece_type, 0.0)
        best_capture = max(best_capture, value)
        queen_hang = queen_hang or victim.piece_type == chess.QUEEN
        major_piece_hang = major_piece_hang or victim.piece_type in {chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT}

    compensation = material_for_turn(after, mover) - before_material
    tactical_blunder = (queen_hang or best_capture >= 5.0 or best_reply_delta <= -5.0) and compensation < 4.0
    return {
        "opponent_best_capture": round(best_capture, 3),
        "opponent_best_reply_delta": round(best_reply_delta, 3),
        "queen_hang": queen_hang,
        "major_piece_hang": major_piece_hang,
        "tactical_blunder": tactical_blunder,
    }


def feature(board: chess.Board, move: chess.Move, heuristic: float) -> torch.Tensor:
    after = board.copy(stack=False)
    before_material = material_for_turn(board, board.turn)
    after.push(move)
    after_material = material_for_turn(after, board.turn)
    audit = tactical_audit(board, move)
    scalars = torch.tensor(
        [
            max(-1.0, min(1.0, heuristic / 12.0)),
            max(-1.0, min(1.0, (after_material - before_material) / 9.0)),
            1.0 if board.gives_check(move) else 0.0,
            max(0.0, min(1.0, capture_value(board, move) / 9.0)),
            max(0.0, min(1.0, audit["opponent_best_capture"] / 9.0)),
            max(-1.0, min(1.0, audit["opponent_best_reply_delta"] / 9.0)),
            1.0 if audit["queen_hang"] else 0.0,
            1.0 if audit["tactical_blunder"] else 0.0,
        ],
        dtype=torch.float32,
    )
    return torch.cat([motif.board_features(board), encode_move(move), scalars])


def heuristic_candidates(board: chess.Board, seen: dict[str, int], top_k: int) -> list[tuple[float, chess.Move]]:
    scored = [(teacher_score(board, move, seen), move) for move in board.legal_moves]
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[: min(top_k, len(scored))]


@torch.no_grad()
def choose_static(board: chess.Board, seen: dict[str, int], top_k: int) -> tuple[chess.Move, dict]:
    score, move = heuristic_candidates(board, seen, top_k)[0]
    return move, {"heuristic": round(score, 4), "value": None, "mixed": round(score, 4)}


@torch.no_grad()
def choose_adaptive(
    model: OnlineValueModel,
    board: chess.Board,
    seen: dict[str, int],
    dev: str,
    top_k: int,
    value_weight: float,
    rng: random.Random,
    exploration: float,
    decision_mode: str,
    max_heuristic_drop: float,
    value_margin: float,
) -> tuple[chess.Move, dict]:
    candidates = heuristic_candidates(board, seen, top_k)
    if rng.random() < exploration and len(candidates) > 1:
        score, move = rng.choice(candidates[: min(4, len(candidates))])
        return move, {"heuristic": round(score, 4), "value": None, "mixed": round(score, 4), "explore": True}
    if value_weight <= 0:
        score, move = candidates[0]
        return move, {"heuristic": round(score, 4), "value": None, "mixed": round(score, 4), "decision": "heuristic_cold_start"}
    feats = torch.stack([feature(board, move, score) for score, move in candidates]).to(dev)
    values = model(feats).detach().cpu().tolist()
    if decision_mode == "veto":
        best_score, best_move = candidates[0]
        best_value = float(values[0])
        eligible = []
        for (score, move), value in zip(candidates, values):
            if best_score - score <= max_heuristic_drop:
                eligible.append((float(value), score, move))
        eligible.sort(key=lambda item: item[0], reverse=True)
        value, score, move = eligible[0]
        if move != best_move and value - best_value >= value_margin:
            return move, {
                "heuristic": round(score, 4),
                "value": round(value, 4),
                "mixed": round(value, 4),
                "decision": "value_veto",
            }
        return best_move, {
            "heuristic": round(best_score, 4),
            "value": round(best_value, 4),
            "mixed": round(best_value, 4),
            "decision": "heuristic_kept",
        }
    mixed = []
    best_h = candidates[0][0]
    for (score, move), value in zip(candidates, values):
        h = math.tanh((score - best_h) / 6.0)
        mixed.append((h + value_weight * float(value), score, float(value), move))
    mixed.sort(key=lambda item: item[0], reverse=True)
    total, score, value, move = mixed[0]
    return move, {"heuristic": round(score, 4), "value": round(value, 4), "mixed": round(total, 4)}


def game_value(board: chess.Board, mover: chess.Color) -> float:
    if board.is_checkmate():
        winner = not board.turn
        return 1.0 if winner == mover else -1.0
    if board.is_stalemate() or board.is_insufficient_material() or board.can_claim_draw():
        return 0.0
    material = material_for_turn(board, mover)
    return max(-0.85, min(0.85, material / 10.0))


def play_match_game(
    adaptive_model: OnlineValueModel,
    adaptive_color: chess.Color,
    start_board: chess.Board,
    args,
    dev: str,
    rng: random.Random,
    exploration: float,
    value_weight: float,
) -> tuple[dict, list[MoveSample]]:
    board = start_board.copy(stack=False)
    seen = {position_key(board): 1}
    moves = []
    trace: list[tuple[str, str, chess.Color, float]] = []
    for _ply in range(args.max_plies):
        if board.is_game_over(claim_draw=True):
            break
        fen = board.fen()
        mover = board.turn
        if mover == adaptive_color:
            move, info = choose_adaptive(
                adaptive_model,
                board,
                seen,
                dev,
                args.top_k,
                value_weight,
                rng,
                exploration,
                args.decision_mode,
                args.max_heuristic_drop,
                args.value_margin,
            )
            policy = "adaptive_world_model"
        else:
            move, info = choose_static(board, seen, args.top_k)
            policy = "static_alpha_lite"
        san = board.san(move)
        audit = tactical_audit(board, move)
        board.push(move)
        seen[position_key(board)] = seen.get(position_key(board), 0) + 1
        moves.append({"san": san, "uci": move.uci(), "policy": policy, **info, "audit": audit})
        trace.append((fen, move.uci(), mover, material_for_turn(board, mover)))

    samples = [
        MoveSample(
            fen=fen,
            move_uci=move_uci,
            mover=mover,
            final_value=game_value(board, mover),
            material_after=material_after,
            terminal="checkmate" if board.is_checkmate() else ("draw" if board.is_game_over(claim_draw=True) else "adjudicated"),
        )
        for fen, move_uci, mover, material_after in trace
    ]
    adaptive_value = game_value(board, adaptive_color)
    winner = "adaptive_world_model" if adaptive_value > 0.2 else ("static_alpha_lite" if adaptive_value < -0.2 else "draw")
    return {
        "adaptive_color": "white" if adaptive_color == chess.WHITE else "black",
        "start_fen": start_board.fen(),
        "plies": len(moves),
        "winner": winner,
        "adaptive_value": round(adaptive_value, 4),
        "final_material_for_adaptive": round(material_for_turn(board, adaptive_color), 2),
        "final_fen": board.fen(),
        "moves_san": [row["san"] for row in moves],
        "audit_summary": summarize_move_audit(moves),
        "move_details": moves[: args.sample_ply_details],
    }, samples


def train_online(model: OnlineValueModel, opt: torch.optim.Optimizer, replay: list[MoveSample], args, dev: str) -> dict:
    if not replay:
        return {"updates": 0, "loss": None}
    model.train()
    losses = []
    batch_size = min(args.batch_size, len(replay))
    for _ in range(args.online_epochs):
        batch = random.sample(replay, batch_size)
        xs = []
        ys = []
        for sample in batch:
            board = chess.Board(sample.fen)
            move = chess.Move.from_uci(sample.move_uci)
            h = teacher_score(board, move, {})
            audit = tactical_audit(board, move)
            blunder_penalty = 0.0
            blunder_penalty += args.blunder_penalty if audit["tactical_blunder"] else 0.0
            blunder_penalty += args.queen_hang_penalty if audit["queen_hang"] else 0.0
            target = 0.72 * sample.final_value + 0.28 * max(-1.0, min(1.0, sample.material_after / 10.0))
            target -= blunder_penalty
            target = max(-1.0, min(1.0, target))
            xs.append(feature(board, move, h))
            ys.append(target)
        x = torch.stack(xs).to(dev)
        y = torch.tensor(ys, dtype=torch.float32, device=dev)
        pred = model(x)
        loss = F.mse_loss(pred, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach().cpu()))
    model.eval()
    return {"updates": args.online_epochs, "loss": round(sum(losses) / len(losses), 5)}


def summarize(games: list[dict]) -> dict:
    wins = sum(1 for game in games if game["winner"] == "adaptive_world_model")
    losses = sum(1 for game in games if game["winner"] == "static_alpha_lite")
    draws = len(games) - wins - losses
    values = [game["adaptive_value"] for game in games]
    audit = summarize_tactical_audit(games)
    return {
        "games": len(games),
        "adaptive_wins": wins,
        "static_wins": losses,
        "draws": draws,
        "adaptive_score_rate": round((wins + 0.5 * draws) / max(len(games), 1), 4),
        "avg_adaptive_value": round(sum(values) / max(len(values), 1), 4),
        "avg_plies": round(sum(game["plies"] for game in games) / max(len(games), 1), 2),
        "tactical_audit": audit,
    }


def empty_audit_totals() -> dict:
    return {"moves": 0, "queen_hangs": 0, "major_piece_hangs": 0, "tactical_blunders": 0, "best_capture_sum": 0.0}


def summarize_move_audit(moves: list[dict]) -> dict:
    totals = {
        "adaptive_world_model": empty_audit_totals(),
        "static_alpha_lite": empty_audit_totals(),
    }
    for move in moves:
        policy = move.get("policy")
        if policy not in totals:
            continue
        audit = move.get("audit") or {}
        totals[policy]["moves"] += 1
        totals[policy]["queen_hangs"] += int(bool(audit.get("queen_hang")))
        totals[policy]["major_piece_hangs"] += int(bool(audit.get("major_piece_hang")))
        totals[policy]["tactical_blunders"] += int(bool(audit.get("tactical_blunder")))
        totals[policy]["best_capture_sum"] += float(audit.get("opponent_best_capture") or 0.0)
    return finish_audit_totals(totals)


def summarize_tactical_audit(games: list[dict]) -> dict:
    totals = {
        "adaptive_world_model": empty_audit_totals(),
        "static_alpha_lite": empty_audit_totals(),
    }
    for game in games:
        summary = game.get("audit_summary") or {}
        for policy, stats in summary.items():
            if policy not in totals:
                continue
            totals[policy]["moves"] += stats.get("moves", 0)
            totals[policy]["queen_hangs"] += stats.get("queen_hangs", 0)
            totals[policy]["major_piece_hangs"] += stats.get("major_piece_hangs", 0)
            totals[policy]["tactical_blunders"] += stats.get("tactical_blunders", 0)
            totals[policy]["best_capture_sum"] += stats.get("best_capture_sum", 0.0)
    return finish_audit_totals(totals)


def finish_audit_totals(totals: dict) -> dict:
    out = {}
    for policy, stats in totals.items():
        n = max(stats["moves"], 1)
        out[policy] = {
            "moves": stats["moves"],
            "queen_hangs": stats["queen_hangs"],
            "major_piece_hangs": stats["major_piece_hangs"],
            "tactical_blunders": stats["tactical_blunders"],
            "best_capture_sum": round(stats["best_capture_sum"], 4),
            "queen_hang_rate": round(stats["queen_hangs"] / n, 4),
            "major_piece_hang_rate": round(stats["major_piece_hangs"] / n, 4),
            "tactical_blunder_rate": round(stats["tactical_blunders"] / n, 4),
            "avg_opponent_best_capture": round(stats["best_capture_sum"] / n, 4),
        }
    return out


def evaluate_fixed(model: OnlineValueModel, iteration: int, args, dev: str, value_weight: float) -> dict:
    rng = random.Random(args.seed + 700_000 + iteration)
    games = []
    for game_idx in range(args.eval_games):
        start = warmup_board(args.seed + 800_000 + game_idx // 2, args.opening_plies)
        adaptive_color = chess.WHITE if game_idx % 2 == 0 else chess.BLACK
        game, _ = play_match_game(model, adaptive_color, start, args, dev, rng, exploration=0.0, value_weight=value_weight)
        games.append(game)
    return {
        "value_weight": round(value_weight, 4),
        "summary": summarize(games),
        "sample_games": games[: args.sample_games],
    }


def run_iteration(iteration: int, model: OnlineValueModel, opt, replay: list[MoveSample], args, dev: str) -> dict:
    rng = random.Random(args.seed + iteration * 10_000)
    games = []
    new_samples: list[MoveSample] = []
    exploration = max(args.min_exploration, args.exploration * (args.exploration_decay ** iteration))
    effective_value_weight = args.value_weight if len(replay) >= args.batch_size else 0.0
    for game_idx in range(args.games_per_iteration):
        start = warmup_board(args.seed + 40_000 + iteration * 100 + game_idx // 2, args.opening_plies)
        adaptive_color = chess.WHITE if game_idx % 2 == 0 else chess.BLACK
        game, samples = play_match_game(model, adaptive_color, start, args, dev, rng, exploration, effective_value_weight)
        games.append(game)
        new_samples.extend(samples)
    replay.extend(new_samples)
    if len(replay) > args.replay_limit:
        del replay[: len(replay) - args.replay_limit]
    update = train_online(model, opt, replay, args, dev)
    evaluation = evaluate_fixed(
        model,
        iteration,
        args,
        dev,
        args.value_weight if len(replay) >= args.batch_size else 0.0,
    )
    return {
        "iteration": iteration,
        "exploration": round(exploration, 4),
        "effective_value_weight": round(effective_value_weight, 4),
        "new_samples": len(new_samples),
        "replay_samples": len(replay),
        "summary": summarize(games),
        "online_update": update,
        "heldout_after_update": evaluation,
        "sample_games": games[: args.sample_games],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=701)
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--games-per-iteration", type=int, default=8)
    parser.add_argument("--eval-games", type=int, default=8)
    parser.add_argument("--max-plies", type=int, default=120)
    parser.add_argument("--opening-plies", type=int, default=6)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--width", type=int, default=192)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--online-epochs", type=int, default=32)
    parser.add_argument("--value-weight", type=float, default=0.35)
    parser.add_argument("--decision-mode", choices=("blend", "veto"), default="blend")
    parser.add_argument("--max-heuristic-drop", type=float, default=3.0)
    parser.add_argument("--value-margin", type=float, default=0.12)
    parser.add_argument("--blunder-penalty", type=float, default=0.35)
    parser.add_argument("--queen-hang-penalty", type=float, default=0.25)
    parser.add_argument("--exploration", type=float, default=0.25)
    parser.add_argument("--exploration-decay", type=float, default=0.72)
    parser.add_argument("--min-exploration", type=float, default=0.03)
    parser.add_argument("--replay-limit", type=int, default=6000)
    parser.add_argument("--sample-games", type=int, default=2)
    parser.add_argument("--sample-ply-details", type=int, default=20)
    args = parser.parse_args()

    t0 = time.time()
    dev = motif.device()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    model = OnlineValueModel(args.width).to(dev)
    model.eval()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    replay: list[MoveSample] = []
    iterations = [run_iteration(i, model, opt, replay, args, dev) for i in range(args.iterations)]
    payload = {
        "status": "online_world_model_self_play",
        "scope": "static alpha-lite heuristic versus the same heuristic augmented by an online value/world model trained from completed self-play games",
        "device": dev,
        "elapsed_s": round(time.time() - t0, 3),
        "config": vars(args),
        "model": {
            "type": "small MLP value model over board, candidate move, heuristic score, material scalars, and explicit tactical-blunder audit features",
            "parameters": sum(p.numel() for p in model.parameters()),
            "feature_dim": FEATURE_DIM,
        },
        "final_summary": iterations[-1]["summary"] if iterations else {},
        "iterations": iterations,
    }
    ARTIFACT.parent.mkdir(exist_ok=True)
    ARTIFACT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
