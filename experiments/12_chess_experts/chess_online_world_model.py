from __future__ import annotations

import argparse
import json
import math
import random
import time
from copy import deepcopy
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
CHECKPOINT_DIR = HERE / "checkpoints" / "chess_online_world_model"
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


def safety_penalty(audit: dict, args) -> float:
    penalty = 0.0
    penalty += args.static_blunder_penalty if audit["tactical_blunder"] else 0.0
    penalty += args.static_queen_hang_penalty if audit["queen_hang"] else 0.0
    penalty += args.static_major_hang_penalty if audit["major_piece_hang"] else 0.0
    penalty += args.static_best_capture_penalty * float(audit["opponent_best_capture"])
    penalty += args.static_reply_delta_penalty * max(0.0, -float(audit["opponent_best_reply_delta"]))
    return penalty


def static_candidates(board: chess.Board, seen: dict[str, int], top_k: int, static_policy: str, args) -> list[tuple[float, float, chess.Move, dict]]:
    scored = []
    for move in board.legal_moves:
        raw = teacher_score(board, move, seen)
        audit = tactical_audit(board, move)
        adjusted = raw if static_policy == "static_alpha_lite" else raw - safety_penalty(audit, args)
        scored.append((adjusted, raw, move, audit))
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[: min(top_k, len(scored))]


def adaptive_candidates(board: chess.Board, seen: dict[str, int], top_k: int, args) -> list[tuple[float, float, chess.Move, dict]]:
    scored = []
    for move in board.legal_moves:
        raw = teacher_score(board, move, seen)
        audit = tactical_audit(board, move)
        adjusted = raw - args.adaptive_safety_weight * safety_penalty(audit, args)
        scored.append((adjusted, raw, move, audit))
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[: min(top_k, len(scored))]


@torch.no_grad()
def choose_static(board: chess.Board, seen: dict[str, int], top_k: int, static_policy: str, args) -> tuple[chess.Move, dict]:
    adjusted, raw, move, audit = static_candidates(board, seen, top_k, static_policy, args)[0]
    return move, {
        "heuristic": round(raw, 4),
        "value": None,
        "mixed": round(adjusted, 4),
        "static_policy": static_policy,
        "static_safety_penalty": round(raw - adjusted, 4),
        "audit": audit,
    }


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
    args,
) -> tuple[chess.Move, dict]:
    candidates = adaptive_candidates(board, seen, top_k, args)
    if rng.random() < exploration and len(candidates) > 1:
        adjusted, score, move, _audit = rng.choice(candidates[: min(4, len(candidates))])
        return move, {
            "heuristic": round(score, 4),
            "value": None,
            "mixed": round(adjusted, 4),
            "explore": True,
            "adaptive_safety_penalty": round(score - adjusted, 4),
        }
    if value_weight <= 0:
        safe_score, score, move, _audit = candidates[0]
        return move, {
            "heuristic": round(score, 4),
            "value": None,
            "mixed": round(safe_score, 4),
            "decision": "safety_cold_start",
            "adaptive_safety_penalty": round(score - safe_score, 4),
        }
    feats = torch.stack([feature(board, move, score) for _adjusted, score, move, _audit in candidates]).to(dev)
    values = model(feats).detach().cpu().tolist()
    if decision_mode == "veto":
        best_score, best_raw, best_move, _best_audit = candidates[0]
        best_value = float(values[0])
        eligible = []
        for (score, raw_score, move, _audit), value in zip(candidates, values):
            if best_score - score <= max_heuristic_drop:
                eligible.append((float(value), score, raw_score, move))
        eligible.sort(key=lambda item: item[0], reverse=True)
        value, score, raw_score, move = eligible[0]
        if move != best_move and value - best_value >= value_margin:
            return move, {
                "heuristic": round(raw_score, 4),
                "value": round(value, 4),
                "mixed": round(value, 4),
                "decision": "value_veto",
                "adaptive_safety_penalty": round(raw_score - score, 4),
            }
        return best_move, {
            "heuristic": round(best_raw, 4),
            "value": round(best_value, 4),
            "mixed": round(best_value, 4),
            "decision": "heuristic_kept",
            "adaptive_safety_penalty": round(best_raw - best_score, 4),
        }
    mixed = []
    best_h = candidates[0][0]
    for (adjusted, score, move, _audit), value in zip(candidates, values):
        h = math.tanh((adjusted - best_h) / 6.0)
        mixed.append((h + value_weight * float(value), score, adjusted, float(value), move))
    mixed.sort(key=lambda item: item[0], reverse=True)
    total, score, adjusted, value, move = mixed[0]
    return move, {
        "heuristic": round(score, 4),
        "value": round(value, 4),
        "mixed": round(total, 4),
        "adaptive_safety_penalty": round(score - adjusted, 4),
    }


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
    static_policy: str,
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
                args,
            )
            policy = "adaptive_world_model"
        else:
            move, info = choose_static(board, seen, args.top_k, static_policy, args)
            policy = static_policy
        san = board.san(move)
        audit = info.pop("audit", None) or tactical_audit(board, move)
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
    winner = "adaptive_world_model" if adaptive_value > 0.2 else (static_policy if adaptive_value < -0.2 else "draw")
    return {
        "adaptive_color": "white" if adaptive_color == chess.WHITE else "black",
        "opponent_policy": static_policy,
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
            played = chess.Move.from_uci(sample.move_uci)
            candidates = adaptive_candidates(board, {}, args.train_candidate_top_k, args)
            if all(move != played for _adjusted, _score, move, _audit in candidates) and played in board.legal_moves:
                audit = tactical_audit(board, played)
                h = teacher_score(board, played, {})
                adjusted = h - args.adaptive_safety_weight * safety_penalty(audit, args)
                candidates.append((adjusted, h, played, audit))
            for _adjusted, h, move, audit in candidates:
                after = board.copy(stack=False)
                before_material = material_for_turn(board, board.turn)
                after.push(move)
                material_delta = material_for_turn(after, board.turn) - before_material
                target = args.counterfactual_material_weight * max(-1.0, min(1.0, material_delta / 9.0))
                target -= args.counterfactual_safety_weight * safety_penalty(audit, args) / 12.0
                if after.is_checkmate():
                    target = 1.0
                if move == played:
                    target += args.played_outcome_weight * sample.final_value
                    target += args.played_material_weight * max(-1.0, min(1.0, sample.material_after / 10.0))
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
    losses = sum(1 for game in games if game["winner"] not in {"adaptive_world_model", "draw"})
    draws = len(games) - wins - losses
    values = [game["adaptive_value"] for game in games]
    audit = summarize_tactical_audit(games)
    opponent_winners: dict[str, int] = {}
    for game in games:
        if game["winner"] in {"adaptive_world_model", "draw"}:
            continue
        opponent_winners[game["winner"]] = opponent_winners.get(game["winner"], 0) + 1
    return {
        "games": len(games),
        "adaptive_wins": wins,
        "static_wins": losses,
        "opponent_wins": losses,
        "opponent_winners": opponent_winners,
        "draws": draws,
        "adaptive_score_rate": round((wins + 0.5 * draws) / max(len(games), 1), 4),
        "avg_adaptive_value": round(sum(values) / max(len(values), 1), 4),
        "avg_plies": round(sum(game["plies"] for game in games) / max(len(games), 1), 2),
        "tactical_audit": audit,
    }


def empty_audit_totals() -> dict:
    return {"moves": 0, "queen_hangs": 0, "major_piece_hangs": 0, "tactical_blunders": 0, "best_capture_sum": 0.0}


def summarize_move_audit(moves: list[dict]) -> dict:
    totals = {"adaptive_world_model": empty_audit_totals()}
    for move in moves:
        policy = move.get("policy")
        if not policy:
            continue
        totals.setdefault(policy, empty_audit_totals())
        audit = move.get("audit") or {}
        totals[policy]["moves"] += 1
        totals[policy]["queen_hangs"] += int(bool(audit.get("queen_hang")))
        totals[policy]["major_piece_hangs"] += int(bool(audit.get("major_piece_hang")))
        totals[policy]["tactical_blunders"] += int(bool(audit.get("tactical_blunder")))
        totals[policy]["best_capture_sum"] += float(audit.get("opponent_best_capture") or 0.0)
    return finish_audit_totals(totals)


def summarize_tactical_audit(games: list[dict]) -> dict:
    totals = {"adaptive_world_model": empty_audit_totals()}
    for game in games:
        summary = game.get("audit_summary") or {}
        for policy, stats in summary.items():
            totals.setdefault(policy, empty_audit_totals())
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


def evaluate_fixed(model: OnlineValueModel, iteration: int, args, dev: str, value_weight: float, static_policy: str) -> dict:
    rng = random.Random(args.seed + 700_000 + iteration)
    games = []
    for game_idx in range(args.eval_games):
        start = warmup_board(args.seed + 800_000 + game_idx // 2, args.opening_plies)
        adaptive_color = chess.WHITE if game_idx % 2 == 0 else chess.BLACK
        game, _ = play_match_game(
            model,
            adaptive_color,
            start,
            args,
            dev,
            rng,
            exploration=0.0,
            value_weight=value_weight,
            static_policy=static_policy,
        )
        games.append(game)
    return {
        "value_weight": round(value_weight, 4),
        "opponent_policy": static_policy,
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
        game, samples = play_match_game(
            model,
            adaptive_color,
            start,
            args,
            dev,
            rng,
            exploration,
            effective_value_weight,
            args.static_policy,
        )
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
        args.static_policy,
    )
    heldout_vs_alpha = evaluate_fixed(model, iteration, args, dev, args.value_weight, "static_alpha_lite")
    heldout_vs_safety = evaluate_fixed(model, iteration, args, dev, args.value_weight, "static_safety_alpha")
    return {
        "iteration": iteration,
        "exploration": round(exploration, 4),
        "effective_value_weight": round(effective_value_weight, 4),
        "new_samples": len(new_samples),
        "replay_samples": len(replay),
        "summary": summarize(games),
        "online_update": update,
        "heldout_after_update": evaluation,
        "heldout_vs_alpha": heldout_vs_alpha,
        "heldout_vs_safety": heldout_vs_safety,
        "sample_games": games[: args.sample_games],
    }


def best_iteration_by(iterations: list[dict], key: str) -> dict:
    if not iterations:
        return {}
    return max(
        iterations,
        key=lambda row: row[key]["summary"]["adaptive_score_rate"],
    )


def audit_score(summary: dict) -> float:
    audit = (summary.get("tactical_audit") or {}).get("adaptive_world_model") or {}
    return (
        2.0 * float(audit.get("tactical_blunder_rate") or 0.0)
        + float(audit.get("queen_hang_rate") or 0.0)
        + 0.75 * float(audit.get("major_piece_hang_rate") or 0.0)
        + 0.1 * float(audit.get("avg_opponent_best_capture") or 0.0)
    )


def checkpoint_kpi(heldout_vs_safety: dict | None) -> dict:
    summary = (heldout_vs_safety or {}).get("summary") or {}
    value = float(summary.get("adaptive_score_rate") or 0.0)
    return {
        "namespace": "12_chess_experts/chess_online_world_model",
        "name": "heldout_vs_static_safety_alpha_score",
        "value": round(max(0.0, min(1.0, value)), 6),
        "range": [0.0, 1.0],
        "higher_is_better": True,
        "source": "champion.heldout_vs_safety.summary.adaptive_score_rate",
    }


def promote_candidate(candidate_eval: dict, champion_eval: dict | None, args) -> tuple[bool, str]:
    if champion_eval is None:
        return True, "first_candidate"
    candidate_summary = candidate_eval["summary"]
    champion_summary = champion_eval["summary"]
    candidate_score = float(candidate_summary["adaptive_score_rate"])
    champion_score = float(champion_summary["adaptive_score_rate"])
    if candidate_score > champion_score + args.promotion_margin:
        return True, "heldout_score_improved"
    if candidate_score + args.promotion_margin < champion_score:
        return False, "heldout_score_regressed"
    candidate_audit = audit_score(candidate_summary)
    champion_audit = audit_score(champion_summary)
    if candidate_audit <= champion_audit * (1.0 + args.audit_tolerance):
        return True, "heldout_score_tied_audit_ok"
    return False, "audit_regressed"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=701)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--games-per-iteration", type=int, default=6)
    parser.add_argument("--eval-games", type=int, default=8)
    parser.add_argument("--max-plies", type=int, default=60)
    parser.add_argument("--opening-plies", type=int, default=6)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--static-policy", choices=("static_alpha_lite", "static_safety_alpha"), default="static_safety_alpha")
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--online-epochs", type=int, default=6)
    parser.add_argument("--train-candidate-top-k", type=int, default=4)
    parser.add_argument("--played-outcome-weight", type=float, default=0.55)
    parser.add_argument("--played-material-weight", type=float, default=0.2)
    parser.add_argument("--counterfactual-material-weight", type=float, default=0.2)
    parser.add_argument("--counterfactual-safety-weight", type=float, default=0.9)
    parser.add_argument("--value-weight", type=float, default=0.12)
    parser.add_argument("--decision-mode", choices=("blend", "veto"), default="blend")
    parser.add_argument("--max-heuristic-drop", type=float, default=3.0)
    parser.add_argument("--value-margin", type=float, default=0.12)
    parser.add_argument("--blunder-penalty", type=float, default=0.35)
    parser.add_argument("--queen-hang-penalty", type=float, default=0.25)
    parser.add_argument("--static-blunder-penalty", type=float, default=6.0)
    parser.add_argument("--static-queen-hang-penalty", type=float, default=8.0)
    parser.add_argument("--static-major-hang-penalty", type=float, default=1.2)
    parser.add_argument("--static-best-capture-penalty", type=float, default=0.35)
    parser.add_argument("--static-reply-delta-penalty", type=float, default=0.25)
    parser.add_argument("--adaptive-safety-weight", type=float, default=1.0)
    parser.add_argument("--exploration", type=float, default=0.25)
    parser.add_argument("--exploration-decay", type=float, default=0.72)
    parser.add_argument("--min-exploration", type=float, default=0.03)
    parser.add_argument("--replay-limit", type=int, default=6000)
    parser.add_argument("--promotion-mode", choices=("latest", "champion"), default="champion")
    parser.add_argument("--promotion-margin", type=float, default=0.0)
    parser.add_argument("--audit-tolerance", type=float, default=0.05)
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
    champion_state = deepcopy(model.state_dict())
    champion_eval = None
    champion_alpha_eval = None
    champion_iteration = None
    champion_promotions = 0
    iterations = []
    for i in range(args.iterations):
        iteration = run_iteration(i, model, opt, replay, args, dev)
        candidate_eval = iteration["heldout_vs_safety"]
        promoted = args.promotion_mode == "latest"
        reason = "latest_mode"
        if args.promotion_mode == "champion":
            promoted, reason = promote_candidate(candidate_eval, champion_eval, args)
        if promoted:
            champion_state = deepcopy(model.state_dict())
            champion_eval = candidate_eval
            champion_alpha_eval = iteration["heldout_vs_alpha"]
            champion_iteration = i
            champion_promotions += 1
        else:
            model.load_state_dict(champion_state)
            opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        iteration["promotion"] = {
            "mode": args.promotion_mode,
            "promoted": promoted,
            "reason": reason,
            "candidate_score": candidate_eval["summary"]["adaptive_score_rate"],
            "candidate_audit_score": round(audit_score(candidate_eval["summary"]), 6),
            "champion_iteration": champion_iteration,
            "champion_promotions": champion_promotions,
            "champion_score": champion_eval["summary"]["adaptive_score_rate"] if champion_eval else None,
            "champion_audit_score": round(audit_score(champion_eval["summary"]), 6) if champion_eval else None,
        }
        if not promoted:
            iteration["rejected_candidate_state_restored"] = True
        iterations.append(iteration)
    model.load_state_dict(champion_state)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    champion_checkpoint = {
        "format": "chess_online_world_model_champion/v1",
        "state_dict": champion_state,
        "config": vars(args),
        "kpi": checkpoint_kpi(champion_eval),
        "feature_dim": FEATURE_DIM,
        "model_width": args.width,
        "champion_iteration": champion_iteration,
        "champion_promotions": champion_promotions,
        "heldout_vs_safety": champion_eval or {},
        "heldout_vs_alpha": champion_alpha_eval or {},
    }
    champion_path = CHECKPOINT_DIR / "online_champion_value.pt"
    torch.save(champion_checkpoint, champion_path)
    best_after_update = best_iteration_by(iterations, "heldout_after_update")
    best_vs_alpha = best_iteration_by(iterations, "heldout_vs_alpha")
    best_vs_safety = best_iteration_by(iterations, "heldout_vs_safety")
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
        "champion": {
            "promotion_mode": args.promotion_mode,
            "iteration": champion_iteration,
            "promotions": champion_promotions,
            "kpi": checkpoint_kpi(champion_eval),
            "checkpoint": str(champion_path.relative_to(HERE)),
            "heldout_vs_safety": champion_eval or {},
            "heldout_vs_alpha": champion_alpha_eval or {},
        },
        "best_heldout_after_update": best_after_update.get("heldout_after_update", {}),
        "best_heldout_vs_alpha": best_vs_alpha.get("heldout_vs_alpha", {}),
        "best_heldout_vs_safety": best_vs_safety.get("heldout_vs_safety", {}),
        "best_iterations": {
            "heldout_after_update": best_after_update.get("iteration"),
            "heldout_vs_alpha": best_vs_alpha.get("iteration"),
            "heldout_vs_safety": best_vs_safety.get("iteration"),
        },
        "iterations": iterations,
    }
    ARTIFACT.parent.mkdir(exist_ok=True)
    ARTIFACT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
