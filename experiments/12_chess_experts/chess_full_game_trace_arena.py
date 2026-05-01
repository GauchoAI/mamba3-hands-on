from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

import chess
import torch
import torch.nn.functional as F

import chess_jepa_bridge as jepa
import chess_motif_generalization as motif
import chess_puzzle_sequence_arena as sequence
from chess_competition_sweep import train_jepa_encoder
from chess_full_game_arena import Player, material_score, play_game, warmup_board
from chess_policy_arena import JepaPolicy


HERE = Path(__file__).resolve().parent
ARTIFACT = HERE / "artifacts" / "chess_full_game_trace_arena_result.json"


@dataclass(frozen=True)
class TraceCase:
    fen: str
    move_uci: str
    phase: str
    ply: int


def mobility(board: chess.Board) -> int:
    return board.legal_moves.count()


def king_safety(board: chess.Board, color: chess.Color) -> float:
    king = board.king(color)
    if king is None:
        return -8.0
    attackers = board.attackers(not color, king)
    score = -1.5 * len(attackers)
    file = chess.square_file(king)
    rank = chess.square_rank(king)
    home_rank = 0 if color == chess.WHITE else 7
    if rank == home_rank and file in (0, 1, 2, 6, 7):
        score += 0.4
    return score


def phase(board: chess.Board) -> str:
    plies = (board.fullmove_number - 1) * 2 + (0 if board.turn == chess.WHITE else 1)
    queens = sum(1 for piece in board.piece_map().values() if piece.piece_type == chess.QUEEN)
    pieces = len(board.piece_map())
    if plies < 16 and queens >= 2:
        return "opening"
    if pieces <= 12 or queens == 0:
        return "endgame"
    return "middlegame"


def teacher_score(board: chess.Board, move: chess.Move, seen: dict[str, int]) -> float:
    before_material = material_score(board)
    before_mobility = mobility(board)
    after = board.copy(stack=False)
    after.push(move)
    if after.is_checkmate():
        return 1_000.0
    if after.is_stalemate() or after.can_claim_draw():
        draw_penalty = -1.2
    else:
        draw_penalty = 0.0

    material_delta = material_score(after) - before_material
    if board.turn == chess.BLACK:
        material_delta = -material_delta
    mobility_delta = mobility(after) - before_mobility
    if board.turn == chess.BLACK:
        mobility_delta = -mobility_delta
    score = 2.0 * material_delta
    score += 0.05 * mobility_delta
    score += 0.8 if board.gives_check(move) else 0.0
    score += 0.7 if board.is_castling(move) else 0.0
    score += 0.8 if move.promotion is not None else 0.0
    score += 0.12 * (3.5 - abs(chess.square_file(move.to_square) - 3.5))
    score += 0.12 * (3.5 - abs(chess.square_rank(move.to_square) - 3.5))
    score += king_safety(after, board.turn) - king_safety(board, board.turn)
    score -= 1.5 * seen.get(" ".join(after.fen().split(" ")[:4]), 0)
    score += draw_penalty
    return score


def teacher_move(board: chess.Board, seen: dict[str, int], rng: random.Random, temperature: float) -> chess.Move:
    scored = [(teacher_score(board, move, seen), move) for move in board.legal_moves]
    scored.sort(key=lambda item: item[0], reverse=True)
    if temperature <= 0:
        return scored[0][1]
    top = scored[: min(6, len(scored))]
    best = top[0][0]
    weights = [pow(2.718281828, (score - best) / temperature) for score, _ in top]
    total = sum(weights)
    pick = rng.random() * total
    running = 0.0
    for weight, (_, move) in zip(weights, top):
        running += weight
        if running >= pick:
            return move
    return top[0][1]


def trace_key(case: TraceCase) -> tuple[str, str]:
    board = chess.Board(case.fen)
    return board.board_fen() + " " + ("w" if board.turn else "b"), case.move_uci


def dedupe_traces(cases: list[TraceCase]) -> list[TraceCase]:
    out = []
    seen: set[tuple[str, str]] = set()
    for case in cases:
        key = trace_key(case)
        if key in seen:
            continue
        seen.add(key)
        out.append(case)
    return out


def generate_trace_cases(args, max_cases: int | None = None, seed_offset: int = 0) -> list[TraceCase]:
    rng = random.Random(args.seed)
    limit = max_cases or args.max_trace_cases
    cases: list[TraceCase] = []
    seen_cases: set[tuple[str, str]] = set()
    for game_idx in range(args.teacher_games):
        opening_plies = rng.randint(0, args.max_opening_plies) if args.diverse_starts else args.opening_plies
        board = warmup_board(args.seed + seed_offset + game_idx, opening_plies)
        seen = {" ".join(board.fen().split(" ")[:4]): 1}
        for ply in range(args.teacher_max_plies):
            if board.is_game_over(claim_draw=True):
                break
            move = teacher_move(board, seen, rng, args.teacher_temperature)
            key = (board.board_fen() + " " + ("w" if board.turn else "b"), move.uci())
            if key not in seen_cases:
                seen_cases.add(key)
                cases.append(TraceCase(board.fen(), move.uci(), phase(board), ply))
            board.push(move)
            position = " ".join(board.fen().split(" ")[:4])
            seen[position] = seen.get(position, 0) + 1
            if len(cases) >= limit:
                return cases
    return cases


def generate_balanced_trace_cases(args, max_cases: int | None = None, seed_offset: int = 10_000) -> list[TraceCase]:
    rng = random.Random(args.seed + seed_offset)
    limit = max_cases or args.max_trace_cases
    target = limit // 3
    targets = {"opening": target, "middlegame": target, "endgame": limit - 2 * target}
    buckets: dict[str, list[TraceCase]] = {"opening": [], "middlegame": [], "endgame": []}
    seen_cases: set[tuple[str, str]] = set()
    attempts = 0
    max_attempts = args.teacher_games * 6
    while attempts < max_attempts and any(len(buckets[name]) < targets[name] for name in buckets):
        attempts += 1
        opening_plies = rng.randint(0, args.max_opening_plies)
        board = warmup_board(args.seed + seed_offset + attempts, opening_plies)
        seen = {" ".join(board.fen().split(" ")[:4]): 1}
        for ply in range(args.teacher_max_plies):
            if board.is_game_over(claim_draw=True):
                break
            move = teacher_move(board, seen, rng, args.teacher_temperature)
            current_phase = phase(board)
            key = (board.board_fen() + " " + ("w" if board.turn else "b"), move.uci())
            if len(buckets[current_phase]) < targets[current_phase] and key not in seen_cases:
                seen_cases.add(key)
                buckets[current_phase].append(TraceCase(board.fen(), move.uci(), current_phase, ply))
            board.push(move)
            position = " ".join(board.fen().split(" ")[:4])
            seen[position] = seen.get(position, 0) + 1
            if all(len(buckets[name]) >= targets[name] for name in buckets):
                break
    cases = buckets["opening"] + buckets["middlegame"] + buckets["endgame"]
    rng.shuffle(cases)
    return cases


def build_training_traces(args) -> tuple[list[TraceCase], dict]:
    base = generate_trace_cases(args) if args.additive_traces else []
    if args.balanced_traces:
        balanced_cases = args.balanced_trace_cases or args.max_trace_cases
        balanced = generate_balanced_trace_cases(args, balanced_cases)
    else:
        balanced = []
    tactical = generate_tactical_trace_cases(args) if args.include_tactical_traces else []
    multi_ply = generate_sequence_trace_cases(args) if args.include_sequence_traces else []
    if not base and not balanced:
        base = generate_trace_cases(args)
    traces = dedupe_traces(base + balanced + tactical + multi_ply)
    return traces, {
        "base_unbalanced": trace_summary(base),
        "balanced": trace_summary(balanced),
        "tactical_puzzles": trace_summary(tactical),
        "multi_ply_puzzles": trace_summary(multi_ply),
        "combined": trace_summary(traces),
    }


def generate_tactical_trace_cases(args) -> list[TraceCase]:
    train_cases, _ = motif.generate_cases(args.tactical_trace_per_family, args.puzzle_val_per_family, args.seed + 120_000)
    return [
        TraceCase(case.fen, case.move_uci, f"tactical_{case.motif}", 0)
        for case in train_cases
    ]


def generate_sequence_trace_cases(args) -> list[TraceCase]:
    seq_args = argparse.Namespace(
        seed=args.seed + 130_000,
        train_per_family=max(args.puzzle_train_per_family, 384),
        val_per_family=max(24, args.sequence_trace_puzzles // 4),
        puzzles=args.sequence_trace_puzzles,
    )
    puzzles = sequence.build_two_step_puzzles(seq_args)
    traces = []
    for puzzle in puzzles:
        board = chess.Board(puzzle.fen)
        reply = chess.Move.from_uci(puzzle.line[0])
        if reply in board.legal_moves:
            board.push(reply)
            traces.append(TraceCase(board.fen(), puzzle.line[1], f"sequence_{puzzle.motif}", 1))
    return traces


def train_direct_policy(cases: list[TraceCase], args, dev: str) -> motif.MateMLP:
    x_train = torch.stack([motif.board_features(chess.Board(case.fen)) for case in cases]).to(dev)
    y_train = torch.tensor(
        [motif.move_class(chess.Move.from_uci(case.move_uci)) for case in cases],
        dtype=torch.long,
        device=dev,
    )
    model = motif.MateMLP(x_train.shape[-1], args.policy_width).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    for _ in range(args.policy_epochs):
        order = torch.randperm(x_train.shape[0], device=dev)
        for start in range(0, x_train.shape[0], args.batch_size):
            idx = order[start : start + args.batch_size]
            loss = F.cross_entropy(model(x_train[idx]), y_train[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    model.eval()
    return model


def train_jepa_trace_policy(encoder: jepa.Encoder, cases: list[TraceCase], args, dev: str) -> JepaPolicy:
    x_train = torch.stack([jepa.board_to_tensor(chess.Board(case.fen)) for case in cases]).to(dev)
    y_train = torch.tensor(
        [motif.move_class(chess.Move.from_uci(case.move_uci)) for case in cases],
        dtype=torch.long,
        device=dev,
    )
    policy = JepaPolicy(encoder, args.latent_dim, args.policy_width).to(dev)
    if args.freeze_encoder:
        for param in policy.encoder.parameters():
            param.requires_grad_(False)
    opt = torch.optim.AdamW((p for p in policy.parameters() if p.requires_grad), lr=args.lr, weight_decay=1e-4)
    for _ in range(args.policy_epochs):
        order = torch.randperm(x_train.shape[0], device=dev)
        for start in range(0, x_train.shape[0], args.batch_size):
            idx = order[start : start + args.batch_size]
            loss = F.cross_entropy(policy(x_train[idx]), y_train[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    policy.eval()
    return policy


def trace_summary(cases: list[TraceCase]) -> dict:
    phases: dict[str, int] = {}
    for case in cases:
        phases[case.phase] = phases.get(case.phase, 0) + 1
    return {
        "cases": len(cases),
        "phases": phases,
    }


def game_summary(games: list[dict]) -> dict:
    wins = {"motif_full_trace": 0, "jepa_full_trace": 0, "draw": 0}
    terminations: dict[str, int] = {}
    plies = 0
    for game in games:
        winner = game["winner"]
        if winner in wins:
            wins[winner] += 1
        else:
            wins["draw"] += 1
        terminations[game["termination"]] = terminations.get(game["termination"], 0) + 1
        plies += game["plies"]
    decisive = wins["motif_full_trace"] + wins["jepa_full_trace"]
    return {
        "games": len(games),
        "motif_full_trace_wins": wins["motif_full_trace"],
        "jepa_full_trace_wins": wins["jepa_full_trace"],
        "draws": wins["draw"],
        "decisive_games": decisive,
        "avg_plies": round(plies / max(len(games), 1), 2),
        "terminations": terminations,
    }


@torch.no_grad()
def legal_policy_move(player: Player, board: chess.Board, dev: str) -> chess.Move:
    if player.kind == "motif":
        x = motif.board_features(board).unsqueeze(0).to(dev)
        logits = player.model(x)[0].detach().cpu()
    else:
        x = jepa.board_to_tensor(board).unsqueeze(0).to(dev)
        logits = player.model(x)[0].detach().cpu()
    legal = list(board.legal_moves)
    legal.sort(key=lambda move: float(logits[motif.move_class(move)]), reverse=True)
    return legal[0]


def evaluate_trace_imitation(players: list[Player], cases: list[TraceCase], dev: str) -> dict:
    out = {}
    for player in players:
        total = 0
        exact = 0
        by_phase: dict[str, dict[str, int]] = {}
        for case in cases:
            board = chess.Board(case.fen)
            predicted = legal_policy_move(player, board, dev)
            expected = chess.Move.from_uci(case.move_uci)
            total += 1
            exact += int(predicted == expected)
            phase_stats = by_phase.setdefault(case.phase, {"n": 0, "exact": 0})
            phase_stats["n"] += 1
            phase_stats["exact"] += int(predicted == expected)
        out[player.name] = {
            "cases": total,
            "exact_move_rate": round(exact / max(total, 1), 4),
            "by_phase": {
                name: {
                    "n": stats["n"],
                    "exact_move_rate": round(stats["exact"] / max(stats["n"], 1), 4),
                }
                for name, stats in by_phase.items()
            },
        }
    return out


def is_mate_after(board: chess.Board, move: chess.Move) -> bool:
    after = board.copy(stack=False)
    after.push(move)
    return after.is_checkmate()


def evaluate_tactical_puzzles(players: list[Player], args, dev: str) -> dict:
    _, val_cases = motif.generate_cases(args.puzzle_train_per_family, args.puzzle_val_per_family, args.seed + 90_000)
    out = {}
    for player in players:
        total = 0
        exact = 0
        mate = 0
        by_family: dict[str, dict[str, int]] = {}
        for case in val_cases:
            board = chess.Board(case.fen)
            predicted = legal_policy_move(player, board, dev)
            expected = chess.Move.from_uci(case.move_uci)
            predicted_mate = is_mate_after(board, predicted)
            total += 1
            exact += int(predicted == expected)
            mate += int(predicted_mate)
            stats = by_family.setdefault(case.motif, {"n": 0, "exact": 0, "mate": 0})
            stats["n"] += 1
            stats["exact"] += int(predicted == expected)
            stats["mate"] += int(predicted_mate)
        out[player.name] = {
            "cases": total,
            "exact_move_rate": round(exact / max(total, 1), 4),
            "legal_mate_rate": round(mate / max(total, 1), 4),
            "by_family": {
                name: {
                    "n": stats["n"],
                    "exact_move_rate": round(stats["exact"] / max(stats["n"], 1), 4),
                    "legal_mate_rate": round(stats["mate"] / max(stats["n"], 1), 4),
                }
                for name, stats in by_family.items()
            },
        }
    return out


def evaluate_sequence_puzzles(players: list[Player], args, dev: str) -> dict:
    seq_args = argparse.Namespace(
        seed=args.seed + 140_000,
        train_per_family=max(args.puzzle_train_per_family, 384),
        val_per_family=max(24, args.sequence_val_puzzles // 4),
        puzzles=args.sequence_val_puzzles,
    )
    puzzles = sequence.build_two_step_puzzles(seq_args)
    out = {}
    for player in players:
        solved = 0
        by_family: dict[str, dict[str, int]] = {}
        for puzzle in puzzles:
            board = chess.Board(puzzle.fen)
            reply = chess.Move.from_uci(puzzle.line[0])
            if reply not in board.legal_moves:
                ok = False
            else:
                board.push(reply)
                predicted = legal_policy_move(player, board, dev)
                board.push(predicted)
                ok = board.is_checkmate()
            solved += int(ok)
            stats = by_family.setdefault(puzzle.motif, {"n": 0, "solved": 0})
            stats["n"] += 1
            stats["solved"] += int(ok)
        out[player.name] = {
            "puzzles": len(puzzles),
            "solve_rate": round(solved / max(len(puzzles), 1), 4),
            "by_family": {
                name: {
                    "n": stats["n"],
                    "solve_rate": round(stats["solved"] / max(stats["n"], 1), 4),
                }
                for name, stats in by_family.items()
            },
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=211)
    parser.add_argument("--teacher-games", type=int, default=180)
    parser.add_argument("--teacher-max-plies", type=int, default=120)
    parser.add_argument("--max-trace-cases", type=int, default=9000)
    parser.add_argument("--teacher-temperature", type=float, default=0.08)
    parser.add_argument("--balanced-traces", action="store_true")
    parser.add_argument("--additive-traces", action="store_true")
    parser.add_argument("--balanced-trace-cases", type=int, default=0)
    parser.add_argument("--diverse-starts", action="store_true")
    parser.add_argument("--max-opening-plies", type=int, default=18)
    parser.add_argument("--val-trace-cases", type=int, default=1500)
    parser.add_argument("--puzzle-train-per-family", type=int, default=256)
    parser.add_argument("--puzzle-val-per-family", type=int, default=64)
    parser.add_argument("--include-tactical-traces", action="store_true")
    parser.add_argument("--tactical-trace-per-family", type=int, default=1024)
    parser.add_argument("--include-sequence-traces", action="store_true")
    parser.add_argument("--sequence-trace-puzzles", type=int, default=512)
    parser.add_argument("--sequence-val-puzzles", type=int, default=96)
    parser.add_argument("--games", type=int, default=32)
    parser.add_argument("--max-plies", type=int, default=240)
    parser.add_argument("--opening-plies", type=int, default=4)
    parser.add_argument("--jepa-pairs", type=int, default=12000)
    parser.add_argument("--jepa-val-pairs", type=int, default=1200)
    parser.add_argument("--jepa-width", type=int, default=384)
    parser.add_argument("--latent-dim", type=int, default=96)
    parser.add_argument("--jepa-epochs", type=int, default=32)
    parser.add_argument("--policy-width", type=int, default=768)
    parser.add_argument("--policy-epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--blend", type=float, default=0.86)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--anti-repetition", type=float, default=2.5)
    args = parser.parse_args()

    t0 = time.time()
    dev = motif.device()
    traces, training_trace_summary = build_training_traces(args)
    val_traces = generate_balanced_trace_cases(args, args.val_trace_cases, seed_offset=80_000)
    direct = Player("motif_full_trace", train_direct_policy(traces, args, dev), "motif")
    encoder, bridge_metrics = train_jepa_encoder(args, dev, args.seed)
    jepa_player = Player("jepa_full_trace", train_jepa_trace_policy(encoder, traces, args, dev), "jepa")
    players = [direct, jepa_player]

    games = []
    for game_idx in range(args.games):
        pair_idx = game_idx // 2
        start = warmup_board(args.seed + 50_000 + pair_idx, args.opening_plies)
        if game_idx % 2 == 0:
            white, black = direct, jepa_player
        else:
            white, black = jepa_player, direct
        games.append(
            play_game(
                white,
                black,
                start,
                args.seed + 70_000 + game_idx,
                args.max_plies,
                dev,
                args.blend,
                args.temperature,
                args.anti_repetition,
            )
        )

    payload = {
        "status": "full_game_trace_arena",
        "scope": "full legal games after training both policies on generated full-game state/action traces",
        "device": dev,
        "elapsed_s": round(time.time() - t0, 3),
        "config": vars(args),
        "trace_summary": training_trace_summary,
        "metrics": {
            "heldout_trace_imitation": evaluate_trace_imitation(players, val_traces, dev),
            "heldout_tactical_puzzles": evaluate_tactical_puzzles(players, args, dev),
            "heldout_multi_ply_puzzles": evaluate_sequence_puzzles(players, args, dev),
        },
        "jepa_bridge_pretrain": {
            "cosine_mean": round(float(bridge_metrics["cosine_mean"]), 6),
            "nearest_neighbor_top1": round(float(bridge_metrics["nearest_neighbor_top1"]), 4),
            "nearest_neighbor_top5": round(float(bridge_metrics["nearest_neighbor_top5"]), 4),
        },
        "summary": game_summary(games),
        "games": games,
    }
    ARTIFACT.parent.mkdir(exist_ok=True)
    ARTIFACT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
