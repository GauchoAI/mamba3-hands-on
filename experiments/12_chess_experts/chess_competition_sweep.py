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
from chess_policy_arena import JepaPolicy, jepa_policy_move


HERE = Path(__file__).resolve().parent
ARTIFACT = HERE / "artifacts" / "chess_competition_sweep_result.json"


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
    return model.encoder, jepa.evaluate(model, x_val, move_val, y_val, val_fens, dev)


def cases_by_family(cases: list[motif.ChessCase]) -> dict[str, list[motif.ChessCase]]:
    out = {name: [] for name in motif.MOTIFS}
    for case in cases:
        out[case.motif].append(case)
    return out


def take_budget(pool: list[motif.ChessCase], budget: int) -> list[motif.ChessCase]:
    grouped = cases_by_family(pool)
    selected = []
    for name in motif.MOTIFS:
        selected.extend(grouped[name][:budget])
    return selected


def train_motif_policy(train_cases: list[motif.ChessCase], args, dev: str) -> motif.MateMLP:
    x_train, y_train = motif.batch(train_cases, dev)
    model = motif.MateMLP(x_train.shape[-1], args.motif_width).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    for _ in range(args.policy_epochs):
        loss = F.cross_entropy(model(x_train), y_train)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    model.eval()
    return model


def train_jepa_policy(encoder: jepa.Encoder, train_cases: list[motif.ChessCase], args, dev: str) -> JepaPolicy:
    x_train = torch.stack([jepa.board_to_tensor(chess.Board(case.fen)) for case in train_cases]).to(dev)
    y_train = torch.tensor(
        [motif.move_class(chess.Move.from_uci(case.move_uci)) for case in train_cases],
        dtype=torch.long,
        device=dev,
    )
    policy = JepaPolicy(encoder, args.latent_dim, args.policy_width).to(dev)
    for p in policy.encoder.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW((p for p in policy.parameters() if p.requires_grad), lr=args.lr, weight_decay=1e-4)
    for _ in range(args.policy_epochs):
        loss = F.cross_entropy(policy(x_train), y_train)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    policy.eval()
    return policy


def is_mate_after(board: chess.Board, move: chess.Move) -> bool:
    after = board.copy(stack=False)
    after.push(move)
    return after.is_checkmate()


def evaluate_policies(motif_model: motif.MateMLP, jepa_model: JepaPolicy, val_cases: list[motif.ChessCase], dev: str) -> dict:
    motif_mates = 0
    jepa_mates = 0
    motif_wins = 0
    jepa_wins = 0
    ties_both = 0
    ties_fail = 0
    by_family: dict[str, dict[str, int]] = {
        name: {"n": 0, "motif": 0, "jepa": 0, "motif_wins": 0, "jepa_wins": 0}
        for name in motif.MOTIFS
    }
    examples = []
    for case in val_cases:
        board = chess.Board(case.fen)
        motif_move = motif.legal_masked_prediction(motif_model, board, dev)
        jepa_move = jepa_policy_move(jepa_model, board, dev)
        motif_ok = is_mate_after(board, motif_move)
        jepa_ok = is_mate_after(board, jepa_move)
        motif_mates += int(motif_ok)
        jepa_mates += int(jepa_ok)
        fam = by_family[case.motif]
        fam["n"] += 1
        fam["motif"] += int(motif_ok)
        fam["jepa"] += int(jepa_ok)
        if motif_ok and not jepa_ok:
            motif_wins += 1
            fam["motif_wins"] += 1
            outcome = "motif_win"
        elif jepa_ok and not motif_ok:
            jepa_wins += 1
            fam["jepa_wins"] += 1
            outcome = "jepa_win"
        elif motif_ok and jepa_ok:
            ties_both += 1
            outcome = "tie_both_mate"
        else:
            ties_fail += 1
            outcome = "tie_both_fail"
        if outcome != "tie_both_mate" and len(examples) < 10:
            examples.append({
                "motif": case.motif,
                "fen": case.fen,
                "expected": case.move_uci,
                "all_mates": list(case.mate_moves),
                "motif_move": motif_move.uci(),
                "motif_mate": motif_ok,
                "jepa_move": jepa_move.uci(),
                "jepa_mate": jepa_ok,
                "outcome": outcome,
            })
    n = len(val_cases)
    return {
        "positions": n,
        "motif_mate_rate": round(motif_mates / n, 4),
        "jepa_mate_rate": round(jepa_mates / n, 4),
        "motif_wins": motif_wins,
        "jepa_wins": jepa_wins,
        "ties_both_mate": ties_both,
        "ties_both_fail": ties_fail,
        "by_family": {
            name: {
                "n": stats["n"],
                "motif_mate_rate": round(stats["motif"] / max(stats["n"], 1), 4),
                "jepa_mate_rate": round(stats["jepa"] / max(stats["n"], 1), 4),
                "motif_wins": stats["motif_wins"],
                "jepa_wins": stats["jepa_wins"],
            }
            for name, stats in by_family.items()
        },
        "decisive_examples": examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=67)
    parser.add_argument("--budgets", default="8,16,32,64,128,256")
    parser.add_argument("--max-train-per-family", type=int, default=256)
    parser.add_argument("--val-per-family", type=int, default=48)
    parser.add_argument("--motif-width", type=int, default=768)
    parser.add_argument("--jepa-pairs", type=int, default=1800)
    parser.add_argument("--jepa-val-pairs", type=int, default=300)
    parser.add_argument("--jepa-width", type=int, default=256)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--jepa-epochs", type=int, default=14)
    parser.add_argument("--policy-width", type=int, default=512)
    parser.add_argument("--policy-epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=2e-3)
    args = parser.parse_args()

    t0 = time.time()
    dev = motif.device()
    train_pool, val_cases = motif.generate_cases(args.max_train_per_family, args.val_per_family, args.seed)
    encoder, bridge_metrics = train_jepa_encoder(args, dev)
    rows = []
    for budget in [int(part) for part in args.budgets.split(",") if part.strip()]:
        train_cases = take_budget(train_pool, budget)
        motif_model = train_motif_policy(train_cases, args, dev)
        jepa_model = train_jepa_policy(encoder, train_cases, args, dev)
        result = evaluate_policies(motif_model, jepa_model, val_cases, dev)
        result["train_per_family"] = budget
        result["train_cases"] = len(train_cases)
        rows.append(result)

    payload = {
        "status": "sample_efficiency_competition_sweep",
        "scope": "held-out mate-in-one tactical policy competition under increasing train budgets",
        "device": dev,
        "elapsed_s": round(time.time() - t0, 3),
        "jepa_bridge_pretrain": {
            "cosine_mean": round(float(bridge_metrics["cosine_mean"]), 6),
            "nearest_neighbor_top1": round(float(bridge_metrics["nearest_neighbor_top1"]), 4),
            "nearest_neighbor_top5": round(float(bridge_metrics["nearest_neighbor_top5"]), 4),
        },
        "rows": rows,
    }
    ARTIFACT.parent.mkdir(exist_ok=True)
    ARTIFACT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
