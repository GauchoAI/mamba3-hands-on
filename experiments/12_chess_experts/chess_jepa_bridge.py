from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import chess
import torch
import torch.nn as nn
import torch.nn.functional as F


HERE = Path(__file__).resolve().parent
DEFAULT_ARTIFACT = HERE / "artifacts" / "chess_jepa_bridge_result.json"
PIECE_TO_PLANE = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
    chess.KING: 5,
}
BOARD_DIM = 12 * 64 + 1 + 4 + 8 + 2
MOVE_DIM = 64 + 64 + 5 + 6 + 6


@dataclass(frozen=True)
class Pair:
    fen_before: str
    move_uci: str
    fen_after: str


@dataclass(frozen=True)
class Config:
    seed: int = 7
    pairs: int = 3600
    val_pairs: int = 720
    epochs: int = 18
    batch_size: int = 256
    width: int = 256
    latent_dim: int = 64
    lr: float = 1.5e-3
    weight_decay: float = 1.0e-4
    artifact: str = str(DEFAULT_ARTIFACT)


class Encoder(nn.Module):
    def __init__(self, width: int, latent_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(BOARD_DIM, width),
            nn.GELU(),
            nn.LayerNorm(width),
            nn.Linear(width, width),
            nn.GELU(),
            nn.LayerNorm(width),
            nn.Linear(width, latent_dim),
        )

    def forward(self, board_features: torch.Tensor) -> torch.Tensor:
        return self.net(board_features)


class Predictor(nn.Module):
    def __init__(self, latent_dim: int, move_dim: int, width: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + move_dim, width),
            nn.GELU(),
            nn.LayerNorm(width),
            nn.Linear(width, width),
            nn.GELU(),
            nn.LayerNorm(width),
            nn.Linear(width, latent_dim),
        )

    def forward(self, z_t: torch.Tensor, move_features: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z_t, move_features], dim=-1))


class ChessJEPA(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.encoder = Encoder(cfg.width, cfg.latent_dim)
        self.predictor = Predictor(cfg.latent_dim, MOVE_DIM, cfg.width)

    def forward(self, x_t: torch.Tensor, move_x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z_t = self.encoder(x_t)
        pred_next = self.predictor(z_t, move_x)
        return z_t, pred_next, F.normalize(pred_next, dim=-1)


def choose_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def board_to_tensor(board: chess.Board) -> torch.Tensor:
    x = torch.zeros(BOARD_DIM, dtype=torch.float32)
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
    if board.ep_square is not None:
        x[base + 5 + chess.square_file(board.ep_square)] = 1.0
    x[base + 13] = min(board.halfmove_clock, 100) / 100.0
    x[base + 14] = min(board.fullmove_number, 120) / 120.0
    return x


def move_to_tensor(board: chess.Board, move: chess.Move) -> torch.Tensor:
    x = torch.zeros(MOVE_DIM, dtype=torch.float32)
    x[move.from_square] = 1.0
    x[64 + move.to_square] = 1.0

    base = 128
    promo_map = {None: 0, chess.KNIGHT: 1, chess.BISHOP: 2, chess.ROOK: 3, chess.QUEEN: 4}
    x[base + promo_map[move.promotion]] = 1.0

    flags = base + 5
    x[flags] = float(board.is_capture(move))
    x[flags + 1] = float(board.is_castling(move))
    x[flags + 2] = float(board.is_en_passant(move))
    x[flags + 3] = float(board.gives_check(move))
    x[flags + 4] = float(board.is_kingside_castling(move))
    x[flags + 5] = float(board.is_queenside_castling(move))

    piece_base = flags + 6
    piece = board.piece_at(move.from_square)
    if piece is not None:
        x[piece_base + PIECE_TO_PLANE[piece.piece_type]] = 1.0
    return x


def tactical_weight(board: chess.Board, move: chess.Move) -> float:
    score = 1.0
    if board.is_capture(move):
        score += 2.0
    if board.gives_check(move):
        score += 2.5
    if move.promotion is not None:
        score += 3.0
    if board.is_castling(move):
        score += 0.8
    return score


def generate_pairs(n: int, seed: int, max_plies: int = 80) -> list[Pair]:
    rng = random.Random(seed)
    pairs: list[Pair] = []
    seen: set[tuple[str, str]] = set()
    while len(pairs) < n:
        board = chess.Board()
        warmup = rng.randint(0, 10)
        for _ in range(warmup):
            if board.is_game_over(claim_draw=True):
                break
            board.push(rng.choice(list(board.legal_moves)))

        for _ in range(max_plies):
            if board.is_game_over(claim_draw=True):
                break
            legal = list(board.legal_moves)
            weights = [tactical_weight(board, mv) for mv in legal]
            move = rng.choices(legal, weights=weights, k=1)[0]
            before = board.fen()
            board.push(move)
            after = board.fen()
            key = (before, move.uci())
            if key not in seen:
                seen.add(key)
                pairs.append(Pair(before, move.uci(), after))
                if len(pairs) >= n:
                    break
    return pairs


def tensorize(pairs: list[Pair]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    current, moves, nxt, next_fens = [], [], [], []
    for pair in pairs:
        board = chess.Board(pair.fen_before)
        move = chess.Move.from_uci(pair.move_uci)
        current.append(board_to_tensor(board))
        moves.append(move_to_tensor(board, move))
        next_board = chess.Board(pair.fen_after)
        nxt.append(board_to_tensor(next_board))
        next_fens.append(next_board.board_fen() + " " + ("w" if next_board.turn else "b"))
    return torch.stack(current), torch.stack(moves), torch.stack(nxt), next_fens


def variance_loss(z: torch.Tensor) -> torch.Tensor:
    std = torch.sqrt(z.var(dim=0) + 1e-4)
    return torch.mean(F.relu(1.0 - std))


def covariance_loss(z: torch.Tensor) -> torch.Tensor:
    z = z - z.mean(dim=0)
    cov = (z.T @ z) / max(z.shape[0] - 1, 1)
    off_diag = cov - torch.diag(torch.diag(cov))
    return off_diag.pow(2).sum() / z.shape[1]


def batch_loss(pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    pred_n = F.normalize(pred, dim=-1)
    target_n = F.normalize(target, dim=-1)
    align_mse = F.mse_loss(pred_n, target_n)
    align_cos = 1.0 - F.cosine_similarity(pred, target, dim=-1).mean()
    vic = 0.05 * (variance_loss(pred) + variance_loss(target)) + 0.005 * (
        covariance_loss(pred) + covariance_loss(target)
    )
    loss = align_mse + align_cos + vic
    return loss, {
        "align_mse": float(align_mse.detach().cpu()),
        "align_cos_loss": float(align_cos.detach().cpu()),
        "vicreg": float(vic.detach().cpu()),
    }


@torch.no_grad()
def evaluate(model: ChessJEPA, x_t: torch.Tensor, move_x: torch.Tensor, x_next: torch.Tensor, next_fens: list[str], dev: str) -> dict:
    model.eval()
    z_t = model.encoder(x_t.to(dev))
    pred = model.predictor(z_t, move_x.to(dev))
    target = model.encoder(x_next.to(dev))
    pred_n = F.normalize(pred, dim=-1)
    target_n = F.normalize(target, dim=-1)
    sims = pred_n @ target_n.T
    nn_idx = sims.argmax(dim=-1).cpu()
    top5 = sims.topk(k=min(5, sims.shape[1]), dim=-1).indices.cpu()

    exact_top1 = 0
    exact_top5 = 0
    for i, fen in enumerate(next_fens):
        exact_top1 += int(next_fens[int(nn_idx[i])] == fen)
        exact_top5 += int(any(next_fens[int(j)] == fen for j in top5[i]))

    return {
        "cosine_mean": float(F.cosine_similarity(pred, target, dim=-1).mean().cpu()),
        "mse_normalized": float(F.mse_loss(pred_n, target_n).cpu()),
        "mse_raw": float(F.mse_loss(pred, target).cpu()),
        "nearest_neighbor_top1": exact_top1 / len(next_fens),
        "nearest_neighbor_top5": exact_top5 / len(next_fens),
        "retrieval_candidates": len(next_fens),
        "latent_std_mean": float(target.std(dim=0).mean().cpu()),
    }


def train(cfg: Config) -> dict:
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    dev = choose_device()

    t0 = time.time()
    pairs = generate_pairs(cfg.pairs, cfg.seed)
    rng = random.Random(cfg.seed)
    rng.shuffle(pairs)
    val_pairs = pairs[: cfg.val_pairs]
    train_pairs = pairs[cfg.val_pairs :]
    x_train, move_train, y_train, _ = tensorize(train_pairs)
    x_val, move_val, y_val, val_next_fens = tensorize(val_pairs)

    model = ChessJEPA(cfg).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    n = x_train.shape[0]
    last_loss = {}

    for epoch in range(cfg.epochs):
        model.train()
        order = torch.randperm(n)
        for start in range(0, n, cfg.batch_size):
            idx = order[start : start + cfg.batch_size]
            x_b = x_train[idx].to(dev)
            move_b = move_train[idx].to(dev)
            y_b = y_train[idx].to(dev)
            z_t = model.encoder(x_b)
            pred = model.predictor(z_t, move_b)
            target = model.encoder(y_b)
            loss, last_loss = batch_loss(pred, target)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        if epoch == cfg.epochs - 1 or epoch % 6 == 5:
            metrics = evaluate(model, x_val, move_val, y_val, val_next_fens, dev)
            print(
                f"epoch={epoch + 1:02d} loss={float(loss.detach().cpu()):.4f} "
                f"cos={metrics['cosine_mean']:.3f} "
                f"mse={metrics['mse_normalized']:.4f} "
                f"nn@1={metrics['nearest_neighbor_top1']:.3f}"
            )

    final_metrics = evaluate(model, x_val, move_val, y_val, val_next_fens, dev)
    elapsed = time.time() - t0
    result = {
        "experiment": "chess_jepa_bridge",
        "claim": "Small JEPA-style chess transition bridge trained without Phi/LLM.",
        "config": asdict(cfg),
        "device": dev,
        "counts": {"train_pairs": len(train_pairs), "val_pairs": len(val_pairs)},
        "final_train_loss_terms": last_loss,
        "heldout": final_metrics,
        "runtime_seconds": elapsed,
        "sample_pairs": [asdict(p) for p in val_pairs[:5]],
    }

    artifact = Path(cfg.artifact)
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(json.dumps(result, indent=2) + "\n")
    return result


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Train a no-LLM JEPA bridge over legal chess state transitions.")
    parser.add_argument("--seed", type=int, default=Config.seed)
    parser.add_argument("--pairs", type=int, default=Config.pairs)
    parser.add_argument("--val-pairs", type=int, default=Config.val_pairs)
    parser.add_argument("--epochs", type=int, default=Config.epochs)
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    parser.add_argument("--width", type=int, default=Config.width)
    parser.add_argument("--latent-dim", type=int, default=Config.latent_dim)
    parser.add_argument("--lr", type=float, default=Config.lr)
    parser.add_argument("--weight-decay", type=float, default=Config.weight_decay)
    parser.add_argument("--artifact", default=Config.artifact)
    args = parser.parse_args()
    if args.val_pairs >= args.pairs:
        raise SystemExit("--val-pairs must be smaller than --pairs")
    return Config(**vars(args))


def main() -> None:
    cfg = parse_args()
    result = train(cfg)
    heldout = result["heldout"]
    print(
        "done "
        f"cos={heldout['cosine_mean']:.3f} "
        f"mse={heldout['mse_normalized']:.4f} "
        f"nn@1={heldout['nearest_neighbor_top1']:.3f} "
        f"artifact={result['config']['artifact']}"
    )


if __name__ == "__main__":
    main()
