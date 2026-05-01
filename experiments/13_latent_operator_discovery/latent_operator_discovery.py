from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from operator_curriculum_lib import ROLE_CLOSE, ROLE_END, ROLE_OPEN, finish, seed_all


PAD = 3


def candidate_is_valid(opens: int, closes: int, target_pairs: int, candidate: int) -> bool:
    if candidate == ROLE_OPEN:
        return opens < target_pairs
    if candidate == ROLE_CLOSE:
        return opens > closes
    if candidate == ROLE_END:
        return opens == target_pairs and closes == target_pairs
    raise ValueError(candidate)


def random_balanced_prefix_rows(max_pairs: int, samples_per_n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    rows: list[dict] = []
    for n in range(1, max_pairs + 1):
        for _ in range(samples_per_n):
            prefix: list[int] = []
            opens = closes = 0
            while True:
                for candidate in (ROLE_OPEN, ROLE_CLOSE, ROLE_END):
                    rows.append({
                        "prefix": prefix[:],
                        "target_pairs": n,
                        "candidate": candidate,
                        "valid": float(candidate_is_valid(opens, closes, n, candidate)),
                        "depth": float(opens - closes),
                        "can_close": float(opens > closes),
                        "can_end": float(opens == n and closes == n),
                    })
                choices = []
                if opens < n:
                    choices.append(ROLE_OPEN)
                if closes < opens:
                    choices.append(ROLE_CLOSE)
                if not choices:
                    break
                role = rng.choice(choices)
                prefix.append(role)
                if role == ROLE_OPEN:
                    opens += 1
                else:
                    closes += 1
    return rows


def batch(rows: list[dict], max_len: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    seq = torch.full((len(rows), max_len), PAD, dtype=torch.long)
    target = torch.tensor([r["target_pairs"] for r in rows], dtype=torch.long)
    candidate = torch.tensor([r["candidate"] for r in rows], dtype=torch.long)
    valid = torch.tensor([[r["valid"]] for r in rows], dtype=torch.float32)
    boundary = torch.tensor([[r["can_close"], r["can_end"]] for r in rows], dtype=torch.float32)
    for i, row in enumerate(rows):
        prefix = row["prefix"][:max_len]
        if prefix:
            seq[i, :len(prefix)] = torch.tensor(prefix, dtype=torch.long)
    return seq, target, candidate, valid, boundary


class LatentPrefixOperator(nn.Module):
    def __init__(self, hidden: int = 64, emb: int = 8, max_pairs_scale: int = 32):
        super().__init__()
        self.max_pairs_scale = max_pairs_scale
        self.embedding = nn.Embedding(4, emb)
        self.encoder = nn.GRU(emb + 1, hidden, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden + 4, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def encode(self, seq: torch.Tensor, target_pairs: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(seq)
        target = (target_pairs.float() / self.max_pairs_scale).view(-1, 1, 1)
        target = target.expand(seq.shape[0], seq.shape[1], 1)
        _, hidden = self.encoder(torch.cat([embedded, target], dim=-1))
        return hidden.squeeze(0)

    def forward(self, seq: torch.Tensor, target_pairs: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
        hidden = self.encode(seq, target_pairs)
        candidate_one_hot = F.one_hot(candidate, 3).float()
        target = (target_pairs.float() / self.max_pairs_scale).view(-1, 1)
        return self.head(torch.cat([hidden, target, candidate_one_hot], dim=-1))


def train_boundary_probe(
    train_hidden: torch.Tensor,
    train_boundary: torch.Tensor,
    heldout_hidden: torch.Tensor,
    heldout_boundary: torch.Tensor,
    seed: int,
) -> tuple[float, float]:
    torch.manual_seed(seed)
    probe = nn.Linear(train_hidden.shape[1], train_boundary.shape[1])
    opt = torch.optim.AdamW(probe.parameters(), lr=5e-3)
    for _ in range(180):
        logits = probe(train_hidden)
        loss = F.binary_cross_entropy_with_logits(logits, train_boundary)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    with torch.no_grad():
        train_pred = (torch.sigmoid(probe(train_hidden)) >= 0.5).float()
        heldout_pred = (torch.sigmoid(probe(heldout_hidden)) >= 0.5).float()
        train_acc = (train_pred == train_boundary).float().mean().item()
        heldout_acc = (heldout_pred == heldout_boundary).float().mean().item()
    return train_acc, heldout_acc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=360)
    parser.add_argument("--max-train-pairs", type=int, default=16)
    parser.add_argument("--min-heldout-pairs", type=int, default=17)
    parser.add_argument("--max-heldout-pairs", type=int, default=20)
    parser.add_argument("--samples-per-n", type=int, default=5)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    start = time.time()
    seed_all(args.seed)
    max_len = args.max_heldout_pairs * 2
    train_rows = random_balanced_prefix_rows(args.max_train_pairs, args.samples_per_n, args.seed)
    heldout_rows = [
        row
        for row in random_balanced_prefix_rows(args.max_heldout_pairs, 2, args.seed + 1)
        if row["target_pairs"] >= args.min_heldout_pairs
    ]
    train = batch(train_rows, max_len)
    heldout = batch(heldout_rows, max_len)

    model = LatentPrefixOperator(hidden=64, max_pairs_scale=max(32, args.max_heldout_pairs))
    opt = torch.optim.AdamW(model.parameters(), lr=5e-3)
    t0 = time.time()
    for _ in range(args.epochs):
        logits = model(train[0], train[1], train[2])
        loss = F.binary_cross_entropy_with_logits(logits, train[3])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    with torch.no_grad():
        train_pred = (torch.sigmoid(model(train[0], train[1], train[2])) >= 0.5).float()
        heldout_pred = (torch.sigmoid(model(heldout[0], heldout[1], heldout[2])) >= 0.5).float()
        train_acc = (train_pred == train[3]).float().mean().item()
        heldout_acc = (heldout_pred == heldout[3]).float().mean().item()
        train_hidden = model.encode(train[0], train[1]).detach()
        heldout_hidden = model.encode(heldout[0], heldout[1]).detach()

    probe_train_acc, probe_holdout_acc = train_boundary_probe(train_hidden, train[4], heldout_hidden, heldout[4], args.seed)

    payload = {
        "script": __file__,
        "experiment": "latent_operator_discovery",
        "claim": "A recurrent learner can recover stack-relevant latent state from raw prefixes without explicit open/close/depth counters.",
        "train": {
            "examples": len(train_rows),
            "pairs": f"1..{args.max_train_pairs}",
            "acc": round(train_acc, 4),
            "loss": round(float(loss.item()), 6),
            "train_elapsed_s": round(time.time() - t0, 4),
            "n_params": sum(p.numel() for p in model.parameters()),
        },
        "heldout": {
            "examples": len(heldout_rows),
            "pairs": f"{args.min_heldout_pairs}..{args.max_heldout_pairs}",
            "candidate_validity_acc": round(heldout_acc, 4),
        },
        "boundary_probe": {
            "labels": ["can_close", "can_end"],
            "train_acc": round(probe_train_acc, 4),
            "heldout_acc": round(probe_holdout_acc, 4),
        },
    }
    finish(start, payload)
    if heldout_acc < 0.95:
        raise SystemExit("heldout candidate validity below threshold")
    if probe_holdout_acc < 0.85:
        raise SystemExit("hidden state boundary probe below threshold")


if __name__ == "__main__":
    main()
