from __future__ import annotations

import json
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


ROLE_OPEN = 0
ROLE_CLOSE = 1
ROLE_END = 2
ROLE_NAMES = ["OPEN", "CLOSE", "END"]


def write_result(script_path: str, payload: dict) -> None:
    out_dir = Path(script_path).resolve().parent / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "result.json").write_text(json.dumps(payload, indent=2) + "\n")


def finish(start: float, payload: dict, max_seconds: float = 60.0) -> None:
    payload["elapsed_s"] = round(time.time() - start, 4)
    payload["one_minute_rule"] = payload["elapsed_s"] < max_seconds
    write_result(payload["script"], payload)
    print(json.dumps(payload, indent=2))
    if not payload["one_minute_rule"]:
        raise SystemExit("one-minute rule violated")


def seed_all(seed: int) -> random.Random:
    random.seed(seed)
    torch.manual_seed(seed)
    return random.Random(seed)


def balanced_roles(n_pairs: int, rng: random.Random) -> list[int]:
    roles: list[int] = []
    opens = closes = 0
    while closes < n_pairs:
        choices = []
        if opens < n_pairs:
            choices.append(ROLE_OPEN)
        if closes < opens:
            choices.append(ROLE_CLOSE)
        role = rng.choice(choices)
        roles.append(role)
        if role == ROLE_OPEN:
            opens += 1
        else:
            closes += 1
    roles.append(ROLE_END)
    return roles


def valid_stack_sequence(roles: list[int], n_pairs: int | None = None) -> bool:
    depth = opens = closes = 0
    for role in roles:
        if role == ROLE_OPEN:
            depth += 1
            opens += 1
        elif role == ROLE_CLOSE:
            depth -= 1
            closes += 1
            if depth < 0:
                return False
        elif role == ROLE_END:
            return depth == 0 and (n_pairs is None or (opens == n_pairs and closes == n_pairs))
    return False


class TinyMLP(nn.Module):
    def __init__(self, n_in: int, n_out: int = 1, hidden: int = 16):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(), nn.Linear(hidden, n_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_binary_mlp(x: torch.Tensor, y: torch.Tensor, hidden: int = 16, epochs: int = 200,
                     lr: float = 1e-2, seed: int = 0) -> tuple[TinyMLP, dict]:
    torch.manual_seed(seed)
    model = TinyMLP(x.shape[1], 1, hidden)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    t0 = time.time()
    for _ in range(epochs):
        logits = model(x)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred = (torch.sigmoid(model(x)) >= 0.5).float()
        acc = (pred == y).float().mean().item()
    return model, {
        "train_acc": round(acc, 4),
        "train_loss": round(float(loss.item()), 6),
        "train_elapsed_s": round(time.time() - t0, 4),
        "n_params": sum(p.numel() for p in model.parameters()),
    }

