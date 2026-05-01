"""Stack operator transfer.

Train a tiny supervised policy that decides whether a candidate role is valid
for the current stack-generation state. The policy is surface-independent:
generation happens over roles, then renderers map roles to parentheses,
brackets, or word blocks.

This is deliberately a one-minute experiment. The default run should complete
comfortably on CPU.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


HERE = Path(__file__).resolve().parent
ROLE_OPEN = 0
ROLE_CLOSE = 1
ROLE_END = 2
ROLE_NAMES = ["OPEN", "CLOSE", "END"]
SURFACES = {
    "paren": ("(", ")"),
    "bracket": ("[", "]"),
    "block": ("BEGIN", "END"),
}


@dataclass(frozen=True)
class StackState:
    opens: int
    closes: int
    target_pairs: int

    @property
    def depth(self) -> int:
        return self.opens - self.closes


def candidate_is_valid(state: StackState, role: int) -> bool:
    if role == ROLE_OPEN:
        return state.opens < state.target_pairs
    if role == ROLE_CLOSE:
        return state.depth > 0
    if role == ROLE_END:
        return state.opens == state.target_pairs and state.closes == state.target_pairs
    raise ValueError(f"unknown role: {role}")


def apply_role(state: StackState, role: int) -> StackState:
    if role == ROLE_OPEN:
        return StackState(state.opens + 1, state.closes, state.target_pairs)
    if role == ROLE_CLOSE:
        return StackState(state.opens, state.closes + 1, state.target_pairs)
    if role == ROLE_END:
        return state
    raise ValueError(f"unknown role: {role}")


def featurize(state: StackState, role: int) -> list[float]:
    n = max(1, state.target_pairs)
    role_one_hot = [1.0 if role == i else 0.0 for i in range(3)]
    return [
        state.opens / n,
        state.closes / n,
        state.depth / n,
        1.0 if state.depth == 0 else 0.0,
        1.0 if state.opens == n else 0.0,
        1.0 if state.closes == n else 0.0,
        *role_one_hot,
    ]


def reachable_states(target_pairs: int) -> list[StackState]:
    states = []
    for opens in range(target_pairs + 1):
        for closes in range(opens + 1):
            states.append(StackState(opens, closes, target_pairs))
    return states


def build_dataset(max_pairs: int) -> tuple[torch.Tensor, torch.Tensor]:
    xs: list[list[float]] = []
    ys: list[float] = []
    for n in range(1, max_pairs + 1):
        for state in reachable_states(n):
            for role in (ROLE_OPEN, ROLE_CLOSE, ROLE_END):
                xs.append(featurize(state, role))
                ys.append(1.0 if candidate_is_valid(state, role) else 0.0)
    return torch.tensor(xs, dtype=torch.float32), torch.tensor(ys, dtype=torch.float32).unsqueeze(-1)


class StackOperatorMLP(nn.Module):
    def __init__(self, d_hidden: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(9, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_operator(max_pairs: int, epochs: int, lr: float, d_hidden: int, seed: int) -> tuple[StackOperatorMLP, dict]:
    torch.manual_seed(seed)
    random.seed(seed)
    x, y = build_dataset(max_pairs)
    model = StackOperatorMLP(d_hidden=d_hidden)
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
        "train_examples": int(x.shape[0]),
        "train_acc": round(acc, 4),
        "train_loss": round(float(loss.item()), 6),
        "train_elapsed_s": round(time.time() - t0, 4),
        "n_params": sum(p.numel() for p in model.parameters()),
    }


def operator_valid_roles(model: StackOperatorMLP, state: StackState) -> list[int]:
    xs = torch.tensor([featurize(state, r) for r in (ROLE_OPEN, ROLE_CLOSE, ROLE_END)], dtype=torch.float32)
    with torch.no_grad():
        probs = torch.sigmoid(model(xs)).flatten().tolist()
    return [r for r, p in enumerate(probs) if p >= 0.5]


def exact_valid_roles(state: StackState) -> list[int]:
    return [r for r in (ROLE_OPEN, ROLE_CLOSE, ROLE_END) if candidate_is_valid(state, r)]


def choose_noisy_role(rng: random.Random) -> int:
    # This intentionally proposes invalid closes/ends often enough that the
    # operator's contribution is visible.
    return rng.choices([ROLE_OPEN, ROLE_CLOSE, ROLE_END], weights=[0.42, 0.42, 0.16], k=1)[0]


def generate_roles(target_pairs: int, rng: random.Random, model: StackOperatorMLP | None, max_steps_factor: int = 4) -> list[int]:
    state = StackState(0, 0, target_pairs)
    roles: list[int] = []
    for _ in range(max(4, max_steps_factor * target_pairs + 4)):
        proposal = choose_noisy_role(rng)
        if model is None:
            role = proposal
        else:
            valid = operator_valid_roles(model, state)
            if proposal in valid:
                role = proposal
            elif valid:
                # Keep the base generator in the loop: pick the first valid
                # candidate according to the same noisy preference order.
                role = valid[0]
            else:
                role = ROLE_END
        roles.append(role)
        if role == ROLE_END:
            break
        if candidate_is_valid(state, role):
            state = apply_role(state, role)
        elif model is None:
            # Baseline generator can corrupt state; keep going so verifier
            # catches the failure instead of repairing it here.
            continue
    return roles


def verify_roles(roles: list[int], target_pairs: int) -> bool:
    state = StackState(0, 0, target_pairs)
    for role in roles:
        if role == ROLE_END:
            return state.opens == target_pairs and state.closes == target_pairs
        if not candidate_is_valid(state, role):
            return False
        state = apply_role(state, role)
    return False


def render_roles(roles: list[int], surface: str) -> str:
    open_tok, close_tok = SURFACES[surface]
    out = []
    for role in roles:
        if role == ROLE_OPEN:
            out.append(open_tok)
        elif role == ROLE_CLOSE:
            out.append(close_tok)
        elif role == ROLE_END:
            break
    if surface == "block":
        return " ".join(out)
    return "".join(out)


def heldout_state_accuracy(model: StackOperatorMLP, min_pairs: int, max_pairs: int) -> float:
    x, y = build_dataset(max_pairs)
    keep = []
    row = 0
    for n in range(1, max_pairs + 1):
        n_rows = len(reachable_states(n)) * 3
        if n >= min_pairs:
            keep.extend(range(row, row + n_rows))
        row += n_rows
    idx = torch.tensor(keep, dtype=torch.long)
    with torch.no_grad():
        pred = (torch.sigmoid(model(x[idx])) >= 0.5).float()
        return float((pred == y[idx]).float().mean().item())


def run_trials(model: StackOperatorMLP | None, trials: int, target_pairs: int, surface: str, seed: int) -> dict:
    rng = random.Random(seed)
    passes = 0
    samples = []
    for i in range(trials):
        roles = generate_roles(target_pairs, rng, model)
        ok = verify_roles(roles, target_pairs)
        passes += int(ok)
        if i < 3:
            samples.append({
                "roles": [ROLE_NAMES[r] for r in roles],
                "rendered": render_roles(roles, surface),
                "valid": ok,
            })
    return {
        "surface": surface,
        "target_pairs": target_pairs,
        "trials": trials,
        "pass_rate": round(passes / max(1, trials), 4),
        "samples": samples,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-max-pairs", type=int, default=6)
    ap.add_argument("--heldout-max-pairs", type=int, default=24)
    ap.add_argument("--epochs", type=int, default=250)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--d-hidden", type=int, default=16)
    ap.add_argument("--trials", type=int, default=80)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out-dir", default=str(HERE / "artifacts"))
    args = ap.parse_args()

    wall0 = time.time()
    model, train_stats = train_operator(
        max_pairs=args.train_max_pairs,
        epochs=args.epochs,
        lr=args.lr,
        d_hidden=args.d_hidden,
        seed=args.seed,
    )
    heldout_acc = heldout_state_accuracy(model, args.train_max_pairs + 1, args.heldout_max_pairs)

    baseline = run_trials(None, args.trials, target_pairs=12, surface="paren", seed=args.seed)
    guided = [
        run_trials(model, args.trials, target_pairs=n, surface=surface, seed=args.seed + n + i)
        for i, (surface, n) in enumerate([
            ("paren", 12),
            ("bracket", 12),
            ("block", 12),
            ("paren", 20),
            ("bracket", 20),
        ])
    ]
    elapsed = time.time() - wall0
    result = {
        "experiment": "stack_operator_transfer",
        "one_minute_rule": elapsed < 60.0,
        "elapsed_s": round(elapsed, 4),
        "train": train_stats,
        "heldout_state_acc": round(heldout_acc, 4),
        "baseline": baseline,
        "guided": guided,
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "config": {
            "d_hidden": args.d_hidden,
            "train_max_pairs": args.train_max_pairs,
            "features": [
                "opens_frac", "closes_frac", "depth_frac", "depth_zero",
                "opens_done", "closes_done", "role_open", "role_close", "role_end",
            ],
        },
        "result": result,
    }, out_dir / "stack_operator.pt")
    (out_dir / "stack_operator_results.json").write_text(json.dumps(result, indent=2) + "\n")

    print(json.dumps(result, indent=2))
    if elapsed >= 60.0:
        raise SystemExit("one-minute rule violated")
    if heldout_acc < 0.98:
        raise SystemExit("held-out state accuracy below acceptance threshold")
    if any(r["pass_rate"] < 0.95 for r in guided):
        raise SystemExit("guided generation pass rate below acceptance threshold")


if __name__ == "__main__":
    main()

