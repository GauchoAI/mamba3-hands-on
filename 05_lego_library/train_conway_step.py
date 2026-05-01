"""train_conway_step — train the Conway-cell step function."""
import argparse, sys, time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, ".")
from conway_step_function import (
    harvest_pairs, correct_action, step_grid, ACTIONS, N_ACTIONS,
)


class ConwayStepMLP(nn.Module):
    def __init__(self, d_emb: int = 4, d_hidden: int = 8):
        super().__init__()
        self.alive_emb = nn.Embedding(2, d_emb)
        self.neigh_emb = nn.Embedding(9, d_emb)
        self.fc1 = nn.Linear(2 * d_emb, d_hidden)
        self.fc2 = nn.Linear(d_hidden, N_ACTIONS)

    def forward(self, state):
        a, n = state.unbind(-1)
        x = torch.cat([self.alive_emb(a), self.neigh_emb(n)], dim=-1)
        return self.fc2(F.relu(self.fc1(x)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--save-to", default="checkpoints/specialists/conway_step.pt")
    args = ap.parse_args()

    pairs = harvest_pairs()
    print(f"Conway dataset: {len(pairs)} pairs (all distinct states)")
    states = torch.tensor([list(s) for s, _ in pairs], dtype=torch.long, device=args.device)
    actions = torch.tensor([a for _, a in pairs], dtype=torch.long, device=args.device)

    model = ConwayStepMLP().to(args.device)
    print(f"Params: {sum(p.numel() for p in model.parameters())}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    t0 = time.time()
    for step in range(args.steps):
        # Just train on the full 18-pair dataset each step.
        logits = model(states)
        loss = F.cross_entropy(logits, actions)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if (step + 1) % 100 == 0:
            with torch.no_grad():
                acc = (model(states).argmax(-1) == actions).float().mean().item()
            print(f"step {step+1:>4}  loss={loss.item():.5f}  acc={acc:.1%}", flush=True)
    print(f"Train wall: {time.time()-t0:.2f}s")

    # Validate on a few interesting Life patterns
    print()
    patterns = {
        "blinker (period 2)": [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ],
        "block (still life)": [
            [0, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0],
        ],
        "glider": [
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [1, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ],
    }
    for name, grid in patterns.items():
        h = len(grid); w = len(grid[0])
        # Build (state per cell) batch and run model
        ref = step_grid(grid)
        batch = []
        for r in range(h):
            for c in range(w):
                n = sum(
                    grid[(r + dr) % h][(c + dc) % w]
                    for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                    if not (dr == 0 and dc == 0)
                )
                batch.append((grid[r][c], n))
        batch_t = torch.tensor(batch, dtype=torch.long, device=args.device)
        with torch.no_grad():
            preds = model(batch_t).argmax(-1).cpu().tolist()
        model_grid = [preds[r * w:(r + 1) * w] for r in range(h)]
        match = (model_grid == ref)
        mark = "✓" if match else "✗"
        print(f"  {name:25} {mark}")

    Path(args.save_to).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": {"d_emb": 4, "d_hidden": 8}},
               args.save_to)
    print(f"\nSaved → {args.save_to}")


if __name__ == "__main__":
    main()
