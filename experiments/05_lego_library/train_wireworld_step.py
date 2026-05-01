"""train_wireworld_step — train the WireWorld per-cell step function."""
import argparse, sys, time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, ".")
from wireworld_step_function import (
    harvest_pairs, step_grid, ACTIONS, N_ACTIONS,
)


class WireWorldStepMLP(nn.Module):
    def __init__(self, d_emb: int = 4, d_hidden: int = 16):
        super().__init__()
        self.cell_emb = nn.Embedding(4, d_emb)
        self.neigh_emb = nn.Embedding(9, d_emb)
        self.fc1 = nn.Linear(2 * d_emb, d_hidden)
        self.fc2 = nn.Linear(d_hidden, N_ACTIONS)

    def forward(self, state):
        c, n = state.unbind(-1)
        x = torch.cat([self.cell_emb(c), self.neigh_emb(n)], dim=-1)
        return self.fc2(F.relu(self.fc1(x)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--save-to", default="checkpoints/specialists/wireworld_step.pt")
    args = ap.parse_args()

    pairs = harvest_pairs()
    print(f"WireWorld dataset: {len(pairs)} pairs (all distinct states)")
    states = torch.tensor([list(s) for s, _ in pairs], dtype=torch.long, device=args.device)
    actions = torch.tensor([a for _, a in pairs], dtype=torch.long, device=args.device)

    model = WireWorldStepMLP().to(args.device)
    print(f"Params: {sum(p.numel() for p in model.parameters())}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    t0 = time.time()
    for step in range(args.steps):
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

    # Validate by running a small wire-and-clock circuit a few generations.
    # A "diode" wire with one electron pulse: head→tail→conductor stream.
    print()
    # Simple wire: a horizontal line of conductor with a head→tail starter
    grid = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 2, 3, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]
    h = len(grid); w = len(grid[0])
    g = grid
    print("WireWorld pulse on a wire (3 steps):")
    for t in range(3):
        ref = step_grid(g)
        # batched model step
        batch = []
        for r in range(h):
            for c in range(w):
                n = sum(
                    1 for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                    if not (dr == 0 and dc == 0)
                    and g[(r + dr) % h][(c + dc) % w] == 2
                )
                batch.append((g[r][c], n))
        bt = torch.tensor(batch, dtype=torch.long, device=args.device)
        with torch.no_grad():
            preds = model(bt).argmax(-1).cpu().tolist()
        new_g = [preds[r * w:(r + 1) * w] for r in range(h)]
        match = (new_g == ref)
        mark = "✓" if match else "✗"
        print(f"  step {t+1}: {mark}  middle row = {new_g[1]}")
        g = new_g

    Path(args.save_to).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": {"d_emb": 4, "d_hidden": 16}},
               args.save_to)
    print(f"\nSaved → {args.save_to}")


if __name__ == "__main__":
    main()
