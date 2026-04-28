"""train_maze_step — train the greedy grid-navigation step function."""
import argparse, sys, time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, ".")
from maze_step_function import (
    harvest_pairs, navigate, ACTIONS, N_ACTIONS,
)


class MazeStepMLP(nn.Module):
    def __init__(self, d_emb: int = 4, d_hidden: int = 8):
        super().__init__()
        self.feat_emb = nn.Embedding(3, d_emb)  # 0,1,2 for sign trit
        self.fc1 = nn.Linear(2 * d_emb, d_hidden)
        self.fc2 = nn.Linear(d_hidden, N_ACTIONS)

    def forward(self, state):
        dx, dy = state.unbind(-1)
        x = torch.cat([self.feat_emb(dx), self.feat_emb(dy)], dim=-1)
        return self.fc2(F.relu(self.fc1(x)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--save-to", default="checkpoints/specialists/maze_step.pt")
    args = ap.parse_args()

    pairs = harvest_pairs()
    states = torch.tensor([list(s) for s, _ in pairs], dtype=torch.long, device=args.device)
    actions = torch.tensor([a for _, a in pairs], dtype=torch.long, device=args.device)

    model = MazeStepMLP().to(args.device)
    print(f"Maze dataset: {len(pairs)} pairs, params: {sum(p.numel() for p in model.parameters())}")
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
            print(f"step {step+1:>3}  loss={loss.item():.5f}  acc={acc:.1%}", flush=True)
    print(f"Train wall: {time.time()-t0:.2f}s")

    print()
    test_pairs = [
        ((0, 0), (0, 0)),         # already at goal
        ((0, 0), (3, 0)),         # 3 east
        ((0, 0), (0, 3)),         # 3 south
        ((0, 0), (5, -7)),        # diagonal
        ((10, 10), (-50, -75)),   # far-away
        ((0, 0), (1000, -1000)),  # very far
    ]
    for s, g in test_pairs:
        final, hist, n = navigate(s, g, model, device=args.device, max_steps=10000)
        ok = (final == g)
        mark = "✓" if ok else "✗"
        print(f"  {s}→{g}  final={final}  steps={n}  {mark}")

    Path(args.save_to).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": {"d_emb": 4, "d_hidden": 8}},
               args.save_to)
    print(f"\nSaved → {args.save_to}")


if __name__ == "__main__":
    main()
