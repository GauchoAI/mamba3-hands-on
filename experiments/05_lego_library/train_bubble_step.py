"""train_bubble_step — train the comparison-decision step function.
Trivially small: 2 states, 2 actions, ~30 params, trains in milliseconds.
Validates by sorting random lists end-to-end.
"""
import argparse, sys, time, random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, ".")
from bubble_step_function import (
    harvest_pairs, bubble_sort_with_step, ACTIONS, N_ACTIONS,
)


class BubbleStepMLP(nn.Module):
    def __init__(self, d_emb: int = 4, d_hidden: int = 4):
        super().__init__()
        self.feat_emb = nn.Embedding(2, d_emb)
        self.fc1 = nn.Linear(d_emb, d_hidden)
        self.fc2 = nn.Linear(d_hidden, N_ACTIONS)

    def forward(self, state):
        x = self.feat_emb(state[..., 0])
        return self.fc2(F.relu(self.fc1(x)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--save-to", default="checkpoints/specialists/bubble_step.pt")
    args = ap.parse_args()

    pairs = harvest_pairs()
    states = torch.tensor([list(s) for s, _ in pairs], dtype=torch.long, device=args.device)
    actions = torch.tensor([a for _, a in pairs], dtype=torch.long, device=args.device)

    model = BubbleStepMLP().to(args.device)
    print(f"Bubble dataset: {len(pairs)} pairs, params: {sum(p.numel() for p in model.parameters())}")
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
    random.seed(0)
    for desc, lst in [
        ("[3,1,2]", [3, 1, 2]),
        ("[5,4,3,2,1]", [5, 4, 3, 2, 1]),
        ("random 10", [random.randint(0, 99) for _ in range(10)]),
        ("random 50", [random.randint(0, 999) for _ in range(50)]),
        ("random 100", [random.randint(0, 9999) for _ in range(100)]),
        ("already sorted [1..20]", list(range(20))),
    ]:
        sorted_arr, n_swaps = bubble_sort_with_step(lst, model, device=args.device)
        ok = (sorted_arr == sorted(lst))
        mark = "✓" if ok else "✗"
        print(f"  {desc:25}  swaps={n_swaps:>5}  result first 5: {sorted_arr[:5]}  {mark}")

    Path(args.save_to).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": {"d_emb": 4, "d_hidden": 4}},
               args.save_to)
    print(f"\nSaved → {args.save_to}")


if __name__ == "__main__":
    main()
