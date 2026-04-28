"""train_gcd_step — clean step function for Euclidean GCD by subtraction.

f(state_4_bits) -> action_one_of_3.

Even simpler than Hanoi's: 3 reachable states (a>b, b>a, a==b),
3 actions (sub_b_from_a, sub_a_from_b, done). MLP with ~few-hundred
params; trains in fractions of a second.

Validate by stepping through GCD(a, b) for huge (a, b) pairs the
model never trained on — should be perfect by construction since
the state space is closed.
"""
import argparse, sys, time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, ".")
from gcd_step_function import (
    state_for_pair, correct_action, gcd_step, gcd_trajectory,
    harvest_pairs, ACTIONS, N_ACTIONS, ACTION_TO_IDX,
)


class GCDStepMLP(nn.Module):
    def __init__(self, d_emb: int = 4, d_hidden: int = 16):
        super().__init__()
        self.feat_emb = nn.Embedding(2, d_emb)  # binary features
        self.fc1 = nn.Linear(4 * d_emb, d_hidden)
        self.fc2 = nn.Linear(d_hidden, N_ACTIONS)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        # state: (B, 4) of 0/1
        agt, bgt, az, bz = state.unbind(-1)
        x = torch.cat([
            self.feat_emb(agt), self.feat_emb(bgt),
            self.feat_emb(az),  self.feat_emb(bz),
        ], dim=-1)
        return self.fc2(F.relu(self.fc1(x)))


def step_through_ar(model, a: int, b: int, device: str,
                    max_steps: int = 100000) -> tuple[bool, int, int]:
    """Run the model autoregressively on GCD(a, b). Returns
    (correct_termination, model_gcd, true_gcd)."""
    from math import gcd as math_gcd
    true = math_gcd(a, b)
    cur_a, cur_b = a, b
    for step in range(max_steps):
        s = state_for_pair(cur_a, cur_b)
        s_t = torch.tensor([list(s)], dtype=torch.long, device=device)
        with torch.no_grad():
            logits = model(s_t)
        act = int(logits[0].argmax().item())
        if act == ACTION_TO_IDX["done"]:
            ok = (cur_a == cur_b) and (cur_a == true)
            return ok, cur_a, true
        # Apply action
        if act == ACTION_TO_IDX["sub_b_from_a"]:
            if cur_a <= cur_b:
                # Wrong action — model would create invalid state
                return False, cur_a, true
            cur_a -= cur_b
        elif act == ACTION_TO_IDX["sub_a_from_b"]:
            if cur_b <= cur_a:
                return False, cur_a, true
            cur_b -= cur_a
    return False, cur_a, true   # didn't terminate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--curriculum-max", type=int, default=10)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--save-to", default="checkpoints/specialists/gcd_step.pt")
    args = ap.parse_args()

    pairs = harvest_pairs(args.curriculum_max)
    distinct = set(s for s, _ in pairs)
    print(f"Curriculum (a,b) in 1..{args.curriculum_max}²: {len(pairs)} pairs, {len(distinct)} distinct states",
          flush=True)

    states = torch.tensor([list(s) for s, _ in pairs], dtype=torch.long, device=args.device)
    actions = torch.tensor([a for _, a in pairs], dtype=torch.long, device=args.device)

    model = GCDStepMLP().to(args.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    t0 = time.time()
    for step in range(args.steps):
        idx = torch.randint(0, len(pairs), (args.batch_size,), device=args.device)
        logits = model(states[idx])
        loss = F.cross_entropy(logits, actions[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if (step + 1) % 200 == 0:
            with torch.no_grad():
                acc = (model(states).argmax(-1) == actions).float().mean().item()
            print(f"step {step+1:>4}  loss={loss.item():.5f}  acc={acc:.1%}", flush=True)
    print(f"Train wall: {time.time()-t0:.2f}s", flush=True)

    print()
    print(f"{'(a,b)':>14}  {'true_gcd':>8}  {'model_gcd':>9}  {'correct':>7}")
    print("-" * 50)
    test_pairs = [
        (12, 8), (15, 6), (100, 75), (1000, 6),
        (123, 456), (12345, 67890), (1_000_000, 999_999),
        (50, 50), (1, 7), (7, 1),
    ]
    fails = 0
    for a, b in test_pairs:
        ok, mg, tg = step_through_ar(model, a, b, args.device)
        mark = "✓" if ok else "✗"
        if not ok:
            fails += 1
        print(f"  ({a:>5},{b:>5})  {tg:>8}  {mg:>9}  {mark:>5}")

    Path(args.save_to).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(),
                "config": {"d_emb": 4, "d_hidden": 16}}, args.save_to)
    print(f"\nSaved → {args.save_to}", flush=True)
    if fails:
        print(f"FAIL — {fails} mismatches")
        return 1
    print("PASS — model is the GCD step function for any (a, b).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
