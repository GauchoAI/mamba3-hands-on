"""train_step_function — train the clean Hanoi step function.

f(state_5_ints) -> action_one_of_6.

Tiny MLP. ~1700 params. Trains on (state, action) pairs harvested
from running Hanoi(n) for n in the curriculum. Validates by
running the model step-by-step on Hanoi(n) for any n; checks
every action against Python's reference.

If the model learns the algorithm as a function, OOD n works
trivially — same 5-int state space, same 6-action output, no
sequence length to worry about.
"""
import argparse, sys, time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, ".")
from hanoi_tool import HanoiTool
from hanoi_step_function import (
    ACTIONS, N_ACTIONS, ACTION_TO_IDX, state_for_step,
    harvest_pairs, expected_action_sequence,
)


class StepFunctionMLP(nn.Module):
    """f(state) -> action. 5 int inputs, 6 outputs.

    Each state feature gets a small embedding; concat + 2-layer MLP.
    Total parameters: ~1.7k by default.
    """
    def __init__(self, max_disk: int = 16, d_emb: int = 8, d_hidden: int = 32):
        super().__init__()
        self.max_disk = max_disk
        self.emb_npar = nn.Embedding(2, d_emb)
        self.emb_mpar = nn.Embedding(2, d_emb)
        self.emb_top = nn.Embedding(max_disk + 1, d_emb)
        self.fc1 = nn.Linear(5 * d_emb, d_hidden)
        self.fc2 = nn.Linear(d_hidden, N_ACTIONS)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        # state: (B, 5) int64 with [n_par, m_par, top_A, top_B, top_C]
        n_par, m_par, ta, tb, tc = state.unbind(-1)
        ta = ta.clamp(0, self.max_disk)
        tb = tb.clamp(0, self.max_disk)
        tc = tc.clamp(0, self.max_disk)
        x = torch.cat([
            self.emb_npar(n_par),
            self.emb_mpar(m_par),
            self.emb_top(ta),
            self.emb_top(tb),
            self.emb_top(tc),
        ], dim=-1)
        return self.fc2(F.relu(self.fc1(x)))


def step_through(model, n: int, device: str, max_disk: int) -> tuple[bool, int, int]:
    """Run the model step-by-step on Hanoi(n). Compare each action
    to Python's reference; return (all_correct, n_correct, total)."""
    tool = HanoiTool(n)
    expected = expected_action_sequence(n)
    correct = 0
    for ref_action in expected:
        s = state_for_step(tool)
        s_t = torch.tensor([list(s)], dtype=torch.long, device=device)
        # Clamp top values to max_disk to keep embedding lookup safe
        s_t[0, 2:].clamp_(0, max_disk)
        with torch.no_grad():
            logits = model(s_t)
        pred = int(logits[0].argmax().item())
        if pred == ref_action:
            correct += 1
        # Apply the REFERENCE action to advance the tool (so a single
        # wrong prediction doesn't cascade and tank the rest of the
        # eval). We're scoring per-step accuracy.
        src, dst = ACTIONS[ref_action]
        # Need to find which disk is moving
        moved_disk = None
        for k_idx, p in enumerate(tool.peg):
            if p == src:
                if moved_disk is None or (k_idx + 1) < moved_disk:
                    moved_disk = k_idx + 1
        if moved_disk is not None:
            tool.peg[moved_disk - 1] = dst
        tool.move_index += 1
    total = len(expected)
    return (correct == total), correct, total


def step_through_autoregressive(model, n: int, device: str, max_disk: int) -> tuple[bool, int, int]:
    """Strictly autoregressive: apply the MODEL'S own action each step.
    A wrong action propagates an incorrect state forward. This is the
    real test of step-function correctness — partial credit doesn't
    apply, the trajectory either matches or diverges."""
    tool = HanoiTool(n)
    expected = expected_action_sequence(n)
    correct = 0
    for ref_action in expected:
        s = state_for_step(tool)
        s_t = torch.tensor([list(s)], dtype=torch.long, device=device)
        s_t[0, 2:].clamp_(0, max_disk)
        with torch.no_grad():
            logits = model(s_t)
        pred = int(logits[0].argmax().item())
        if pred == ref_action:
            correct += 1
        # Apply the MODEL's chosen action.
        src, dst = ACTIONS[pred]
        moved_disk = None
        for k_idx, p in enumerate(tool.peg):
            if p == src:
                if moved_disk is None or (k_idx + 1) < moved_disk:
                    moved_disk = k_idx + 1
        if moved_disk is not None:
            tool.peg[moved_disk - 1] = dst
        tool.move_index += 1
    return (correct == len(expected)), correct, len(expected)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-disk", type=int, default=16)
    ap.add_argument("--d-emb", type=int, default=8)
    ap.add_argument("--d-hidden", type=int, default=32)
    ap.add_argument("--curriculum", default="2,3,4,5,6")
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--save-to", default="checkpoints/specialists/hanoi_step_fn.pt")
    args = ap.parse_args()

    ns = [int(s) for s in args.curriculum.split(",")]
    pairs = harvest_pairs(ns)
    print(f"Curriculum: {ns}, pairs: {len(pairs)}, distinct states: {len(set(s for s,_ in pairs))}",
          flush=True)

    states = torch.tensor([list(s) for s, _ in pairs], dtype=torch.long, device=args.device)
    actions = torch.tensor([a for _, a in pairs], dtype=torch.long, device=args.device)

    model = StepFunctionMLP(max_disk=args.max_disk, d_emb=args.d_emb,
                             d_hidden=args.d_hidden).to(args.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    t0 = time.time()
    for step in range(args.steps):
        idx = torch.randint(0, len(pairs), (args.batch_size,), device=args.device)
        s_b = states[idx]
        a_b = actions[idx]
        logits = model(s_b)
        loss = F.cross_entropy(logits, a_b)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if (step + 1) % 200 == 0:
            with torch.no_grad():
                full_logits = model(states)
                acc = (full_logits.argmax(-1) == actions).float().mean().item()
            print(f"step {step+1:>4}  loss={loss.item():.4f}  full_dataset_acc={acc:.1%}",
                  flush=True)
    print(f"Train wall: {time.time()-t0:.1f}s", flush=True)

    # Final teacher-forced eval per n in curriculum + a few OOD
    print()
    print(f"{'n':>3}  {'in_dist':>7}  {'tf_correct':>11}  {'ar_correct':>11}  {'ar_pass':>8}")
    print("-" * 55)
    for n_ in [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20]:
        in_dist = "yes" if n_ in ns else "no"
        ok_tf, c_tf, t_tf = step_through(model, n_, args.device, args.max_disk)
        ok_ar, c_ar, t_ar = step_through_autoregressive(model, n_, args.device, args.max_disk)
        ar_pass = "✓" if ok_ar else "✗"
        print(f"{n_:>3}  {in_dist:>7}  {c_tf}/{t_tf:<7}  {c_ar}/{t_ar:<7}  {ar_pass:>5}")

    Path(args.save_to).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "config": {"max_disk": args.max_disk, "d_emb": args.d_emb,
                   "d_hidden": args.d_hidden},
    }, args.save_to)
    print(f"\nSaved → {args.save_to}", flush=True)


if __name__ == "__main__":
    main()
