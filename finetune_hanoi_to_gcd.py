"""finetune_hanoi_to_gcd — test idea A (fast fine-tune across tasks).

Take the trained Hanoi step-function MLP, swap its output head for
GCD's 3-class action space, continue-train on GCD pairs. Compare
to training a fresh GCD model from scratch.

Question: does the Hanoi-trained substrate accelerate convergence,
or is each step function so small that initialization barely
matters?
"""
import argparse, sys, time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, ".")
from train_step_function import StepFunctionMLP
from gcd_step_function import (
    state_for_pair, harvest_pairs, ACTIONS, N_ACTIONS, ACTION_TO_IDX,
)
from train_gcd_step import GCDStepMLP, step_through_ar


class HanoiTransferToGCD(nn.Module):
    """Reuse the Hanoi MLP's state-embedding + hidden layer; replace
    the output head + adjust input embedding for GCD's 4-bit state."""
    def __init__(self, hanoi_model: StepFunctionMLP):
        super().__init__()
        # Reuse the n_parity / move_parity embeddings as the 4 binary feature
        # embeddings (they all have d_emb dim and are 2-class).
        d_emb = hanoi_model.emb_npar.weight.shape[1]
        d_hidden = hanoi_model.fc1.out_features
        # 4 binary features → reuse npar/mpar embeddings as 2 of them; new for the other 2.
        self.feat_emb_a = hanoi_model.emb_npar  # frozen reuse
        self.feat_emb_b = hanoi_model.emb_mpar  # frozen reuse
        self.feat_emb_c = nn.Embedding(2, d_emb)
        self.feat_emb_d = nn.Embedding(2, d_emb)
        # Transfer the hidden layer fully but adapt the input projection
        # since GCD has 4 features × d_emb instead of Hanoi's 5.
        self.fc1 = nn.Linear(4 * d_emb, d_hidden)
        # Initialize fc1 from hanoi's fc1 (just take the first 4 feature columns).
        with torch.no_grad():
            self.fc1.weight.copy_(hanoi_model.fc1.weight[:, : 4 * d_emb])
            self.fc1.bias.copy_(hanoi_model.fc1.bias)
        # New 3-class output head (GCD has 3 actions vs Hanoi's 6).
        self.fc2 = nn.Linear(d_hidden, N_ACTIONS)

    def forward(self, state):
        a, b, c, d = state.unbind(-1)
        x = torch.cat([
            self.feat_emb_a(a), self.feat_emb_b(b),
            self.feat_emb_c(c), self.feat_emb_d(d),
        ], dim=-1)
        return self.fc2(F.relu(self.fc1(x)))


def train_loop(model, states, actions, steps: int, lr: float, batch_size: int,
               device: str, verbose: bool = True):
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    t0 = time.time()
    history = []
    for step in range(steps):
        idx = torch.randint(0, len(states), (batch_size,), device=device)
        logits = model(states[idx])
        loss = F.cross_entropy(logits, actions[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if (step + 1) % 50 == 0 or step == 0:
            with torch.no_grad():
                acc = (model(states).argmax(-1) == actions).float().mean().item()
            history.append({"step": step + 1, "loss": loss.item(), "acc": acc})
            if verbose:
                print(f"  step {step+1:>4}  loss={loss.item():.5f}  acc={acc:.1%}", flush=True)
    return time.time() - t0, history


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}\n", flush=True)

    pairs = harvest_pairs(curriculum_max=10)
    states = torch.tensor([list(s) for s, _ in pairs], dtype=torch.long, device=device)
    actions = torch.tensor([a for _, a in pairs], dtype=torch.long, device=device)

    # Run A: from-scratch GCD MLP
    print("--- from-scratch GCD MLP ---")
    fresh = GCDStepMLP().to(device)
    fresh_params = sum(p.numel() for p in fresh.parameters())
    print(f"params: {fresh_params}")
    fresh_wall, fresh_hist = train_loop(fresh, states, actions, 500, 3e-3, 64, device)
    print(f"wall: {fresh_wall:.2f}s\n")

    # Run B: Hanoi-pretrained MLP, fine-tune on GCD
    print("--- Hanoi -> GCD fine-tune ---")
    ck = torch.load("checkpoints/specialists/hanoi_step_fn.pt",
                    map_location=device, weights_only=False)
    hanoi = StepFunctionMLP(max_disk=ck["config"]["max_disk"],
                            d_emb=ck["config"]["d_emb"],
                            d_hidden=ck["config"]["d_hidden"]).to(device)
    hanoi.load_state_dict(ck["model"])
    transfer = HanoiTransferToGCD(hanoi).to(device)
    transfer_params = sum(p.numel() for p in transfer.parameters())
    print(f"params: {transfer_params}")
    transfer_wall, transfer_hist = train_loop(transfer, states, actions, 500, 3e-3, 64, device)
    print(f"wall: {transfer_wall:.2f}s\n")

    # Compare convergence: how many steps to hit 100% accuracy?
    def steps_to_100(hist):
        for h in hist:
            if h["acc"] >= 0.999:
                return h["step"]
        return None

    fs_steps = steps_to_100(fresh_hist)
    tf_steps = steps_to_100(transfer_hist)
    print(f"Steps to 100% acc:")
    print(f"  fresh:    {fs_steps}")
    print(f"  transfer: {tf_steps}")
    if fs_steps and tf_steps:
        ratio = fs_steps / tf_steps
        print(f"  ratio: transfer is {ratio:.1f}x {'faster' if ratio>1 else 'slower'}")

    # AR validate the transferred model on a few tough GCD pairs
    print()
    print("AR validation (transfer model on real GCD problems):")
    for a, b in [(12, 8), (100, 75), (123, 456), (12345, 67890)]:
        ok, mg, tg = step_through_ar(transfer, a, b, device)
        print(f"  ({a}, {b}) -> true={tg} model={mg} {'✓' if ok else '✗'}")


if __name__ == "__main__":
    main()
