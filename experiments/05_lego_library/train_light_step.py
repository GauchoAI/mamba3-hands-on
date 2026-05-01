"""train_light_step — train the light-propagation step Lego."""
import argparse, sys, time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, ".")
import numpy as np
from light_step_function import (
    harvest_random, correct_outgoing,
    N_MATERIALS, N_DIRS, N_CHANNELS,
    EMPTY, WHITE, RED, GREEN, LIGHT, EMISSION,
)


class LightStepMLP(nn.Module):
    """Structured Lego with hard material-gating.

    Light propagation has three modes (passthrough, scatter, emit) that
    are determined by *material*, not by a learned gate. The orchestrator
    uses the material to pick the mode; the MLP learns only the colors:
      - scatter_color: RGB albedo for SOLID materials
      - emit_color:    RGB emission for LIGHT material

    Hard-gating eliminates compound-error drift: empty cells passthrough
    exactly (no leaks), light cells emit exactly (no over/under), solid
    cells scatter at exactly the learned albedo.

    The MLP body only has to map material → (scatter_color, emit_color).
    A 13-parameter learned function (5 mat × 3 RGB × 2 outputs minus
    the EMPTY/LIGHT slots that don't matter), wrapped in a tiny MLP.
    """
    def __init__(self, d_emb: int = 8, d_hidden: int = 32):
        super().__init__()
        self.mat_emb = nn.Embedding(N_MATERIALS, d_emb)
        self.fc1 = nn.Linear(d_emb, d_hidden)
        self.scatter_head = nn.Linear(d_hidden, 3)
        # Emission is constrained to the -Y direction only (ceiling lamp).
        # MLP outputs 3 RGB; we place them in the -Y bin, zeros elsewhere.
        # This makes "emit only down" exact by construction.
        self.emit_head    = nn.Linear(d_hidden, 3)
        # NY = 3 in the direction order [+X, -X, +Y, -Y, +Z, -Z].
        self._emit_dir_idx = 3

    def forward(self, material, incoming):
        # material: (N,)  incoming: (N, N_DIRS, 3)
        m = self.mat_emb(material)
        h = F.relu(self.fc1(m))
        scat_color = self.scatter_head(h)              # (N, 3)
        emit_dir_y = self.emit_head(h)                 # (N, 3)

        N = material.shape[0]
        emit = torch.zeros(N, N_DIRS, 3, device=incoming.device, dtype=incoming.dtype)
        emit[:, self._emit_dir_idx] = emit_dir_y

        mean_in = incoming.mean(dim=1)                  # (N, 3)
        scatter = (scat_color * mean_in).unsqueeze(1).expand(-1, N_DIRS, -1)

        # Hard gates by material — no soft gate, no compound drift.
        is_empty = (material == EMPTY).unsqueeze(-1).unsqueeze(-1)
        is_light = (material == LIGHT).unsqueeze(-1).unsqueeze(-1)
        is_solid = (~is_empty.squeeze(-1).squeeze(-1)
                    & ~is_light.squeeze(-1).squeeze(-1)).unsqueeze(-1).unsqueeze(-1)

        out = torch.where(is_empty, incoming,
              torch.where(is_light, emit,
              torch.where(is_solid, scatter, incoming)))
        # No relu here: would kill gradients to emit/scatter heads
        # near zero init. Orchestrator clamps to >= 0 at inference.
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--n-samples", type=int, default=200_000)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--save-to", default="checkpoints/specialists/light_step.pt")
    args = ap.parse_args()

    print(f"Generating {args.n_samples:,} (material, incoming, outgoing) tuples…")
    mats_np, incs_np, outs_np = harvest_random(args.n_samples)
    mats = torch.tensor(mats_np, device=args.device)
    incs = torch.tensor(incs_np, device=args.device)
    outs = torch.tensor(outs_np, device=args.device)

    model = LightStepMLP().to(args.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"LightStepMLP params: {n_params}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    t0 = time.time()
    N = args.n_samples
    B = args.batch
    for step in range(args.steps):
        idx = torch.randint(0, N, (B,), device=args.device)
        m = mats[idx]
        i = incs[idx]
        o = outs[idx]
        pred = model(m, i)
        loss = F.mse_loss(pred, o)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if (step + 1) % 200 == 0:
            with torch.no_grad():
                # Eval on a fresh batch
                idx2 = torch.randint(0, N, (8192,), device=args.device)
                pred2 = model(mats[idx2], incs[idx2])
                err = (pred2 - outs[idx2]).abs().mean().item()
            print(f"step {step+1:>4}  loss={loss.item():.5f}  abs_err={err:.5f}", flush=True)
    print(f"Train wall: {time.time()-t0:.2f}s")

    # Validate against the symbolic rule on hand-crafted cases
    print()
    print("Validation against symbolic rule:")
    test_np = np.zeros((N_DIRS, N_CHANNELS), dtype=np.float32)
    test_np[0] = [1.0, 0.5, 0.2]
    test_np[4] = [0.3, 0.3, 0.3]
    test_inc = torch.tensor(test_np, device=args.device).unsqueeze(0)
    for mat in range(N_MATERIALS):
        m = torch.tensor([mat], device=args.device)
        with torch.no_grad():
            pred = model(m, test_inc).squeeze(0).cpu().numpy()
        truth = correct_outgoing(mat, test_inc.squeeze(0).cpu().numpy())
        err = abs(pred - truth).max()
        mark = "✓" if err < 0.3 else "✗"
        names = ["EMPTY", "WHITE", "RED", "GREEN", "LIGHT"]
        print(f"  {names[mat]:>6}: max_err={err:.4f}  {mark}")

    Path(args.save_to).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(),
                "config": {"d_emb": 8, "d_hidden": 32}},
               args.save_to)
    print(f"\nSaved → {args.save_to}")


if __name__ == "__main__":
    main()
