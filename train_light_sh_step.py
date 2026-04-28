"""train_light_sh_step — train the SH-native light Lego.

Same hard-gating pattern as the 6-dir version: material decides which
mode (EMPTY/LIGHT/SOLID), and the MLP only learns *parameters*:
  albedo: 3 RGB per material
  emit_color: 3 RGB per material (used only by LIGHT)

The propagation math (irradiance, hemisphere SH, downward emission) is
analytical and lives in the forward pass.
"""
import argparse, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, ".")
from light_sh_step_function import (
    harvest_random, correct_outgoing_sh,
    N_MATERIALS, N_SH, N_CHANNELS,
    EMPTY, WHITE, RED, GREEN, LIGHT,
    K0, K1, SQRT_PI, SQRT_PI_OVER_3,
)


class LightSHStepMLP(nn.Module):
    """SH-native Lego with hard material-gating.

    Body: material → (albedo, emit_color). Forward applies the analytical
    SH math (irradiance, hemisphere SH for SOLID, downward emission SH
    for LIGHT, passthrough for EMPTY).
    """
    def __init__(self, d_emb: int = 8, d_hidden: int = 24):
        super().__init__()
        self.mat_emb = nn.Embedding(N_MATERIALS, d_emb)
        self.fc1 = nn.Linear(d_emb, d_hidden)
        self.albedo_head = nn.Linear(d_hidden, 3)
        self.emit_head   = nn.Linear(d_hidden, 3)

    def forward(self, material, normal, incoming_sh):
        # material: (N,)  normal: (N, 3)  incoming_sh: (N, 4, 3)
        m = self.mat_emb(material)
        h = F.relu(self.fc1(m))
        albedo = self.albedo_head(h)        # (N, 3)
        emit_color = self.emit_head(h)      # (N, 3)

        # Irradiance E(n) = (sqrt(pi)/2)·c0 - sqrt(pi/3)·(n · c_vec)
        c0 = incoming_sh[:, 0]              # (N, 3)
        cy = incoming_sh[:, 1]
        cz = incoming_sh[:, 2]
        cx = incoming_sh[:, 3]
        nx = normal[:, 0:1]; ny = normal[:, 1:2]; nz = normal[:, 2:3]
        E = (SQRT_PI / 2.0) * c0 - SQRT_PI_OVER_3 * (cx*nx + cy*ny + cz*nz)  # (N, 3)

        N = material.shape[0]
        out_sh = torch.zeros(N, N_SH, N_CHANNELS,
                             device=incoming_sh.device, dtype=incoming_sh.dtype)

        is_empty = (material == EMPTY).unsqueeze(-1).unsqueeze(-1)  # (N,1,1)
        is_light = (material == LIGHT).unsqueeze(-1).unsqueeze(-1)
        # SOLID = not empty and not light
        is_solid = ~is_empty & ~is_light

        # EMPTY branch
        empty_out = incoming_sh

        # LIGHT branch: fixed clamped-cosine downward emission, scaled per-channel
        # by emit_color. c_0 = emit · sqrt(pi)/2, c_y = -emit · sqrt(pi/3), others = 0.
        light_out = torch.zeros_like(out_sh)
        light_out[:, 0] = emit_color * (SQRT_PI / 2.0)
        light_out[:, 1] = -emit_color * SQRT_PI_OVER_3

        # SOLID branch: outgoing = (albedo · E / pi) · hemisphere_SH(n)
        # hemisphere_SH(n): c_0 = sqrt(pi), c_l1m_axis = sqrt(3*pi)/2 · n_axis
        # so outgoing.c_0 = albedo·E / sqrt(pi), outgoing.c_axis = albedo·E·K1·n_axis
        common = albedo * E                     # (N, 3)
        solid_out = torch.zeros_like(out_sh)
        solid_out[:, 0] = common / SQRT_PI
        solid_out[:, 1] = common * K1 * ny      # c_y
        solid_out[:, 2] = common * K1 * nz      # c_z
        solid_out[:, 3] = common * K1 * nx      # c_x

        out_sh = torch.where(is_empty, empty_out,
                  torch.where(is_light, light_out, solid_out))
        return out_sh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--n-samples", type=int, default=200_000)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--save-to", default="checkpoints/specialists/light_sh_step.pt")
    args = ap.parse_args()

    print(f"Generating {args.n_samples:,} (mat, normal, in_SH, out_SH) tuples…")
    mats_np, norms_np, incs_np, outs_np = harvest_random(args.n_samples)
    mats  = torch.tensor(mats_np,  device=args.device)
    norms = torch.tensor(norms_np, device=args.device)
    incs  = torch.tensor(incs_np,  device=args.device)
    outs  = torch.tensor(outs_np,  device=args.device)

    model = LightSHStepMLP().to(args.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"LightSHStepMLP params: {n_params}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    t0 = time.time()
    N = args.n_samples
    B = args.batch
    for step in range(args.steps):
        idx = torch.randint(0, N, (B,), device=args.device)
        pred = model(mats[idx], norms[idx], incs[idx])
        loss = F.mse_loss(pred, outs[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if (step + 1) % 200 == 0:
            with torch.no_grad():
                idx2 = torch.randint(0, N, (8192,), device=args.device)
                pred2 = model(mats[idx2], norms[idx2], incs[idx2])
                err = (pred2 - outs[idx2]).abs().mean().item()
            print(f"step {step+1:>4}  loss={loss.item():.6f}  abs_err={err:.5f}", flush=True)
    print(f"Train wall: {time.time()-t0:.2f}s")

    # Validation against symbolic rule
    print()
    print("Validation against symbolic rule:")
    test_inc = np.zeros((N_SH, N_CHANNELS), dtype=np.float32)
    test_inc[0] = [10.0, 10.0, 10.0]   # uniform DC
    for mat in range(N_MATERIALS):
        # axis-aligned normal +Y for testing
        n = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        m = torch.tensor([mat], device=args.device)
        nn_t = torch.tensor(n, device=args.device).unsqueeze(0)
        in_t = torch.tensor(test_inc, device=args.device).unsqueeze(0)
        with torch.no_grad():
            pred = model(m, nn_t, in_t).squeeze(0).cpu().numpy()
        truth = correct_outgoing_sh(mat, n, test_inc)
        err = np.abs(pred - truth).max()
        names = ["EMPTY", "WHITE", "RED", "GREEN", "LIGHT"]
        mark = "✓" if err < 0.5 else "✗"
        print(f"  {names[mat]:>6}: max_err={err:.4f}  {mark}")

    Path(args.save_to).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(),
                "config": {"d_emb": 8, "d_hidden": 24}},
               args.save_to)
    print(f"\nSaved → {args.save_to}")


if __name__ == "__main__":
    main()
