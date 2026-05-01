"""cornell_lightca — Cornell-flat rendered by the light-step Lego.

The orchestrator: builds a 2D Cornell scene as a grid of materials,
runs the trained light-step Lego for N propagation steps, and writes
out the resulting irradiance image.

Per step:
  1. Gather incoming light at each cell from neighbors' previous outgoing.
     - incoming[N] of (r, c) ← outgoing[N] of (r+1, c)
     - incoming[S] of (r, c) ← outgoing[S] of (r-1, c)
     - incoming[E] of (r, c) ← outgoing[E] of (r, c-1)
     - incoming[W] of (r, c) ← outgoing[W] of (r, c+1)
     Boundary cells: incoming from outside the grid is 0.
  2. Run Lego on every cell in parallel: outgoing = model(material, incoming).
  3. Accumulate per-cell irradiance for visualization.

Compare against:
  - naive Python: same algorithm, per-cell loop. Slow as molasses.
  - NumPy: vectorized but uses the symbolic rule directly (no Lego).
  - Lego on MPS: uses the trained MLP, all cells in one forward.
"""
import argparse, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
from light_step_function import (
    EMPTY, WHITE, RED, GREEN, LIGHT,
    N_DIRS, N_CHANNELS, N_MATERIALS,
    ALBEDO_WHITE, ALBEDO_RED, ALBEDO_GREEN, EMISSION,
    correct_outgoing,
)
from train_light_step import LightStepMLP

# Direction indices: N=0, S=1, E=2, W=3
N, S, E, W = 0, 1, 2, 3


def build_cornell(H: int = 64, Wd: int = 64, light_w: int = 12):
    """Cornell-flat top-down: red left, green right, white top/bottom,
    light strip near the top."""
    mat = np.full((H, Wd), EMPTY, dtype=np.int64)
    mat[0, :]    = WHITE       # ceiling
    mat[H-1, :]  = WHITE       # floor
    mat[:, 0]    = RED         # left
    mat[:, Wd-1] = GREEN       # right
    # corners get a defined material — left wall takes priority
    mat[0, 0] = WHITE; mat[0, Wd-1] = WHITE
    mat[H-1, 0] = WHITE; mat[H-1, Wd-1] = WHITE
    # Light strip at row 1, centered
    cx = Wd // 2
    mat[1, cx - light_w//2 : cx + light_w//2] = LIGHT
    return mat


def gather_incoming(outgoing: torch.Tensor) -> torch.Tensor:
    """Build incoming[r, c, d] = outgoing[neighbor_in_-d, c, d]."""
    H, Wd, _, _ = outgoing.shape
    incoming = torch.zeros_like(outgoing)
    # N (decreasing row): incoming[r, c, N] = outgoing[r+1, c, N]
    incoming[:H-1, :, N, :] = outgoing[1:, :, N, :]
    # S (increasing row): incoming[r, c, S] = outgoing[r-1, c, S]
    incoming[1:, :, S, :]   = outgoing[:H-1, :, S, :]
    # E (increasing col): incoming[r, c, E] = outgoing[r, c-1, E]
    incoming[:, 1:, E, :]   = outgoing[:, :Wd-1, E, :]
    # W (decreasing col): incoming[r, c, W] = outgoing[r, c+1, W]
    incoming[:, :Wd-1, W, :] = outgoing[:, 1:, W, :]
    return incoming


def step_lego(materials: torch.Tensor, incoming: torch.Tensor, model) -> torch.Tensor:
    """Apply the trained Lego to every cell in one forward pass."""
    H, Wd = materials.shape
    flat_mat = materials.flatten()
    flat_inc = incoming.view(H * Wd, N_DIRS, N_CHANNELS)
    with torch.no_grad():
        flat_out = model(flat_mat, flat_inc)
    return flat_out.view(H, Wd, N_DIRS, N_CHANNELS)


def step_symbolic_torch(materials: torch.Tensor, incoming: torch.Tensor,
                        sc_albedo, sc_emission) -> torch.Tensor:
    """Symbolic rule via torch ops — for sanity comparison."""
    H, Wd = materials.shape
    mat_flat = materials.flatten()                          # (HW,)
    inc_flat = incoming.view(H * Wd, N_DIRS, N_CHANNELS)    # (HW, 4, 3)
    is_empty = (mat_flat == EMPTY).unsqueeze(-1).unsqueeze(-1)  # (HW, 1, 1)
    is_light = (mat_flat == LIGHT).unsqueeze(-1).unsqueeze(-1)
    is_solid = (~is_empty.squeeze(-1).squeeze(-1) & ~is_light.squeeze(-1).squeeze(-1)).unsqueeze(-1).unsqueeze(-1)
    albedo = sc_albedo[mat_flat]                             # (HW, 3)
    emission = sc_emission[mat_flat]                         # (HW, 3)
    mean_in = inc_flat.mean(dim=1, keepdim=True)             # (HW, 1, 3)
    scatter = (albedo.unsqueeze(1) * mean_in).expand(-1, N_DIRS, -1)
    emit = emission.unsqueeze(1).expand(-1, N_DIRS, -1)
    out_flat = torch.where(is_empty, inc_flat,
               torch.where(is_light, emit,
               torch.where(is_solid, scatter, inc_flat)))
    return out_flat.view(H, Wd, N_DIRS, N_CHANNELS)


def render(materials_np: np.ndarray, n_steps: int, model, device,
           use_lego: bool = True):
    """Run propagation and accumulate irradiance per cell."""
    H, Wd = materials_np.shape
    materials = torch.tensor(materials_np, device=device, dtype=torch.long)
    outgoing = torch.zeros((H, Wd, N_DIRS, N_CHANNELS), device=device)
    accum = torch.zeros((H, Wd, N_CHANNELS), device=device)

    sc_albedo = torch.zeros((N_MATERIALS, N_CHANNELS), device=device)
    sc_albedo[WHITE] = torch.tensor(ALBEDO_WHITE, device=device)
    sc_albedo[RED]   = torch.tensor(ALBEDO_RED,   device=device)
    sc_albedo[GREEN] = torch.tensor(ALBEDO_GREEN, device=device)
    sc_emission = torch.zeros((N_MATERIALS, N_CHANNELS), device=device)
    sc_emission[LIGHT] = torch.tensor(EMISSION, device=device)

    for _ in range(n_steps):
        incoming = gather_incoming(outgoing)
        if use_lego:
            outgoing = step_lego(materials, incoming, model)
        else:
            outgoing = step_symbolic_torch(materials, incoming, sc_albedo, sc_emission)
        # Accumulate brightness: total light leaving each cell
        accum = accum + outgoing.sum(dim=2)

    return accum.cpu().numpy() / n_steps


def tonemap_to_png(arr: np.ndarray, path: str):
    a = np.clip(arr, 0, None)
    a = a / (a + 1.0)
    a = np.power(a, 1/2.2)
    a = (np.clip(a, 0, 1) * 255).astype(np.uint8)
    try:
        from PIL import Image
        Image.fromarray(a).save(path)
        print(f"  saved {path}")
    except ImportError:
        print(f"  PIL not installed; skipping save of {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--H", type=int, default=64)
    ap.add_argument("--W", type=int, default=64)
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--ckpt", default="checkpoints/specialists/light_step.pt")
    args = ap.parse_args()

    print(f"Device: {args.device}")
    print()

    ck = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    model = LightStepMLP(**ck["config"]).to(args.device)
    model.load_state_dict(ck["model"])
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Lego: LightStepMLP, {n_params} params")
    print()

    materials = build_cornell(args.H, args.W)
    print(f"Cornell-flat scene: {args.H}×{args.W}, {args.steps} propagation steps")
    print()

    sync = (torch.mps.synchronize if args.device == "mps"
            else torch.cuda.synchronize if args.device == "cuda" else None)

    # Run Lego
    if sync: sync()
    t0 = time.time()
    img_lego = render(materials, args.steps, model, args.device, use_lego=True)
    if sync: sync()
    t_lego = time.time() - t0
    print(f"  Lego (trained MLP)    : {t_lego*1000:>8.1f} ms   max={img_lego.max():.3f}  mean={img_lego.mean():.3f}")

    # Run symbolic for comparison
    if sync: sync()
    t0 = time.time()
    img_sym = render(materials, args.steps, model, args.device, use_lego=False)
    if sync: sync()
    t_sym = time.time() - t0
    print(f"  Symbolic (rule via torch): {t_sym*1000:>8.1f} ms   max={img_sym.max():.3f}  mean={img_sym.mean():.3f}")

    diff = np.abs(img_lego - img_sym)
    print(f"  max diff Lego vs symbolic: {diff.max():.4f}, mean: {diff.mean():.4f}")
    print()

    tonemap_to_png(img_lego, "cornell_lego.png")
    tonemap_to_png(img_sym,  "cornell_symbolic.png")


if __name__ == "__main__":
    main()
