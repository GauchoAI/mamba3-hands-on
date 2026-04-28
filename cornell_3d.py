"""cornell_3d — proper 3D Cornell box rendered by the light-step Lego.

The 3D voxel orchestrator. Each cell holds (material, 6 directions × RGB).
Per step:
  1. Gather incoming light at each cell from its 6 neighbors' outgoing.
  2. Run the trained Lego on every cell in one MLP forward.
  3. Accumulate per-cell brightness; clamp to non-negative.

Then "render": orthographic camera in front of the box, looking toward
the back. For each pixel, march the column of voxels and sample the
first non-empty cell's outgoing[-Z] (the light leaving that surface
toward the camera). This gives a Cornell-looking image.

Scene: 5 walls (RED left, GREEN right, WHITE floor/ceiling/back),
one ceiling light, optionally two interior WHITE boxes (the canonical
Cornell tall-box + short-box).

Dimensions:
  X: width  (0 = left/RED, W-1 = right/GREEN)
  Y: height (0 = floor/WHITE, H-1 = ceiling/WHITE)
  Z: depth  (0 = camera side, no wall; D-1 = back/WHITE)
"""
import argparse, sys, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, ".")
from light_step_function import (
    EMPTY, WHITE, RED, GREEN, LIGHT,
    N_DIRS, N_CHANNELS, N_MATERIALS,
    ALBEDO_WHITE, ALBEDO_RED, ALBEDO_GREEN, EMISSION_DOWN,
)
from train_light_step import LightStepMLP

# Direction indices matching light_step_function:
PX, NX, PY, NY, PZ, NZ = 0, 1, 2, 3, 4, 5


def build_cornell_3d(W: int, H: int, D: int,
                     light_size: int = 12,
                     with_boxes: bool = True) -> np.ndarray:
    """Build a 3D Cornell scene as a (W, H, D) int array of materials."""
    mat = np.full((W, H, D), EMPTY, dtype=np.int64)

    # Walls
    mat[ 0, :, :]   = RED        # left wall (X=0)
    mat[W-1, :, :]  = GREEN      # right wall (X=W-1)
    mat[:,  0, :]   = WHITE      # floor    (Y=0)
    mat[:, H-1, :]  = WHITE      # ceiling  (Y=H-1)
    mat[:, :, D-1]  = WHITE      # back wall (Z=D-1)
    # No front wall (Z=0): camera side stays open.

    # Light is part of the ceiling itself (replaces ceiling cells in the
    # central patch). LIGHT cells emit only in -Y, like a real ceiling lamp.
    cx, cz = W // 2, D // 2
    half = light_size // 2
    mat[cx-half:cx+half, H-1, cz-half:cz+half] = LIGHT

    if with_boxes:
        # Tall box: WHITE block on the left side of the box.
        x0, x1 = W // 6,  W // 6 + W // 5
        z0, z1 = D // 4,  D // 4 + D // 4
        y0, y1 = 1,        H // 2
        mat[x0:x1, y0:y1, z0:z1] = WHITE

        # Short box: WHITE block on the right side, in front of the tall.
        x0, x1 = W // 2 + W // 8, W // 2 + W // 8 + W // 5
        z0, z1 = D // 8,            D // 8 + D // 4
        y0, y1 = 1,                  H // 4
        mat[x0:x1, y0:y1, z0:z1] = WHITE

    return mat


def gather_incoming(outgoing: torch.Tensor) -> torch.Tensor:
    """incoming[x, y, z, d] = outgoing[neighbor_in_-d, d].

    Light moving in direction d arrives at this cell from the neighbor
    on the opposite side. So incoming[+X] of (x, y, z) comes from the
    outgoing[+X] of (x-1, y, z) (the neighbor to the left, which sent
    light moving rightward).
    """
    W, H, D, _, _ = outgoing.shape
    incoming = torch.zeros_like(outgoing)
    # +X: from x-1
    incoming[1:, :, :, PX, :] = outgoing[:W-1, :, :, PX, :]
    # -X: from x+1
    incoming[:W-1, :, :, NX, :] = outgoing[1:, :, :, NX, :]
    # +Y: from y-1
    incoming[:, 1:, :, PY, :] = outgoing[:, :H-1, :, PY, :]
    # -Y: from y+1
    incoming[:, :H-1, :, NY, :] = outgoing[:, 1:, :, NY, :]
    # +Z: from z-1
    incoming[:, :, 1:, PZ, :] = outgoing[:, :, :D-1, PZ, :]
    # -Z: from z+1
    incoming[:, :, :D-1, NZ, :] = outgoing[:, :, 1:, NZ, :]
    return incoming


def step_lego(materials, incoming, model):
    W, H, D = materials.shape
    flat_mat = materials.flatten()
    flat_inc = incoming.view(W * H * D, N_DIRS, N_CHANNELS)
    with torch.no_grad():
        flat_out = model(flat_mat, flat_inc)
    return flat_out.view(W, H, D, N_DIRS, N_CHANNELS)


def step_symbolic(materials, incoming, sc_albedo, sc_emission_per_dir):
    """sc_emission_per_dir: (N_MATERIALS, N_DIRS, 3) — directional emission."""
    W, H, D = materials.shape
    mat_flat = materials.flatten()
    inc_flat = incoming.view(W * H * D, N_DIRS, N_CHANNELS)
    is_empty = (mat_flat == EMPTY)
    is_light = (mat_flat == LIGHT)
    albedo   = sc_albedo[mat_flat]                            # (N, 3)
    mean_in  = inc_flat.mean(dim=1, keepdim=True)             # (N, 1, 3)
    scatter  = (albedo.unsqueeze(1) * mean_in).expand(-1, N_DIRS, -1)
    emit     = sc_emission_per_dir[mat_flat]                  # (N, N_DIRS, 3)
    out_flat = torch.where(
        is_empty.unsqueeze(-1).unsqueeze(-1), inc_flat,
        torch.where(is_light.unsqueeze(-1).unsqueeze(-1), emit, scatter)
    )
    return torch.relu(out_flat).view(W, H, D, N_DIRS, N_CHANNELS)


def propagate(materials_np, n_steps, model, device, use_lego=True):
    W, H, D = materials_np.shape
    materials = torch.tensor(materials_np, device=device, dtype=torch.long)
    outgoing = torch.zeros((W, H, D, N_DIRS, N_CHANNELS), device=device)

    sc_albedo = torch.zeros((N_MATERIALS, N_CHANNELS), device=device)
    sc_albedo[WHITE] = torch.tensor(ALBEDO_WHITE, device=device)
    sc_albedo[RED]   = torch.tensor(ALBEDO_RED,   device=device)
    sc_albedo[GREEN] = torch.tensor(ALBEDO_GREEN, device=device)
    # Per-direction emission: only LIGHT emits, only in -Y (index 3, NY).
    sc_emission_pd = torch.zeros((N_MATERIALS, N_DIRS, N_CHANNELS), device=device)
    sc_emission_pd[LIGHT, NY] = torch.tensor(EMISSION_DOWN, device=device)

    for _ in range(n_steps):
        incoming = gather_incoming(outgoing)
        if use_lego:
            outgoing = step_lego(materials, incoming, model)
        else:
            outgoing = step_symbolic(materials, incoming, sc_albedo, sc_emission_pd)

    return outgoing.cpu().numpy(), materials.cpu().numpy()


def render_camera(materials_np, outgoing_np, view_W=256, view_H=256):
    """Orthographic camera at z=-1 looking toward +Z. For each pixel,
    march the corresponding voxel column from z=0 toward z=D-1 and find
    the first non-empty cell. Sample its outgoing[-Z] (light heading
    toward the camera). That's what the camera sees on that pixel."""
    W, H, D = materials_np.shape
    img = np.zeros((view_H, view_W, 3), dtype=np.float32)
    # Sample (px, py) → voxel (x, y) by linear mapping.
    # Image y=0 at top, scene y=H-1 at top → invert.
    for py in range(view_H):
        for px in range(view_W):
            x = int(px / view_W * W)
            y = (H - 1) - int(py / view_H * H)
            x = max(0, min(W - 1, x))
            y = max(0, min(H - 1, y))
            for z in range(D):
                if materials_np[x, y, z] != EMPTY:
                    # Light heading toward camera = -Z direction.
                    img[py, px] = outgoing_np[x, y, z, NZ]
                    break
    return img


def _direction_bin(d: np.ndarray) -> int:
    """Map a continuous direction vector to the closest of 6 axis-aligned
    bins: +X=0, -X=1, +Y=2, -Y=3, +Z=4, -Z=5."""
    ax = int(np.argmax(np.abs(d)))
    sign_pos = d[ax] >= 0
    return ax * 2 + (0 if sign_pos else 1)


def render_perspective(materials_np, outgoing_np, view_W=512, view_H=512,
                       cam_offset_z: float = -0.6,
                       cam_offset_y: float = 0.05,
                       fov_deg: float = 55.0,
                       max_march: int = 256):
    """Perspective camera with vectorized ray-marching through the voxel
    grid. For each pixel, cast a ray from the camera; at the first
    non-empty voxel, sample its outgoing in the closest axis-aligned
    direction back toward the camera.

    Default camera: in front of the box, slightly above center, FOV 55°.
    This puts the side walls inside the frustum so red/green color bleed
    becomes visible at the floor near the walls.
    """
    W, H, D = materials_np.shape
    cam = np.array([W * 0.5, H * (0.5 + cam_offset_y), W * cam_offset_z], dtype=np.float32)
    target = np.array([W * 0.5, H * 0.5, D * 0.5], dtype=np.float32)

    forward = target - cam
    forward /= np.linalg.norm(forward)
    up0 = np.array([0, 1, 0], dtype=np.float32)
    # Right-handed camera basis: right = cross(up, forward). Without this
    # the right vector flips and the side-wall colors swap.
    right = np.cross(up0, forward); right /= np.linalg.norm(right)
    up = np.cross(forward, right); up /= np.linalg.norm(up)

    aspect = view_W / view_H
    half_h = np.tan(np.radians(fov_deg) / 2)
    half_w = aspect * half_h

    py, px = np.mgrid[0:view_H, 0:view_W]
    u = (px + 0.5) / view_W * 2 - 1            # (H, W) in [-1, 1]
    v = (py + 0.5) / view_H * 2 - 1
    ray_dir = (forward[None, None, :]
               + u[..., None] * half_w * right[None, None, :]
               - v[..., None] * half_h * up[None, None, :])
    ray_dir /= np.linalg.norm(ray_dir, axis=-1, keepdims=True)
    ray_dir = ray_dir.reshape(-1, 3)            # (N, 3)
    N = ray_dir.shape[0]

    pos = np.broadcast_to(cam, (N, 3)).copy()
    img = np.zeros((N, 3), dtype=np.float32)
    alive = np.ones(N, dtype=bool)

    # Step size: small enough to not skip a voxel.
    step = 0.5

    for _ in range(max_march):
        if not alive.any():
            break
        ix = pos[:, 0].astype(np.int32)
        iy = pos[:, 1].astype(np.int32)
        iz = pos[:, 2].astype(np.int32)
        in_bounds = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H) & (iz >= 0) & (iz < D)
        can_check = alive & in_bounds
        if can_check.any():
            ix_safe = np.clip(ix, 0, W - 1)
            iy_safe = np.clip(iy, 0, H - 1)
            iz_safe = np.clip(iz, 0, D - 1)
            mat = materials_np[ix_safe, iy_safe, iz_safe]
            hit = (mat != EMPTY) & can_check
            if hit.any():
                hit_idx = np.where(hit)[0]
                # Sampling rule:
                #   LIGHT  → outgoing[-Y]: emission direction (downward).
                #   SOLID  → outgoing[0]:  any direction (all 6 equal for our rule).
                hit_mat = mat[hit_idx]
                is_light_hit = (hit_mat == LIGHT)
                # NY = 3 in [+X, -X, +Y, -Y, +Z, -Z]
                bins = np.where(is_light_hit, NY, 0)
                img[hit_idx] = outgoing_np[
                    ix_safe[hit_idx], iy_safe[hit_idx], iz_safe[hit_idx],
                    bins,
                ]
                alive &= ~hit

        pos += ray_dir * step

    return img.reshape(view_H, view_W, 3)


def tonemap_to_png(arr: np.ndarray, path: str, exposure: float = 1.0):
    a = np.clip(arr * exposure, 0, None)
    a = a / (a + 1.0)              # Reinhard
    a = np.power(a, 1/2.2)         # gamma
    a = (np.clip(a, 0, 1) * 255).astype(np.uint8)
    from PIL import Image
    Image.fromarray(a).save(path)
    print(f"  saved {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--W", type=int, default=32)
    ap.add_argument("--H", type=int, default=32)
    ap.add_argument("--D", type=int, default=32)
    ap.add_argument("--steps", type=int, default=96)
    ap.add_argument("--view-W", type=int, default=256)
    ap.add_argument("--view-H", type=int, default=256)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--ckpt", default="checkpoints/specialists/light_step.pt")
    ap.add_argument("--no-boxes", action="store_true")
    ap.add_argument("--exposure", type=float, default=1.0)
    ap.add_argument("--camera", choices=["ortho", "persp"], default="persp")
    args = ap.parse_args()

    print(f"Device: {args.device}")
    ck = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    model = LightStepMLP(**ck["config"]).to(args.device)
    model.load_state_dict(ck["model"])
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Lego: LightStepMLP, {n_params} params, {N_DIRS} directions")
    print()

    materials = build_cornell_3d(args.W, args.H, args.D, with_boxes=not args.no_boxes)
    n_voxels = args.W * args.H * args.D
    print(f"Cornell-3D scene: {args.W}×{args.H}×{args.D} = {n_voxels:,} voxels, "
          f"{args.steps} propagation steps")
    print()

    sync = (torch.mps.synchronize if args.device == "mps"
            else torch.cuda.synchronize if args.device == "cuda" else None)

    if sync: sync()
    t0 = time.time()
    out_lego, mat_np = propagate(materials, args.steps, model, args.device, use_lego=True)
    if sync: sync()
    t_lego = time.time() - t0

    if sync: sync()
    t0 = time.time()
    out_sym, _ = propagate(materials, args.steps, model, args.device, use_lego=False)
    if sync: sync()
    t_sym = time.time() - t0

    print(f"Propagation:")
    print(f"  Lego                  : {t_lego*1000:>8.1f} ms   max={out_lego.max():.3f}  mean={out_lego.mean():.4f}")
    print(f"  Symbolic (torch ref)  : {t_sym*1000:>8.1f} ms   max={out_sym.max():.3f}  mean={out_sym.mean():.4f}")
    diff = np.abs(out_lego - out_sym)
    print(f"  max diff Lego vs sym  : {diff.max():.4f}, mean: {diff.mean():.5f}")
    print()

    if args.camera == "ortho":
        print("Rendering camera view (orthographic, looking at +Z)…")
        t0 = time.time()
        img_lego = render_camera(mat_np, out_lego, args.view_W, args.view_H)
        img_sym  = render_camera(mat_np, out_sym,  args.view_W, args.view_H)
    else:
        print("Rendering camera view (perspective, ray-marched)…")
        t0 = time.time()
        img_lego = render_perspective(mat_np, out_lego, args.view_W, args.view_H)
        img_sym  = render_perspective(mat_np, out_sym,  args.view_W, args.view_H)
    print(f"  render time: {(time.time()-t0)*1000:.1f} ms")
    print()

    tonemap_to_png(img_lego, "cornell3d_lego.png", args.exposure)
    tonemap_to_png(img_sym,  "cornell3d_symbolic.png", args.exposure)


if __name__ == "__main__":
    main()
