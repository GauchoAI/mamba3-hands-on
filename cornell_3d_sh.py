"""cornell_3d_sh — proper 3D Cornell box with SH-native light propagation.

Per-cell state: 4 SH coefs × 3 RGB = 12 floats (representing the outgoing
radiance distribution).

Per step (LPV-style):
  1. Pre-compute per-cell surface normals (where empty neighbors live).
  2. Gather: for each cell, sum flux from 6 neighbors as delta beams.
  3. Apply per-material rule (EMPTY: passthrough, LIGHT: emission SH,
     SOLID: Lambertian via irradiance + hemisphere SH).
  4. Render: SH evaluation at -ray_dir (smooth radiance lookup).

Key change from 6-dir version: Lambertian scatter at SOLID cells now
concentrates outgoing into the hemisphere of the surface normal (l=1 SH
coefs aligned with n) instead of dispersing across all 6 dirs. Per-bounce
per-direction retention rises ~4×, so indirect light survives many more
bounces — proper color bleed across the whole scene.
"""
import argparse, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
from light_sh_step_function import (
    EMPTY, WHITE, RED, GREEN, LIGHT,
    N_SH, N_CHANNELS, N_MATERIALS,
    K0, K1, SQRT_PI, SQRT_PI_OVER_3, SQRT_3PI_OVER_2,
    light_emission_sh, ALBEDO_WHITE, ALBEDO_RED, ALBEDO_GREEN, EMISSION_DOWN,
)
from train_light_sh_step import LightSHStepMLP


# ── Scene construction ─────────────────────────────────────────

def build_cornell_3d(W: int, H: int, D: int, light_size: int = 12,
                     with_boxes: bool = True) -> np.ndarray:
    mat = np.full((W, H, D), EMPTY, dtype=np.int64)
    mat[ 0, :, :]   = RED
    mat[W-1, :, :]  = GREEN
    mat[:,  0, :]   = WHITE
    mat[:, H-1, :]  = WHITE
    mat[:, :, D-1]  = WHITE
    cx, cz = W // 2, D // 2
    half = light_size // 2
    mat[cx-half:cx+half, H-1, cz-half:cz+half] = LIGHT
    if with_boxes:
        x0, x1 = W // 6,  W // 6 + W // 5
        z0, z1 = D // 4,  D // 4 + D // 4
        y0, y1 = 1,        H // 2
        mat[x0:x1, y0:y1, z0:z1] = WHITE
        x0, x1 = W // 2 + W // 8, W // 2 + W // 8 + W // 5
        z0, z1 = D // 8,            D // 8 + D // 4
        y0, y1 = 1,                  H // 4
        mat[x0:x1, y0:y1, z0:z1] = WHITE
    return mat


def compute_normals(materials: np.ndarray) -> np.ndarray:
    """Per-cell outward surface normal from material adjacency.

    For each cell, look at its 6 neighbors. Each EMPTY-neighbor direction
    contributes to the outward normal. Sum those direction vectors and
    normalize. EMPTY cells (and SOLID interior cells with no empty
    neighbors) get a zero normal (it doesn't matter — they don't scatter).
    """
    W, H, D = materials.shape
    normals = np.zeros((W, H, D, 3), dtype=np.float32)

    # Direction unit vectors (must match the LPV gather order).
    dirs = np.array([
        [ 1, 0, 0], [-1, 0, 0],
        [ 0, 1, 0], [ 0,-1, 0],
        [ 0, 0, 1], [ 0, 0,-1],
    ], dtype=np.float32)

    pad = np.full((W+2, H+2, D+2), -1, dtype=np.int64)  # -1 = "outside" (treat as solid)
    pad[1:-1, 1:-1, 1:-1] = materials

    # For each direction d, the neighbor at offset d. Cells whose neighbor
    # in direction d is EMPTY get +d added to their normal.
    for d in dirs:
        dx, dy, dz = int(d[0]), int(d[1]), int(d[2])
        nb = pad[1+dx:1+dx+W, 1+dy:1+dy+H, 1+dz:1+dz+D]
        is_empty_neighbor = (nb == EMPTY)
        normals[..., 0] += dx * is_empty_neighbor.astype(np.float32)
        normals[..., 1] += dy * is_empty_neighbor.astype(np.float32)
        normals[..., 2] += dz * is_empty_neighbor.astype(np.float32)

    # Normalize. Where length == 0, leave as zero (interior solid cells
    # or empty cells; they don't scatter).
    lengths = np.linalg.norm(normals, axis=-1, keepdims=True)
    lengths = np.where(lengths > 0, lengths, 1.0)
    normals = normals / lengths
    return normals


# ── LPV propagation ────────────────────────────────────────────

def gather_incoming_sh(outgoing_sh: torch.Tensor) -> torch.Tensor:
    """LPV gather: build incoming SH at each cell from its 6 neighbors.

    For a neighbor at direction d_n from us (e.g. +X neighbor at higher x):
      flux F = (sqrt(pi)/2)·neighbor.c0 + sqrt(pi/3)·(d_face · neighbor.c_vec)
      where d_face = -d_n is the face's outward normal from the
      neighbor's perspective (pointing toward us).
      The flux enters our cell as a delta beam in direction d_face
      (the traveling direction at our cell).
      Project as delta:
        incoming.c0 += F · K0
        incoming.c_y += F · K1 · d_face_y
        incoming.c_z += F · K1 · d_face_z
        incoming.c_x += F · K1 · d_face_x

    outgoing_sh: (W, H, D, 4, 3) — neighbor cells' outgoing SH
    returns: (W, H, D, 4, 3) — incoming SH at each cell
    """
    W, H, D, _, _ = outgoing_sh.shape
    device = outgoing_sh.device

    # Neighbor offsets and face directions.
    # d_n: direction from us to neighbor; d_face: from neighbor toward us = -d_n.
    # For each neighbor offset, we slice the outgoing tensor accordingly.
    incoming = torch.zeros_like(outgoing_sh)

    # We'll roll outgoing_sh by -d_n to align: incoming[r,c,...] gets
    # contribution from neighbor at r+d_n.
    # For LPV, contribution from neighbor at offset (dx, dy, dz):
    #   F = sqrt(pi)/2 · neighbor.c0 + sqrt(pi/3) · (d_face · neighbor.c_vec)
    #   where d_face = -(dx, dy, dz)
    #   incoming += F · Y(d_face)

    # Direction list and corresponding d_face vectors as torch tensors.
    dirs_offset = [
        ( 1, 0, 0), (-1, 0, 0),
        ( 0, 1, 0), ( 0,-1, 0),
        ( 0, 0, 1), ( 0, 0,-1),
    ]

    # Decompose outgoing_sh's bands for the per-channel formula:
    #   c0 = outgoing_sh[..., 0, :]
    #   cy = outgoing_sh[..., 1, :]
    #   cz = outgoing_sh[..., 2, :]
    #   cx = outgoing_sh[..., 3, :]
    # F = sqrt(pi)/2 · c0 + sqrt(pi/3) · (d_face_x·cx + d_face_y·cy + d_face_z·cz)

    # Energy-conservation correction: when 6 axis-aligned faces tile the
    # sphere, the sum of cosine-weighted flux integrals across 6 hemispheres
    # over-counts the sphere integral by 3/2 (the average of |dx|+|dy|+|dz|
    # over the unit sphere). Scale the SH delta projection by 2/3 so an
    # EMPTY cell preserves total flux (input flux = output flux).
    LPV_PROJ = 2.0 / 3.0

    for (dx, dy, dz) in dirs_offset:
        src_x_lo = max(0,  dx); src_x_hi = W + min(0, dx)
        src_y_lo = max(0,  dy); src_y_hi = H + min(0, dy)
        src_z_lo = max(0,  dz); src_z_hi = D + min(0, dz)
        dst_x_lo = max(0, -dx); dst_x_hi = W + min(0, -dx)
        dst_y_lo = max(0, -dy); dst_y_hi = H + min(0, -dy)
        dst_z_lo = max(0, -dz); dst_z_hi = D + min(0, -dz)

        nb = outgoing_sh[src_x_lo:src_x_hi, src_y_lo:src_y_hi, src_z_lo:src_z_hi]

        c0_nb = nb[..., 0, :]
        cy_nb = nb[..., 1, :]
        cz_nb = nb[..., 2, :]
        cx_nb = nb[..., 3, :]

        dfx, dfy, dfz = -dx, -dy, -dz
        F = (SQRT_PI / 2.0) * c0_nb + SQRT_PI_OVER_3 * (
            dfx * cx_nb + dfy * cy_nb + dfz * cz_nb
        )

        incoming[dst_x_lo:dst_x_hi, dst_y_lo:dst_y_hi, dst_z_lo:dst_z_hi, 0] += F * (LPV_PROJ * K0)
        incoming[dst_x_lo:dst_x_hi, dst_y_lo:dst_y_hi, dst_z_lo:dst_z_hi, 1] += F * (LPV_PROJ * K1 * dfy)
        incoming[dst_x_lo:dst_x_hi, dst_y_lo:dst_y_hi, dst_z_lo:dst_z_hi, 2] += F * (LPV_PROJ * K1 * dfz)
        incoming[dst_x_lo:dst_x_hi, dst_y_lo:dst_y_hi, dst_z_lo:dst_z_hi, 3] += F * (LPV_PROJ * K1 * dfx)

    return incoming


def step_lego_sh(materials, normals, incoming_sh, model):
    W, H, D = materials.shape
    flat_mat  = materials.flatten()
    flat_norm = normals.view(W * H * D, 3)
    flat_inc  = incoming_sh.view(W * H * D, N_SH, N_CHANNELS)
    with torch.no_grad():
        flat_out = model(flat_mat, flat_norm, flat_inc)
    return flat_out.view(W, H, D, N_SH, N_CHANNELS)


def step_symbolic_sh(materials, normals, incoming_sh, sc_albedo, sc_emission_color):
    """Pure-torch symbolic SH step (reference implementation)."""
    W, H, D = materials.shape
    mat_flat = materials.flatten()                  # (HWD,)
    inc_flat = incoming_sh.view(W*H*D, N_SH, N_CHANNELS)
    nrm_flat = normals.view(W*H*D, 3)

    is_empty = (mat_flat == EMPTY)
    is_light = (mat_flat == LIGHT)

    # Irradiance E(n) = (sqrt(pi)/2)·c0 - sqrt(pi/3)·(n · c_vec) per channel
    c0 = inc_flat[:, 0]                              # (HWD, 3)
    cy = inc_flat[:, 1]
    cz = inc_flat[:, 2]
    cx = inc_flat[:, 3]
    nx = nrm_flat[:, 0:1]; ny = nrm_flat[:, 1:2]; nz = nrm_flat[:, 2:3]
    E = (SQRT_PI / 2.0) * c0 - SQRT_PI_OVER_3 * (cx*nx + cy*ny + cz*nz)  # (HWD, 3)

    albedo   = sc_albedo[mat_flat]                    # (HWD, 3)
    emit_col = sc_emission_color[mat_flat]            # (HWD, 3)

    # Solid outgoing: (albedo · E / pi) · hemisphere_SH(n)
    # hemisphere_SH(n) = (sqrt(pi), sqrt(3*pi)/2 · n_axis)
    # so out.c0 = albedo·E/sqrt(pi), out.c_axis = albedo·E·K1·n_axis
    common = albedo * E                                # (HWD, 3)
    solid_out = torch.zeros_like(inc_flat)
    solid_out[:, 0] = common / SQRT_PI
    solid_out[:, 1] = common * K1 * ny
    solid_out[:, 2] = common * K1 * nz
    solid_out[:, 3] = common * K1 * nx

    # Light outgoing: c0 = emit · sqrt(pi)/2, c_y = -emit · sqrt(pi/3), others = 0
    light_out = torch.zeros_like(inc_flat)
    light_out[:, 0] = emit_col * (SQRT_PI / 2.0)
    light_out[:, 1] = -emit_col * SQRT_PI_OVER_3

    is_empty_b = is_empty.view(-1, 1, 1).expand(-1, N_SH, N_CHANNELS)
    is_light_b = is_light.view(-1, 1, 1).expand(-1, N_SH, N_CHANNELS)

    out_flat = torch.where(is_empty_b, inc_flat,
               torch.where(is_light_b, light_out, solid_out))
    return out_flat.view(W, H, D, N_SH, N_CHANNELS)


def propagate(materials_np, normals_np, n_steps, model, device, use_lego=True):
    W, H, D = materials_np.shape
    materials = torch.tensor(materials_np, device=device, dtype=torch.long)
    normals   = torch.tensor(normals_np,   device=device, dtype=torch.float32)
    outgoing  = torch.zeros((W, H, D, N_SH, N_CHANNELS), device=device)

    sc_albedo = torch.zeros((N_MATERIALS, N_CHANNELS), device=device)
    sc_albedo[WHITE] = torch.tensor(ALBEDO_WHITE, device=device)
    sc_albedo[RED]   = torch.tensor(ALBEDO_RED,   device=device)
    sc_albedo[GREEN] = torch.tensor(ALBEDO_GREEN, device=device)
    sc_emission_color = torch.zeros((N_MATERIALS, N_CHANNELS), device=device)
    sc_emission_color[LIGHT] = torch.tensor(EMISSION_DOWN, device=device)

    for _ in range(n_steps):
        incoming = gather_incoming_sh(outgoing)
        if use_lego:
            outgoing = step_lego_sh(materials, normals, incoming, model)
        else:
            outgoing = step_symbolic_sh(materials, normals, incoming,
                                        sc_albedo, sc_emission_color)

    return outgoing.cpu().numpy(), materials.cpu().numpy()


# ── Renderer ────────────────────────────────────────────────────

def sh_evaluate_4(coefs: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Evaluate orthonormal SH order-1 at direction d, per channel.
    coefs: (..., 4, C) with bands [c_0, c_y, c_z, c_x].  d: (..., 3).
    Returns: (..., C). Clamped non-negative.
    """
    c0 = coefs[..., 0, :]
    cy = coefs[..., 1, :]
    cz = coefs[..., 2, :]
    cx = coefs[..., 3, :]
    rad = (K0 * c0
           + K1 * d[..., 1:2] * cy
           + K1 * d[..., 2:3] * cz
           + K1 * d[..., 0:1] * cx)
    return np.maximum(rad, 0.0)


def render_perspective(materials_np, outgoing_np, view_W=384, view_H=384,
                       cam_offset_z: float = -0.6, cam_offset_y: float = 0.05,
                       fov_deg: float = 55.0, max_march: int = 256):
    W, H, D = materials_np.shape
    cam = np.array([W * 0.5, H * (0.5 + cam_offset_y), W * cam_offset_z], dtype=np.float32)
    target = np.array([W * 0.5, H * 0.5, D * 0.5], dtype=np.float32)
    forward = target - cam; forward /= np.linalg.norm(forward)
    up0 = np.array([0, 1, 0], dtype=np.float32)
    right = np.cross(up0, forward); right /= np.linalg.norm(right)
    up = np.cross(forward, right); up /= np.linalg.norm(up)

    aspect = view_W / view_H
    half_h = np.tan(np.radians(fov_deg) / 2)
    half_w = aspect * half_h

    py, px = np.mgrid[0:view_H, 0:view_W]
    u = (px + 0.5) / view_W * 2 - 1
    v = (py + 0.5) / view_H * 2 - 1
    ray_dir = (forward[None, None, :]
               + u[..., None] * half_w * right[None, None, :]
               - v[..., None] * half_h * up[None, None, :])
    ray_dir /= np.linalg.norm(ray_dir, axis=-1, keepdims=True)
    ray_dir = ray_dir.reshape(-1, 3)
    N = ray_dir.shape[0]

    pos = np.broadcast_to(cam, (N, 3)).copy()
    img = np.zeros((N, 3), dtype=np.float32)
    alive = np.ones(N, dtype=bool)
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
                cell_sh = outgoing_np[
                    ix_safe[hit_idx], iy_safe[hit_idx], iz_safe[hit_idx]
                ]                                          # (n_hit, 4, 3)
                back = -ray_dir[hit_idx]                   # (n_hit, 3)
                img[hit_idx] = sh_evaluate_4(cell_sh, back)
                alive &= ~hit

        pos += ray_dir * step

    return img.reshape(view_H, view_W, 3)


def tonemap_to_png(arr: np.ndarray, path: str, exposure: float = 1.0):
    a = np.clip(arr * exposure, 0, None)
    a = a / (a + 1.0)
    a = np.power(a, 1/2.2)
    a = (np.clip(a, 0, 1) * 255).astype(np.uint8)
    from PIL import Image
    Image.fromarray(a).save(path)
    print(f"  saved {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--W", type=int, default=32)
    ap.add_argument("--H", type=int, default=32)
    ap.add_argument("--D", type=int, default=32)
    ap.add_argument("--steps", type=int, default=256)
    ap.add_argument("--view-W", type=int, default=384)
    ap.add_argument("--view-H", type=int, default=384)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--ckpt", default="checkpoints/specialists/light_sh_step.pt")
    ap.add_argument("--no-boxes", action="store_true")
    ap.add_argument("--exposure", type=float, default=1.0)
    args = ap.parse_args()

    print(f"Device: {args.device}")
    ck = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    model = LightSHStepMLP(**ck["config"]).to(args.device)
    model.load_state_dict(ck["model"])
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Lego: LightSHStepMLP, {n_params} params, SH order-1")
    print()

    materials = build_cornell_3d(args.W, args.H, args.D, with_boxes=not args.no_boxes)
    normals = compute_normals(materials)
    print(f"Cornell-3D scene: {args.W}×{args.H}×{args.D} = "
          f"{args.W*args.H*args.D:,} voxels, {args.steps} propagation steps")
    n_solid_with_normal = ((np.linalg.norm(normals, axis=-1) > 0)
                           & (materials != EMPTY)).sum()
    print(f"  cells with non-zero normal: {n_solid_with_normal:,}")
    print()

    sync = (torch.mps.synchronize if args.device == "mps"
            else torch.cuda.synchronize if args.device == "cuda" else None)

    if sync: sync()
    t0 = time.time()
    out_lego, mat_np = propagate(materials, normals, args.steps, model, args.device, use_lego=True)
    if sync: sync()
    t_lego = time.time() - t0

    if sync: sync()
    t0 = time.time()
    out_sym, _ = propagate(materials, normals, args.steps, model, args.device, use_lego=False)
    if sync: sync()
    t_sym = time.time() - t0

    print(f"Propagation:")
    print(f"  Lego                  : {t_lego*1000:>8.1f} ms   max={out_lego.max():.3f}  mean={out_lego.mean():.4f}")
    print(f"  Symbolic (torch ref)  : {t_sym*1000:>8.1f} ms   max={out_sym.max():.3f}  mean={out_sym.mean():.4f}")
    diff = np.abs(out_lego - out_sym)
    print(f"  max diff Lego vs sym  : {diff.max():.4f}, mean: {diff.mean():.5f}")
    print()

    print("Rendering perspective view…")
    t0 = time.time()
    img_lego = render_perspective(mat_np, out_lego, args.view_W, args.view_H)
    img_sym  = render_perspective(mat_np, out_sym,  args.view_W, args.view_H)
    print(f"  render time: {(time.time()-t0)*1000:.1f} ms")
    print()

    tonemap_to_png(img_lego, "cornell3d_sh_lego.png", args.exposure)
    tonemap_to_png(img_sym,  "cornell3d_sh_symbolic.png", args.exposure)


if __name__ == "__main__":
    main()
