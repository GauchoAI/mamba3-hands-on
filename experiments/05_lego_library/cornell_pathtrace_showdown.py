"""cornell_pathtrace_showdown — naive Python vs NumPy vs PyTorch-MPS.

Cornell box path tracer in three flavors. Same scene, same algorithm,
three implementations:

  1. naive Python  — per-pixel for-loop, the worst case
  2. NumPy         — vectorized: all rays / all bounces, masked
  3. PyTorch MPS   — same vectorization, on the GPU

This is the speed regime CAs couldn't reach: per-pixel work is heavy
(many primitive intersections + branchy material logic + variable path
length), exactly where vectorized NumPy starts to hurt and the GPU
should win.

Scene: 5 Lambertian walls (red left, green right, white floor/ceiling/back)
+ 1 emissive light on the ceiling. Cosine-weighted hemisphere sampling.
"""
import argparse, time
import numpy as np
import torch


# ── Scene ───────────────────────────────────────────────────────

WHITE = [0.75, 0.75, 0.75]
RED   = [0.75, 0.25, 0.25]
GREEN = [0.25, 0.75, 0.25]
BLACK = [0.0,  0.0,  0.0]
LIGHT = [15.0, 15.0, 15.0]

# (axis, value, lo, hi, normal, albedo, emission)
SCENE = [
    (1, 0.0,   [0,0,0], [1,0,1],            [ 0, 1, 0], WHITE, BLACK),  # Floor
    (1, 1.0,   [0,1,0], [1,1,1],            [ 0,-1, 0], WHITE, BLACK),  # Ceiling
    (2, 1.0,   [0,0,1], [1,1,1],            [ 0, 0,-1], WHITE, BLACK),  # Back
    (0, 0.0,   [0,0,0], [0,1,1],            [ 1, 0, 0], RED,   BLACK),  # Left  (red)
    (0, 1.0,   [1,0,0], [1,1,1],            [-1, 0, 0], GREEN, BLACK),  # Right (green)
    (1, 0.999, [0.4,0.999,0.4], [0.6,0.999,0.6], [0,-1,0], BLACK, LIGHT),  # Light
]
N_PRIMS = len(SCENE)


def make_basis(normal):
    n = np.asarray(normal, dtype=np.float32)
    t = np.array([1,0,0], dtype=np.float32) if abs(n[0]) < 0.9 else np.array([0,1,0], dtype=np.float32)
    t = t - n.dot(t) * n
    t = t / np.linalg.norm(t)
    b = np.cross(n, t).astype(np.float32)
    return np.stack([t, b, n], axis=1)


SCN_AXIS  = np.array([s[0] for s in SCENE], dtype=np.int64)
SCN_VAL   = np.array([s[1] for s in SCENE], dtype=np.float32)
SCN_LO    = np.array([s[2] for s in SCENE], dtype=np.float32)
SCN_HI    = np.array([s[3] for s in SCENE], dtype=np.float32)
SCN_N     = np.array([s[4] for s in SCENE], dtype=np.float32)
SCN_ALB   = np.array([s[5] for s in SCENE], dtype=np.float32)
SCN_EMS   = np.array([s[6] for s in SCENE], dtype=np.float32)
SCN_BASIS = np.array([make_basis(s[4]) for s in SCENE], dtype=np.float32)
CAM = np.array([0.5, 0.5, -1.5], dtype=np.float32)
FOV = 1.0
EPS = 1e-4


# ── Naive Python ────────────────────────────────────────────────

def trace_naive(origin, direction, max_bounces, rng):
    L = np.zeros(3, dtype=np.float32)
    throughput = np.ones(3, dtype=np.float32)
    for _ in range(max_bounces):
        best_t, best_i = float('inf'), -1
        for i in range(N_PRIMS):
            a = SCN_AXIS[i]
            denom = direction[a]
            if abs(denom) < 1e-9:
                continue
            t = (SCN_VAL[i] - origin[a]) / denom
            if t < EPS:
                continue
            hit = origin + t * direction
            ok = True
            for j in range(3):
                if j == a:
                    continue
                if hit[j] < SCN_LO[i, j] - EPS or hit[j] > SCN_HI[i, j] + EPS:
                    ok = False
                    break
            if ok and t < best_t:
                best_t, best_i = t, i
        if best_i == -1:
            break
        L += throughput * SCN_EMS[best_i]
        if SCN_EMS[best_i].sum() > 0:
            break
        hit = origin + best_t * direction
        origin = hit + EPS * SCN_N[best_i]
        r1, r2 = rng.random(), rng.random()
        phi = 2 * np.pi * r1
        st, ct = np.sqrt(r2), np.sqrt(1.0 - r2)
        local = np.array([np.cos(phi)*st, np.sin(phi)*st, ct], dtype=np.float32)
        direction = SCN_BASIS[best_i] @ local
        throughput = throughput * SCN_ALB[best_i]
    return L


def render_naive(W, H, spp, max_bounces, seed=0):
    rng = np.random.default_rng(seed)
    img = np.zeros((H, W, 3), dtype=np.float32)
    for y in range(H):
        for x in range(W):
            acc = np.zeros(3, dtype=np.float32)
            for _ in range(spp):
                jx, jy = rng.random(), rng.random()
                px, py = (x + jx) / W, (y + jy) / H
                d = np.array([(px - 0.5)*FOV, -(py - 0.5)*FOV, 1.0], dtype=np.float32)
                d /= np.linalg.norm(d)
                acc += trace_naive(CAM, d, max_bounces, rng)
            img[y, x] = acc / spp
    return img


# ── NumPy vectorized ────────────────────────────────────────────

def render_numpy(W, H, spp, max_bounces, seed=0):
    rng = np.random.default_rng(seed)
    N = H * W * spp

    h_idx = np.repeat(np.arange(H, dtype=np.float32), W * spp)
    w_idx = np.tile(np.repeat(np.arange(W, dtype=np.float32), spp), H)

    pix_rand = rng.random((N, 2)).astype(np.float32)
    px = (w_idx + pix_rand[:, 0]) / W
    py = (h_idx + pix_rand[:, 1]) / H

    directions = np.stack([(px - 0.5)*FOV, -(py - 0.5)*FOV, np.ones(N, dtype=np.float32)], axis=1)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)

    origins = np.broadcast_to(CAM, (N, 3)).copy()
    throughput = np.ones((N, 3), dtype=np.float32)
    L = np.zeros((N, 3), dtype=np.float32)
    alive = np.ones(N, dtype=bool)

    for _ in range(max_bounces):
        t_all = np.full((N, N_PRIMS), np.inf, dtype=np.float32)
        for i in range(N_PRIMS):
            a = int(SCN_AXIS[i])
            denom = directions[:, a]
            safe = np.where(np.abs(denom) > 1e-9, denom, 1e-9)
            t_i = (SCN_VAL[i] - origins[:, a]) / safe
            valid = (np.abs(denom) > 1e-9) & (t_i > EPS)
            hit = origins + t_i[:, None] * directions
            for j in range(3):
                if j == a:
                    continue
                valid &= (hit[:, j] >= SCN_LO[i, j] - EPS) & (hit[:, j] <= SCN_HI[i, j] + EPS)
            t_all[:, i] = np.where(valid, t_i, np.inf)

        hit_idx = np.argmin(t_all, axis=1)
        hit_t = np.take_along_axis(t_all, hit_idx[:, None], axis=1).squeeze(-1)
        no_hit = ~np.isfinite(hit_t)

        emission = SCN_EMS[hit_idx]
        albedo   = SCN_ALB[hit_idx]
        normal   = SCN_N[hit_idx]
        basis    = SCN_BASIS[hit_idx]

        contrib = alive & ~no_hit
        L += contrib[:, None] * throughput * emission

        is_em = (emission > 0).any(axis=1)
        alive = alive & ~no_hit & ~is_em
        if not alive.any():
            break

        hit_point = origins + hit_t[:, None] * directions
        new_origin = hit_point + EPS * normal

        br = rng.random((N, 2)).astype(np.float32)
        phi = 2 * np.pi * br[:, 0]
        st, ct = np.sqrt(br[:, 1]), np.sqrt(1.0 - br[:, 1])
        local = np.stack([np.cos(phi)*st, np.sin(phi)*st, ct], axis=1)
        new_dir = np.einsum('nij,nj->ni', basis, local)

        am = alive[:, None]
        origins    = np.where(am, new_origin, origins)
        directions = np.where(am, new_dir,    directions)
        throughput = np.where(am, throughput * albedo, throughput)

    return L.reshape(H, W, spp, 3).mean(axis=2)


# ── PyTorch / MPS ───────────────────────────────────────────────

def render_torch(W, H, spp, max_bounces, device, seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    N = H * W * spp

    sc_axis  = torch.tensor(SCN_AXIS,  dtype=torch.long,    device=device)
    sc_val   = torch.tensor(SCN_VAL,   dtype=torch.float32, device=device)
    sc_lo    = torch.tensor(SCN_LO,    dtype=torch.float32, device=device)
    sc_hi    = torch.tensor(SCN_HI,    dtype=torch.float32, device=device)
    sc_n     = torch.tensor(SCN_N,     dtype=torch.float32, device=device)
    sc_alb   = torch.tensor(SCN_ALB,   dtype=torch.float32, device=device)
    sc_ems   = torch.tensor(SCN_EMS,   dtype=torch.float32, device=device)
    sc_basis = torch.tensor(SCN_BASIS, dtype=torch.float32, device=device)
    cam      = torch.tensor(CAM,       dtype=torch.float32, device=device)

    h_idx = torch.arange(H, device=device, dtype=torch.float32).repeat_interleave(W * spp)
    w_idx = torch.arange(W, device=device, dtype=torch.float32).repeat_interleave(spp).repeat(H)

    pix_rand = torch.rand(N, 2, device=device, generator=g, dtype=torch.float32)
    px = (w_idx + pix_rand[:, 0]) / W
    py = (h_idx + pix_rand[:, 1]) / H

    directions = torch.stack([(px - 0.5)*FOV, -(py - 0.5)*FOV, torch.ones(N, device=device)], dim=1)
    directions = directions / directions.norm(dim=1, keepdim=True)

    origins    = cam.expand(N, 3).clone()
    throughput = torch.ones(N, 3, device=device)
    L          = torch.zeros(N, 3, device=device)
    alive      = torch.ones(N, dtype=torch.bool, device=device)

    INF = torch.tensor(float('inf'), device=device)
    SAFE = torch.tensor(1e-9, device=device)

    for _ in range(max_bounces):
        t_all = torch.full((N, N_PRIMS), float('inf'), device=device)
        for i in range(N_PRIMS):
            a = int(sc_axis[i].item())
            denom = directions[:, a]
            safe = torch.where(denom.abs() > 1e-9, denom, SAFE)
            t_i = (sc_val[i] - origins[:, a]) / safe
            valid = (denom.abs() > 1e-9) & (t_i > EPS)
            hit = origins + t_i.unsqueeze(-1) * directions
            for j in range(3):
                if j == a:
                    continue
                valid = valid & (hit[:, j] >= sc_lo[i, j] - EPS) & (hit[:, j] <= sc_hi[i, j] + EPS)
            t_all[:, i] = torch.where(valid, t_i, INF)

        hit_t, hit_idx = t_all.min(dim=1)
        no_hit = ~hit_t.isfinite()

        emission = sc_ems[hit_idx]
        albedo   = sc_alb[hit_idx]
        normal   = sc_n[hit_idx]
        basis    = sc_basis[hit_idx]

        contrib = alive & ~no_hit
        L = L + contrib.unsqueeze(-1).to(torch.float32) * throughput * emission

        is_em = (emission > 0).any(dim=1)
        alive = alive & ~no_hit & ~is_em
        if not alive.any():
            break

        hit_point  = origins + hit_t.unsqueeze(-1) * directions
        new_origin = hit_point + EPS * normal

        br = torch.rand(N, 2, device=device, generator=g, dtype=torch.float32)
        phi = 2 * torch.pi * br[:, 0]
        st, ct = br[:, 1].sqrt(), (1 - br[:, 1]).sqrt()
        local = torch.stack([phi.cos()*st, phi.sin()*st, ct], dim=1)
        new_dir = torch.einsum('nij,nj->ni', basis, local)

        am = alive.unsqueeze(-1)
        origins    = torch.where(am, new_origin, origins)
        directions = torch.where(am, new_dir,    directions)
        throughput = torch.where(am, throughput * albedo, throughput)

    return L.reshape(H, W, spp, 3).mean(dim=2).cpu().numpy()


# ── Driver ──────────────────────────────────────────────────────

def time_it(fn, *a, sync=None, **kw):
    if sync is not None: sync()
    t0 = time.time()
    out = fn(*a, **kw)
    if sync is not None: sync()
    return out, time.time() - t0


def save_image(arr, path):
    a = np.clip(arr, 0, None)
    a = a / (1 + a)             # Reinhard tonemap
    a = np.power(a, 1/2.2)      # gamma
    a = (np.clip(a, 0, 1) * 255).astype(np.uint8)
    try:
        from PIL import Image
        Image.fromarray(a).save(path)
        print(f"  saved {path}")
    except ImportError:
        print(f"  (PIL not installed; skipping save of {path})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bounces", type=int, default=3)
    ap.add_argument("--naive-W", type=int, default=32)
    ap.add_argument("--naive-spp", type=int, default=4)
    ap.add_argument("--mid-W", type=int, default=128)
    ap.add_argument("--mid-spp", type=int, default=16)
    ap.add_argument("--big-W", type=int, default=512)
    ap.add_argument("--big-spp", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()

    print(f"Device: {args.device}")
    print()

    sync = (torch.mps.synchronize if args.device == "mps"
            else torch.cuda.synchronize if args.device == "cuda" else None)

    # Round 1: small, all three
    Wn, Hn, sn = args.naive_W, args.naive_W, args.naive_spp
    print(f"── Round 1: {Wn}×{Hn}, {sn} spp, {args.bounces} bounces (all three) ──")
    img_naive, t_naive = time_it(render_naive, Wn, Hn, sn, args.bounces, args.seed)
    img_numpy, t_numpy = time_it(render_numpy, Wn, Hn, sn, args.bounces, args.seed)
    img_torch, t_torch = time_it(render_torch, Wn, Hn, sn, args.bounces, args.device, args.seed, sync=sync)
    rays = Wn * Hn * sn
    rate = lambda t: rays / t / 1e3
    print(f"  naive_python : {t_naive*1000:>10.1f} ms   ({rate(t_naive):>9.1f} k rays/s)")
    print(f"  numpy        : {t_numpy*1000:>10.1f} ms   ({rate(t_numpy):>9.1f} k rays/s)")
    print(f"  torch ({args.device:<3}) : {t_torch*1000:>10.1f} ms   ({rate(t_torch):>9.1f} k rays/s)")
    print(f"  numpy beats naive  by  {t_naive/t_numpy:>7.1f}×")
    print(f"  torch beats naive  by  {t_naive/t_torch:>7.1f}×")
    print(f"  torch beats numpy  by  {t_numpy/t_torch:>7.1f}×")
    print(f"  mean RGB: naive={img_naive.mean():.4f}, numpy={img_numpy.mean():.4f}, torch={img_torch.mean():.4f}")
    print()

    # Round 2: medium, NumPy vs Torch
    Wm, Hm, sm = args.mid_W, args.mid_W, args.mid_spp
    print(f"── Round 2: {Wm}×{Hm}, {sm} spp, {args.bounces} bounces (numpy vs torch) ──")
    img_numpy2, t_numpy2 = time_it(render_numpy, Wm, Hm, sm, args.bounces, args.seed)
    img_torch2, t_torch2 = time_it(render_torch, Wm, Hm, sm, args.bounces, args.device, args.seed, sync=sync)
    rays = Wm * Hm * sm
    print(f"  numpy        : {t_numpy2*1000:>10.1f} ms   ({rays/t_numpy2/1e6:>7.2f} M rays/s)")
    print(f"  torch ({args.device:<3}) : {t_torch2*1000:>10.1f} ms   ({rays/t_torch2/1e6:>7.2f} M rays/s)")
    print(f"  torch beats numpy by  {t_numpy2/t_torch2:>6.1f}×")
    print()

    # Round 3: big, torch only
    Wb, Hb, sb = args.big_W, args.big_W, args.big_spp
    print(f"── Round 3: {Wb}×{Hb}, {sb} spp, {args.bounces} bounces (torch only) ──")
    img_torch3, t_torch3 = time_it(render_torch, Wb, Hb, sb, args.bounces, args.device, args.seed, sync=sync)
    rays = Wb * Hb * sb
    print(f"  torch ({args.device:<3}) : {t_torch3*1000:>10.1f} ms   ({rays/t_torch3/1e6:>7.2f} M rays/s)")
    print()

    if args.save:
        save_image(img_torch3, "cornell_torch.png")
        save_image(img_numpy2, "cornell_numpy.png")
        save_image(img_naive,  "cornell_naive.png")


if __name__ == "__main__":
    main()
