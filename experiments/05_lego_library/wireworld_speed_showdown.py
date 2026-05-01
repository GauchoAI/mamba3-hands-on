"""wireworld_speed_showdown — naive Python vs NumPy vs neural-batched.

WireWorld has 4 states with branchy per-cell rules. NumPy can still
vectorize, but now it needs multiple boolean masks + a where-cascade.
The neural batch handles everything in one MLP forward.

  1. naive_python  — nested loops + branches per cell
  2. numpy_branch  — vectorized via 4 boolean masks + np.where cascade
  3. neural_batch  — torch conv for head-counts, ONE MLP forward over all cells

All three must produce identical grids.
"""
import argparse, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
from train_wireworld_step import WireWorldStepMLP

EMPTY, CONDUCTOR, HEAD, TAIL = 0, 1, 2, 3


# ── Implementations ─────────────────────────────────────────────

def naive_python(grid: np.ndarray, n_gens: int) -> np.ndarray:
    g = grid.tolist()
    h = len(g); w = len(g[0])
    for _ in range(n_gens):
        out = [[0] * w for _ in range(h)]
        for r in range(h):
            for c in range(w):
                cell = g[r][c]
                if cell == EMPTY:
                    out[r][c] = EMPTY
                elif cell == HEAD:
                    out[r][c] = TAIL
                elif cell == TAIL:
                    out[r][c] = CONDUCTOR
                else:  # CONDUCTOR
                    n = 0
                    for dr in (-1, 0, 1):
                        for dc in (-1, 0, 1):
                            if dr == 0 and dc == 0:
                                continue
                            if g[(r + dr) % h][(c + dc) % w] == HEAD:
                                n += 1
                    out[r][c] = HEAD if (n == 1 or n == 2) else CONDUCTOR
        g = out
    return np.array(g, dtype=np.uint8)


def numpy_branch(grid: np.ndarray, n_gens: int) -> np.ndarray:
    """Vectorized but branchy: 4 boolean masks + np.where cascade."""
    g = grid.astype(np.uint8).copy()
    for _ in range(n_gens):
        is_head = (g == HEAD).astype(np.uint8)
        # neighbor head count via 8 rolls
        n = (
            np.roll(is_head, ( 1,  0), (0, 1)) + np.roll(is_head, (-1,  0), (0, 1)) +
            np.roll(is_head, ( 0,  1), (0, 1)) + np.roll(is_head, ( 0, -1), (0, 1)) +
            np.roll(is_head, ( 1,  1), (0, 1)) + np.roll(is_head, ( 1, -1), (0, 1)) +
            np.roll(is_head, (-1,  1), (0, 1)) + np.roll(is_head, (-1, -1), (0, 1))
        )
        # Branch cascade
        out = np.where(g == EMPTY, EMPTY,
              np.where(g == HEAD, TAIL,
              np.where(g == TAIL, CONDUCTOR,
              # CONDUCTOR: head if n in (1,2) else conductor
              np.where((n == 1) | (n == 2), HEAD, CONDUCTOR))))
        g = out.astype(np.uint8)
    return g


def neural_batch(grid: np.ndarray, n_gens: int, model, device) -> np.ndarray:
    """Conv to count head-neighbors, then MLP forward over all cells."""
    h, w = grid.shape
    g = torch.tensor(grid, dtype=torch.long, device=device)
    kernel = torch.tensor(
        [[1, 1, 1], [1, 0, 1], [1, 1, 1]],
        dtype=torch.float32, device=device,
    ).view(1, 1, 3, 3)

    for _ in range(n_gens):
        is_head = (g == HEAD).to(torch.float32).view(1, 1, h, w)
        is_head_p = F.pad(is_head, (1, 1, 1, 1), mode="circular")
        n = F.conv2d(is_head_p, kernel).view(h, w).to(torch.long)

        states = torch.stack([g.flatten(), n.flatten()], dim=-1)
        with torch.no_grad():
            logits = model(states)
        g = logits.argmax(-1).view(h, w)

    return g.to(torch.uint8).cpu().numpy()


# ── Driver ──────────────────────────────────────────────────────

def time_it(fn, *a, sync=None, **kw):
    if sync is not None: sync()
    t0 = time.time()
    out = fn(*a, **kw)
    if sync is not None: sync()
    return out, time.time() - t0


def make_grid(rng, h, w):
    """Random grid with all 4 states represented (~25% each).
    Seeds enough heads to keep activity going."""
    return rng.integers(0, 4, (h, w), dtype=np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=1000)
    ap.add_argument("--gens", type=int, default=100)
    ap.add_argument("--naive-size", type=int, default=200)
    ap.add_argument("--naive-gens", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = ap.parse_args()

    print(f"Device: {args.device}")
    print()

    ck = torch.load("checkpoints/specialists/wireworld_step.pt",
                    map_location=args.device, weights_only=False)
    model = WireWorldStepMLP(**ck["config"]).to(args.device)
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"Lego: WireWorldStepMLP, {sum(p.numel() for p in model.parameters())} params")
    print()

    rng = np.random.default_rng(args.seed)
    sync = (torch.mps.synchronize if args.device == "mps"
            else torch.cuda.synchronize if args.device == "cuda" else None)

    # ── Round 1: small grid, all three ───────────────────────────
    print(f"── Round 1: {args.naive_size}×{args.naive_size}, "
          f"{args.naive_gens} gens (all three) ──")
    small = make_grid(rng, args.naive_size, args.naive_size)

    out_naive, t_naive   = time_it(naive_python, small, args.naive_gens)
    out_numpy, t_numpy   = time_it(numpy_branch, small, args.naive_gens)
    out_neural, t_neural = time_it(neural_batch, small, args.naive_gens,
                                   model=model, device=args.device, sync=sync)

    cells = args.naive_size * args.naive_size * args.naive_gens
    rate = lambda t: cells / t / 1e6
    print(f"  naive_python  : {t_naive*1000:>9.1f} ms   ({rate(t_naive):>7.2f} M cell-gens/s)")
    print(f"  numpy_branch  : {t_numpy*1000:>9.1f} ms   ({rate(t_numpy):>7.2f} M cell-gens/s)")
    print(f"  neural_batch  : {t_neural*1000:>9.1f} ms   ({rate(t_neural):>7.2f} M cell-gens/s)")
    print(f"  naive == numpy ? {'✓' if np.array_equal(out_naive, out_numpy) else '✗'}")
    print(f"  naive == neural? {'✓' if np.array_equal(out_naive, out_neural) else '✗'}")
    print()
    print(f"  neural beats naive  by  {t_naive/t_neural:>6.1f}×")
    print(f"  numpy  beats naive  by  {t_naive/t_numpy:>6.1f}×")
    if t_neural < t_numpy:
        print(f"  → neural beats numpy by  {t_numpy/t_neural:>6.1f}×")
    else:
        print(f"  → numpy beats neural by  {t_neural/t_numpy:>6.1f}×")
    print()

    # ── Round 2: big grid, NumPy vs neural only ──────────────────
    print(f"── Round 2: {args.size}×{args.size}, {args.gens} gens "
          f"(numpy vs neural) ──")
    big = make_grid(rng, args.size, args.size)

    out_numpy,  t_numpy  = time_it(numpy_branch, big, args.gens)
    out_neural, t_neural = time_it(neural_batch, big, args.gens,
                                   model=model, device=args.device, sync=sync)

    cells = args.size * args.size * args.gens
    print(f"  numpy_branch  : {t_numpy*1000:>9.1f} ms   ({cells/t_numpy/1e6:>7.2f} M cell-gens/s)")
    print(f"  neural_batch  : {t_neural*1000:>9.1f} ms   ({cells/t_neural/1e6:>7.2f} M cell-gens/s)")
    print(f"  numpy == neural? {'✓' if np.array_equal(out_numpy, out_neural) else '✗'}")
    print()
    if t_neural < t_numpy:
        print(f"  → neural beats numpy by  {t_numpy/t_neural:>6.1f}×")
    else:
        print(f"  → numpy beats neural by  {t_neural/t_numpy:>6.1f}×")
    print()
    print(f"Same {sum(p.numel() for p in model.parameters())}-param Lego ran "
          f"{cells:,} cell-decisions in Round 2.")


if __name__ == "__main__":
    main()
