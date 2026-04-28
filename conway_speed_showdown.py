"""conway_speed_showdown — naive Python vs NumPy vs neural-batched Conway.

The honest benchmark. Three implementations of the SAME Conway rule
applied to a 1000×1000 toroidal grid for many generations:

  1. naive_python  — nested loops, the worst case
  2. numpy_conv    — vectorized via np.roll neighbor sum + boolean rule
  3. neural_batch  — torch conv for neighbor counts, then our 134-param
                     ConwayStepMLP forward over all 1M cells in ONE pass

All three must agree byte-for-byte on the final grid.

What we're testing: where our tiny Lego beats software, where it doesn't,
and *why*. The Lego is the same MLP — the orchestrator changes.
"""
import argparse, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
from train_conway_step import ConwayStepMLP


# ── Implementations ─────────────────────────────────────────────

def naive_python(grid: np.ndarray, n_gens: int) -> np.ndarray:
    """The thing software-engineering interview answers look like."""
    g = grid.tolist()
    h = len(g); w = len(g[0])
    for _ in range(n_gens):
        out = [[0] * w for _ in range(h)]
        for r in range(h):
            for c in range(w):
                n = 0
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        n += g[(r + dr) % h][(c + dc) % w]
                alive = g[r][c]
                if alive and n in (2, 3):
                    out[r][c] = 1
                elif not alive and n == 3:
                    out[r][c] = 1
        g = out
    return np.array(g, dtype=np.uint8)


def numpy_conv(grid: np.ndarray, n_gens: int) -> np.ndarray:
    """Hand-tuned NumPy: 8 rolls summed, boolean rule. The thing your
    co-worker writes after one cup of coffee."""
    g = grid.astype(np.uint8).copy()
    for _ in range(n_gens):
        n = (
            np.roll(g, ( 1, 0), (0, 1)) + np.roll(g, (-1,  0), (0, 1)) +
            np.roll(g, ( 0, 1), (0, 1)) + np.roll(g, ( 0, -1), (0, 1)) +
            np.roll(g, ( 1, 1), (0, 1)) + np.roll(g, ( 1, -1), (0, 1)) +
            np.roll(g, (-1, 1), (0, 1)) + np.roll(g, (-1, -1), (0, 1))
        )
        g = ((n == 3) | ((g == 1) & (n == 2))).astype(np.uint8)
    return g


def neural_batch(grid: np.ndarray, n_gens: int, model, device) -> np.ndarray:
    """Neural batched: one MLP forward per generation, over ALL cells.

    Uses a torch 2D conv to count neighbors (the "tool"), then feeds
    every (alive, n_neighbors) cell-state through the 134-param Lego
    in one shot.

    The Lego doesn't know it's processing a million cells. It just
    sees a million 2-int states and emits a million 0/1 actions.
    """
    h, w = grid.shape
    g = torch.tensor(grid, dtype=torch.long, device=device)
    # Neighbor-count kernel (torch, on the same device).
    kernel = torch.tensor(
        [[1, 1, 1], [1, 0, 1], [1, 1, 1]],
        dtype=torch.float32, device=device,
    ).view(1, 1, 3, 3)

    for _ in range(n_gens):
        # Count neighbors (toroidal: pad with wrap via F.pad).
        gf = g.to(torch.float32).view(1, 1, h, w)
        gp = F.pad(gf, (1, 1, 1, 1), mode="circular")
        n = F.conv2d(gp, kernel).view(h, w).to(torch.long)

        # Feed (alive, n) for all H*W cells through the MLP.
        states = torch.stack([g.flatten(), n.flatten()], dim=-1)
        with torch.no_grad():
            logits = model(states)
        g = logits.argmax(-1).view(h, w)

    return g.to(torch.uint8).cpu().numpy()


# ── Driver ──────────────────────────────────────────────────────

def time_it(fn, *a, sync=None, **kw):
    if sync is not None:
        sync()
    t0 = time.time()
    out = fn(*a, **kw)
    if sync is not None:
        sync()
    return out, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=1000, help="grid is size×size")
    ap.add_argument("--gens", type=int, default=100)
    ap.add_argument("--naive-size", type=int, default=200,
                    help="separate (smaller) grid for the naive Python run")
    ap.add_argument("--naive-gens", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = ap.parse_args()

    print(f"Device: {args.device}")
    print()

    # Load the trained 134-param Lego.
    ck = torch.load("checkpoints/specialists/conway_step.pt",
                    map_location=args.device, weights_only=False)
    model = ConwayStepMLP(**ck["config"]).to(args.device)
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"Lego: ConwayStepMLP, {sum(p.numel() for p in model.parameters())} params")
    print()

    rng = np.random.default_rng(args.seed)

    sync = None
    if args.device == "mps":
        sync = torch.mps.synchronize
    elif args.device == "cuda":
        sync = torch.cuda.synchronize

    # ── Round 1: small grid, all three implementations ───────────
    print(f"── Round 1: {args.naive_size}×{args.naive_size} grid, "
          f"{args.naive_gens} generations (all three) ──")
    small = rng.integers(0, 2, (args.naive_size, args.naive_size), dtype=np.uint8)

    out_naive,  t_naive  = time_it(naive_python, small, args.naive_gens)
    out_numpy,  t_numpy  = time_it(numpy_conv,   small, args.naive_gens)
    out_neural, t_neural = time_it(neural_batch, small, args.naive_gens,
                                   model=model, device=args.device, sync=sync)

    cells = args.naive_size * args.naive_size * args.naive_gens
    rate = lambda t: cells / t / 1e6
    print(f"  naive_python  : {t_naive*1000:>9.1f} ms   ({rate(t_naive):>7.2f} M cell-gens/s)")
    print(f"  numpy_conv    : {t_numpy*1000:>9.1f} ms   ({rate(t_numpy):>7.2f} M cell-gens/s)")
    print(f"  neural_batch  : {t_neural*1000:>9.1f} ms   ({rate(t_neural):>7.2f} M cell-gens/s)")

    same_nu = np.array_equal(out_naive, out_numpy)
    same_ne = np.array_equal(out_naive, out_neural)
    print(f"  naive == numpy ? {'✓' if same_nu else '✗'}")
    print(f"  naive == neural? {'✓' if same_ne else '✗'}")
    print()
    print(f"  neural beats naive  by  {t_naive/t_neural:>6.1f}×")
    print(f"  numpy  beats naive  by  {t_naive/t_numpy:>6.1f}×")
    print(f"  numpy  vs   neural      "
          f"{'numpy ' + str(round(t_neural/t_numpy, 1)) + '× faster' if t_numpy < t_neural else 'neural ' + str(round(t_numpy/t_neural, 1)) + '× faster'}")
    print()

    # ── Round 2: big grid, NumPy vs neural only ──────────────────
    print(f"── Round 2: {args.size}×{args.size} grid, {args.gens} generations "
          f"(numpy vs neural) ──")
    big = rng.integers(0, 2, (args.size, args.size), dtype=np.uint8)

    out_numpy,  t_numpy  = time_it(numpy_conv,   big, args.gens)
    out_neural, t_neural = time_it(neural_batch, big, args.gens,
                                   model=model, device=args.device, sync=sync)

    cells = args.size * args.size * args.gens
    print(f"  numpy_conv    : {t_numpy*1000:>9.1f} ms   ({cells/t_numpy/1e6:>7.2f} M cell-gens/s)")
    print(f"  neural_batch  : {t_neural*1000:>9.1f} ms   ({cells/t_neural/1e6:>7.2f} M cell-gens/s)")

    same = np.array_equal(out_numpy, out_neural)
    print(f"  numpy == neural? {'✓' if same else '✗'}")
    print()
    if t_neural < t_numpy:
        print(f"  → neural beats numpy by  {t_numpy/t_neural:>6.1f}×")
    else:
        print(f"  → numpy beats neural by  {t_neural/t_numpy:>6.1f}×")
    print()
    print(f"Same {sum(p.numel() for p in model.parameters())}-param Lego ran "
          f"{cells:,} cell-decisions across both rounds.")


if __name__ == "__main__":
    main()
