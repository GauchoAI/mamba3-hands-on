"""hanoi_solve_gru — solver-mode test for the order-invariant GRU.

If the GRU truly generalizes, it should solve Hanoi optimally at any n
within n_max_pad, including n's never seen during training.
"""
import argparse, time
import numpy as np
import torch
import torch.nn.functional as F

from discover_hanoi_invariant import HanoiInvariantGRU
from discover_hanoi_roles_mixed import (
    legal_action_mask, ACTION_PAIRS, N_ACTIONS,
)


def load_gru(path: str, device: str):
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = ck["config"]
    model = HanoiInvariantGRU(d_hidden=cfg["d_hidden"], n_layers=cfg["n_layers"]).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck["n_max_pad"]


def solve_n(n: int, model, n_max_pad: int, device: str, max_steps_factor: int = 4):
    pegs = np.array([0] * n_max_pad, dtype=np.int64)
    for i in range(n, n_max_pad):
        pegs[i] = -1
    optimal = (1 << n) - 1
    max_steps = optimal * max_steps_factor + 16
    step_count = 0
    visited = set()

    while step_count < max_steps:
        if all(pegs[i] == 2 for i in range(n)):
            return {"solved": True, "steps": step_count, "optimal": optimal}
        st_tuple = tuple(pegs[:n])
        if st_tuple in visited:
            return {"solved": False, "steps": step_count, "optimal": optimal,
                    "reason": "cycle"}
        visited.add(st_tuple)

        state_2d = pegs[None, :]
        legal = legal_action_mask(state_2d, n_max_pad)
        legal_t = torch.tensor(legal, device=device)
        states_t = torch.tensor(state_2d, device=device)
        with torch.no_grad():
            logits = model(states_t).masked_fill(~legal_t, -1e9)
            action = int(logits.argmax(-1).item())
        src, dst = ACTION_PAIRS[action]
        disk = next(i for i in range(n_max_pad) if pegs[i] == src)
        pegs[disk] = dst
        step_count += 1
    return {"solved": False, "steps": step_count, "optimal": optimal,
            "reason": "max_steps"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/hanoi_invariant_gru.pt")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n-list", type=int, nargs="+",
                    default=[2, 5, 10, 15, 17, 20, 22, 23])
    args = ap.parse_args()
    print(f"Device: {args.device}")
    model, n_max_pad = load_gru(args.ckpt, args.device)
    print(f"Loaded GRU, n_max_pad={n_max_pad}\n")
    print(f"{'n':>3} | {'optimal':>10} | {'achieved':>10} | {'time':>8} | {'verdict'}")
    print("-" * 60)
    for n in args.n_list:
        if n + 1 > n_max_pad:
            print(f"  n={n} skipped (exceeds n_max_pad={n_max_pad})")
            continue
        t0 = time.time()
        r = solve_n(n, model, n_max_pad, args.device)
        dt = time.time() - t0
        if r["solved"]:
            verdict = "✓ optimal" if r["steps"] == r["optimal"] else f"✓ {r['steps']/r['optimal']:.2f}x"
        else:
            verdict = f"✗ {r['reason']}"
        print(f"{n:>3} | {r['optimal']:>10} | {r['steps']:>10} | {dt:>7.1f}s | {verdict}")


if __name__ == "__main__":
    main()
