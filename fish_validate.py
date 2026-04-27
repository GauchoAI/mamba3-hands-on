"""fish_validate — full cross-check that the trained HANOIBIN model
produces *exactly* what the Python reference would produce.

Iron-solid means: for every n in the supported range,
model.autoregressive_decode(n) == reference(n) byte-for-byte.

Three test suites:

  1. Length-gen sweep: n in [1, max_n] inclusive. The reference for
     n is 'n ones'. We compare bytes exactly.

  2. Counterfactual control: feed input n_input but counter trajectory
     for n_counter. Reference: n_counter ones. The model must follow
     the counter, not the input.

  3. Edge cases: n=0 (empty answer), n=1 (single token), n=max_n
     (table cap), and n_input != n_counter at the boundaries.

Uses the O(L) step decoder (mamba3 SSM hidden state cached per layer)
so the full sweep over n=1..256 finishes in seconds.

Exits with non-zero status if any test fails.
"""
import argparse, sys
from pathlib import Path

import torch
sys.path.insert(0, ".")
from progressive_model import ProgressiveModel, ByteTokenizer
from step_decode import step_decode, BOS_TOKEN, SEP_TOKEN

EOS = 257


def python_reference(n: int) -> str:
    """The single source of truth: HANOIBIN n -> '1' * n."""
    return "1" * n


def load_model(pt_path: str, device: str):
    ck = torch.load(pt_path, map_location=device, weights_only=False)
    cfg = ck["config"]
    sd = ck["model"]
    has_lc = any(k.startswith("loop_counter.") for k in sd.keys())
    lc_max = (sd["loop_counter.c_emb.weight"].shape[0] - 2
              if has_lc else 1024)
    if not has_lc:
        raise SystemExit(f"checkpoint {pt_path} has no LoopCounter — "
                         f"FISH validation is for the loop-counter model")
    model = ProgressiveModel(
        d_model=cfg["d_model"], d_state=cfg["d_state"],
        expand=2, headdim=cfg["headdim"],
        use_loop_counter=True, loop_counter_max=lc_max,
    )
    for _ in range(cfg["n_kernel_layers"]):
        model.add_kernel_layer()
    model.load_state_dict(sd)
    model.eval().to(device)
    return model, lc_max


def decode(model, n_input: int, n_counter: int, max_count: int, device: str):
    inp = list(f"HANOIBIN {n_input}".encode("utf-8"))
    prefix = [BOS_TOKEN] + inp + [SEP_TOKEN]
    sep_pos = len(prefix) - 1
    out = step_decode(model, prefix, sep_pos, n_counter,
                      max_steps=max(n_counter * 2 + 8, 32),
                      device=device, max_count=max_count)
    return bytes(b for b in out if b < 256)


def suite_1_length_gen(model, max_n, max_count, device):
    fails = []
    for n in range(1, max_n + 1):
        actual = decode(model, n, n, max_count, device)
        expected = python_reference(n).encode("ascii")
        if actual != expected:
            fails.append((n, expected, actual))
    return fails


def suite_2_counterfactual(model, max_count, device):
    pairs = [(20, 10), (10, 20), (5, 100), (100, 5), (50, 50),
             (15, 7), (7, 15), (3, 200), (200, 3),
             (1, 250), (250, 1), (100, 100), (150, 150),
             (256, 1), (1, 256), (0, 50), (50, 0)]
    fails = []
    for ni, nc in pairs:
        actual = decode(model, ni, nc, max_count, device)
        expected = python_reference(nc).encode("ascii")
        if actual != expected:
            fails.append((ni, nc, expected, actual))
    return fails


def suite_3_edges(model, max_count, device):
    cases = [0, 1, 2, max_count - 1, max_count]
    fails = []
    for n in cases:
        actual = decode(model, n, n, max_count, device)
        expected = python_reference(n).encode("ascii")
        if actual != expected:
            fails.append((n, expected, actual))
    return fails


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", default="checkpoints/specialists/tower_of_hanoi_binary_eosgate_v2.pt")
    ap.add_argument("--max-n", type=int, default=None,
                    help="Max n for the length-gen sweep. Defaults to "
                         "the loop_counter table cap.")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = ap.parse_args()

    if not Path(args.pt).exists():
        raise SystemExit(f"checkpoint not found: {args.pt}")

    import time
    t_total = time.time()
    model, lc_max = load_model(args.pt, args.device)
    max_n = args.max_n if args.max_n is not None else lc_max

    print(f"Model: {args.pt}")
    print(f"Counter table max: {lc_max}, sentinel: {lc_max + 1}")
    print(f"Reference: HANOIBIN n -> '1' * n  (step-decoder)")
    print()

    t = time.time()
    fails1 = suite_1_length_gen(model, max_n, lc_max, args.device)
    print(f"Suite 1: length-gen sweep n=1..{max_n} ({time.time()-t:.1f}s)")
    print(f"  {max_n - len(fails1)}/{max_n} match Python reference")
    for n, exp, act in fails1[:5]:
        print(f"  ✗ n={n}: expected len={len(exp)}, got len={len(act)}")
    print()

    t = time.time()
    fails2 = suite_2_counterfactual(model, lc_max, args.device)
    n_total_2 = 17
    print(f"Suite 2: counterfactual control ({time.time()-t:.1f}s)")
    print(f"  {n_total_2 - len(fails2)}/{n_total_2} pass")
    for ni, nc, exp, act in fails2[:5]:
        print(f"  ✗ input={ni} counter={nc}: expected len={len(exp)}, got len={len(act)}")
    print()

    t = time.time()
    fails3 = suite_3_edges(model, lc_max, args.device)
    n_total_3 = 5
    print(f"Suite 3: edge cases ({time.time()-t:.1f}s)")
    print(f"  {n_total_3 - len(fails3)}/{n_total_3} pass")
    for n, exp, act in fails3:
        print(f"  ✗ n={n}: expected len={len(exp)}, got len={len(act)}")
    print()

    total_fails = len(fails1) + len(fails2) + len(fails3)
    print(f"Total wall: {time.time()-t_total:.1f}s")
    if total_fails == 0:
        print("FISH PASS — model matches Python reference byte-for-byte across all suites.")
        return 0
    else:
        print(f"FISH FAIL — {total_fails} mismatches found.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
