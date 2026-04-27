"""fib_validate — full cross-check that the trained FIB model produces
exactly what Python's iterative Fibonacci would produce.

Mirrors fish_validate (HANOIBIN) but for FIB:
  - input "FIB n" -> output '1' * F(n)
  - oracle counter at SEP is F(n), which the oracle computes from
    the parsed n. The model only handles the loop body.

Suite 1: length-gen for n in [1, max_n].
Suite 2: counterfactual — feed FIB n_input but counter F(n_counter).
Suite 3: edge cases (n=0 input, n=1, far-OOD n).

Uses step-decode so even F(20)=6765 ones decodes in ~13s.
"""
import argparse, sys
from pathlib import Path

import torch
sys.path.insert(0, ".")
from progressive_model import ProgressiveModel
from step_decode import step_decode, BOS_TOKEN, SEP_TOKEN

EOS = 257


def fib(n: int) -> int:
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a


def python_reference(n: int) -> str:
    return "1" * fib(n)


def load_model(pt_path: str, device: str):
    ck = torch.load(pt_path, map_location=device, weights_only=False)
    cfg = ck["config"]; sd = ck["model"]
    has_lc = any(k.startswith("loop_counter.") for k in sd.keys())
    if not has_lc:
        raise SystemExit(f"checkpoint {pt_path} has no LoopCounter")
    lc_max = sd["loop_counter.c_emb.weight"].shape[0] - 2
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


def decode(model, n_input: int, n_counter_value: int, max_count: int, device: str):
    """n_counter_value is the actual counter (= F(n) at training; here we
    pass it explicitly so we can do counterfactual tests)."""
    inp = list(f"FIB {n_input}".encode("utf-8"))
    prefix = [BOS_TOKEN] + inp + [SEP_TOKEN]
    sep_pos = len(prefix) - 1
    out = step_decode(model, prefix, sep_pos, n_counter_value,
                      max_steps=max(n_counter_value * 2 + 8, 32),
                      device=device, max_count=max_count)
    return bytes(b for b in out if b < 256)


def suite_1_length_gen(model, max_n, max_count, device):
    fails = []
    for n in range(1, max_n + 1):
        fn = fib(n)
        actual = decode(model, n, fn, max_count, device)
        expected = python_reference(n).encode("ascii")
        if actual != expected:
            fails.append((n, fn, len(expected), len(actual)))
    return fails


def suite_2_counterfactual(model, max_count, device):
    """Same as HANOIBIN counterfactual: input says one n, counter says
    F(other_n). The output must follow F(other_n)."""
    pairs = [(5, 1), (5, 5), (5, 10), (5, 15),
             (1, 10), (10, 1),
             (3, 18), (18, 3),
             (10, 20), (20, 10)]
    fails = []
    for ni, nc in pairs:
        target_fn = fib(nc)
        actual = decode(model, ni, target_fn, max_count, device)
        expected = ("1" * target_fn).encode("ascii")
        if actual != expected:
            fails.append((ni, nc, target_fn, len(expected), len(actual)))
    return fails


def suite_3_edges(model, max_count, device):
    """n=0 (counter=F(0)=0, immediate stop), small/large counters."""
    cases = [0, 1, 2, 7, 10, 15, 20]
    fails = []
    for n in cases:
        fn = fib(n)
        actual = decode(model, n, fn, max_count, device)
        expected = python_reference(n).encode("ascii")
        if actual != expected:
            fails.append((n, fn, len(expected), len(actual)))
    return fails


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", default="checkpoints/specialists/fib_unary.pt")
    ap.add_argument("--max-n", type=int, default=20,
                    help="Max n for the length-gen sweep. F(20)=6765, F(25)=75025.")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = ap.parse_args()

    if not Path(args.pt).exists():
        raise SystemExit(f"checkpoint not found: {args.pt}")

    import time
    t_total = time.time()
    model, lc_max = load_model(args.pt, args.device)

    print(f"Model: {args.pt}")
    print(f"Counter table max: {lc_max}")
    print(f"Reference: FIB n -> '1' * F(n)  (step-decoder)")
    print()

    t = time.time()
    fails1 = suite_1_length_gen(model, args.max_n, lc_max, args.device)
    print(f"Suite 1: length-gen sweep n=1..{args.max_n} ({time.time()-t:.1f}s)")
    print(f"  {args.max_n - len(fails1)}/{args.max_n} match Python reference")
    for n, fn, expected_len, actual_len in fails1[:5]:
        print(f"  ✗ n={n} F(n)={fn}: expected len={expected_len}, got len={actual_len}")
    print()

    t = time.time()
    fails2 = suite_2_counterfactual(model, lc_max, args.device)
    n_total_2 = 10
    print(f"Suite 2: counterfactual control ({time.time()-t:.1f}s)")
    print(f"  {n_total_2 - len(fails2)}/{n_total_2} pass")
    for ni, nc, fn, expected_len, actual_len in fails2[:5]:
        print(f"  ✗ input=FIB {ni} counter=F({nc})={fn}: expected len={expected_len}, got len={actual_len}")
    print()

    t = time.time()
    fails3 = suite_3_edges(model, lc_max, args.device)
    n_total_3 = 7
    print(f"Suite 3: edge cases ({time.time()-t:.1f}s)")
    print(f"  {n_total_3 - len(fails3)}/{n_total_3} pass")
    for n, fn, expected_len, actual_len in fails3:
        print(f"  ✗ n={n} F(n)={fn}: expected len={expected_len}, got len={actual_len}")
    print()

    total_fails = len(fails1) + len(fails2) + len(fails3)
    print(f"Total wall: {time.time()-t_total:.1f}s")
    if total_fails == 0:
        print("FIB FISH PASS — model matches Python reference byte-for-byte.")
        return 0
    else:
        print(f"FIB FISH FAIL — {total_fails} mismatches.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
