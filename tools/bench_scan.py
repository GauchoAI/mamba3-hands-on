#!/usr/bin/env python3
"""Benchmark SSM scan backends across devices.

Usage:
    python bench_scan.py                    # all backends, all devices
    python bench_scan.py --backend native   # specific backend
"""

import sys
import time
import torch
sys.path.insert(0, ".")

from mamba_platform.ssm_triton import ssm_scan_jit
from mamba_platform.ssm_scan_native import ssm_scan_native, ssm_scan_compiled


def bench_one(fn, args, warmup=5, iters=50):
    """Benchmark a single function. Returns ms per call."""
    for _ in range(warmup):
        fn(*args)
    if args[0].is_cuda:
        torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(iters):
        fn(*args)
    if args[0].is_cuda:
        torch.cuda.synchronize()
    elapsed = (time.time() - t0) / iters * 1000
    return elapsed


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default=None)
    parser.add_argument("-B", type=int, default=32)
    parser.add_argument("-L", type=int, default=64)
    args = parser.parse_args()

    B, L, H, hD, dS = args.B, args.L, 8, 16, 16
    print(f"Benchmark: B={B}, L={L}, H={H}, hD={hD}, dS={dS}")
    print(f"{'Backend':12s} {'Device':8s} {'Fwd (ms)':>10s} {'Tokens/s':>12s}")
    print("-" * 48)

    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        devices.append("mps")

    backends = {
        "jit": ssm_scan_jit,
        "native": ssm_scan_native,
        "compiled": ssm_scan_compiled,
    }
    try:
        from mamba_platform.ssm_triton import ssm_scan_triton
        backends["triton"] = ssm_scan_triton
    except Exception:
        pass

    if args.backend:
        backends = {args.backend: backends[args.backend]}

    for name, fn in backends.items():
        for device in devices:
            if name == "triton" and device != "cuda":
                continue
            torch.manual_seed(42)
            inp = torch.randn(B, L, H, hD, dS, device=device)
            decay = torch.sigmoid(torch.randn(B, L, H, device=device))
            C = torch.randn(B, L, H, dS, device=device)
            x = torch.randn(B, L, H, hD, device=device)
            z = torch.randn(B, L, H, hD, device=device)
            D = torch.randn(H, device=device)
            a = (inp, decay, C, x, z, D)

            try:
                ms = bench_one(fn, a)
                tps = B * L / (ms / 1000)
                print(f"{name:12s} {device:8s} {ms:>9.2f}ms {tps:>11,.0f}")
            except Exception as e:
                print(f"{name:12s} {device:8s}    ERROR: {e}")


if __name__ == "__main__":
    main()
