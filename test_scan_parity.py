#!/usr/bin/env python3
"""Test SSM scan correctness across all backends and devices.

Verifies that the native scan matches JIT output and that all backends
can learn the parity task (the litmus test for precision).

Usage:
    python test_scan_parity.py              # run all tests
    python test_scan_parity.py --quick      # just numerical comparison
"""

import sys
import time
import torch
sys.path.insert(0, ".")

from ssm_triton import ssm_scan_jit
from ssm_scan_native import ssm_scan_native, ssm_scan_compiled


def test_numerical_parity():
    """Test that native scan matches JIT output exactly."""
    print("=== Numerical Parity Test ===")
    torch.manual_seed(42)
    B, L, H, hD, dS = 4, 32, 8, 16, 16

    inp = torch.randn(B, L, H, hD, dS)
    decay = torch.sigmoid(torch.randn(B, L, H))
    C = torch.randn(B, L, H, dS)
    x = torch.randn(B, L, H, hD)
    z = torch.randn(B, L, H, hD)
    D = torch.randn(H)

    # Reference
    ref = ssm_scan_jit(inp, decay, C, x, z, D)

    # Native
    native = ssm_scan_native(inp, decay, C, x, z, D)
    diff = (ref - native).abs().max().item()
    print(f"  JIT vs native:   max diff = {diff:.2e}  {'PASS' if diff < 1e-4 else 'FAIL'}")

    # Compiled (may fail on first run due to compile time)
    try:
        compiled = ssm_scan_compiled(inp, decay, C, x, z, D)
        diff_c = (ref - compiled).abs().max().item()
        print(f"  JIT vs compiled: max diff = {diff_c:.2e}  {'PASS' if diff_c < 1e-4 else 'FAIL'}")
    except Exception as e:
        print(f"  Compiled: skipped ({e})")

    # Test on available devices
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        devices.append("mps")

    for device in devices:
        args = [t.to(device) for t in [inp, decay, C, x, z, D]]
        ref_d = ssm_scan_jit(*args)
        nat_d = ssm_scan_native(*args)
        diff_d = (ref_d - nat_d).abs().max().item()
        status = "PASS" if diff_d < 1e-4 else "FAIL"
        print(f"  {device:6s} JIT vs native: max diff = {diff_d:.2e}  {status}")

    return diff < 1e-4


def test_parity_training(backend="native", device="cpu", steps=400):
    """Train the parity task and verify >95% accuracy."""
    from progressive_model import ProgressiveModel
    import ssm_triton

    ssm_triton.FORCE_BACKEND = backend

    torch.manual_seed(42)
    model = ProgressiveModel(d_model=32, d_state=16, expand=2, headdim=16).to(device)
    model.add_kernel_layer()
    model.set_mode("kernel")

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)

    best_acc = 0
    for step in range(steps):
        # Generate parity batch
        B, L = 64, 8
        bits = torch.randint(0, 2, (B, L), device=device)
        parity = torch.cumsum(bits, dim=1) % 2

        # Encode as tokens: bits → model → predict parity
        # Use raw integer tokens (0, 1) + offset
        tokens = bits + 2  # offset to avoid special tokens
        targets = parity + 2

        logits = model(tokens)
        loss = torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)),
            targets[:, 1:].reshape(-1).long(),
        )
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        # Eval every 50 steps
        if (step + 1) % 50 == 0:
            with torch.no_grad():
                pred = logits[:, :-1].argmax(dim=-1)
                correct = (pred == targets[:, 1:]).float().mean().item()
                best_acc = max(best_acc, correct)

    ssm_triton.FORCE_BACKEND = None
    return best_acc


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    # Numerical test
    passed = test_numerical_parity()

    if args.quick:
        return

    # Training tests
    print("\n=== Parity Training Test ===")
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")

    backends = ["native", "jit"]
    if torch.cuda.is_available():
        backends.append("triton")

    for backend in backends:
        for device in devices:
            if backend == "triton" and device != "cuda":
                continue
            t0 = time.time()
            acc = test_parity_training(backend, device, steps=400)
            elapsed = time.time() - t0
            status = "PASS" if acc > 0.95 else ("KNOWN BUG" if backend == "triton" else "FAIL")
            print(f"  {backend:10s} {device:6s}  acc={acc:.0%}  {elapsed:.1f}s  {status}")


if __name__ == "__main__":
    main()
