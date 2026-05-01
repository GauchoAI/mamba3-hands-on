"""all_validate — single command, three byte-for-byte cross-checks.

Runs fish_validate (HANOIBIN), fib_validate (FIB-unary), and
fib_decimal_validate (FIB-decimal) end-to-end on their default
checkpoints. Exits non-zero if any of them fail.

This is the gate: when working on the LoopCounter / SSM core, before
declaring anything iron-solid, run `python all_validate.py`. If it
prints "ALL PASS" with three sub-results, the recurrence + the three
oracle channels (counter, scalar iter_token, per-position iter_token)
are working byte-for-byte vs Python reference.

Usage:
  python all_validate.py
  python all_validate.py --fibd-max-n 200  (push FIBD harder)
"""
import argparse, subprocess, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parent


def run(label, cmd, env=None):
    """Run a child Python validator. Returns (rc, captured_lines)."""
    print(f"\n=== {label} ===", flush=True)
    print(f"$ {' '.join(cmd[1:])}", flush=True)
    t0 = time.time()
    res = subprocess.run(cmd, capture_output=True, text=True)
    dt = time.time() - t0
    out = res.stdout + res.stderr
    # Print last ~12 lines (the summary block)
    tail = "\n".join(out.strip().splitlines()[-12:])
    print(tail, flush=True)
    print(f"({dt:.1f}s, rc={res.returncode})", flush=True)
    return res.returncode, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hanoi-pt", default="checkpoints/specialists/tower_of_hanoi_binary_eosgate_v2.pt")
    ap.add_argument("--fib-pt",   default="checkpoints/specialists/fib_unary.pt")
    ap.add_argument("--fibd-pt",  default="checkpoints/specialists/fib_decimal.pt")
    ap.add_argument("--fibd-max-n", type=int, default=40)
    ap.add_argument("--fib-max-n",  type=int, default=20)
    ap.add_argument("--device", default=None,
                    help="Device override (mps/cpu); pass through to validators")
    args = ap.parse_args()

    py = sys.executable
    failures = []

    rc, _ = run("HANOIBIN (fish_validate)", [
        py, str(REPO / "fish_validate.py"),
        "--pt", args.hanoi_pt,
    ] + (["--device", args.device] if args.device else []))
    if rc != 0:
        failures.append("HANOIBIN")

    rc, _ = run("FIB unary (fib_validate)", [
        py, str(REPO / "fib_validate.py"),
        "--pt", args.fib_pt,
        "--max-n", str(args.fib_max_n),
    ] + (["--device", args.device] if args.device else []))
    if rc != 0:
        failures.append("FIB-unary")

    rc, _ = run("FIB decimal (fib_decimal_validate)", [
        py, str(REPO / "fib_decimal_validate.py"),
        "--pt", args.fibd_pt,
        "--max-n", str(args.fibd_max_n),
    ] + (["--device", args.device] if args.device else []))
    if rc != 0:
        failures.append("FIB-decimal")

    print("\n" + "=" * 50)
    if not failures:
        print("ALL PASS — HANOIBIN + FIB-unary + FIB-decimal")
        return 0
    print(f"FAIL: {', '.join(failures)}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
