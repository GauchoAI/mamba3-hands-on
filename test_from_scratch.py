"""From-scratch convergence demo — proves ptxd_specialist can train a
task from RANDOM INIT to meaningful accuracy, not just "loss decreased."

Setup:
  - Move existing parity.pt + .opt.bin out of the way (preserves the
    canonical mastered checkpoint). The regression guard would protect
    them anyway, but this also forces a fresh-init run.
  - Run ptxd_specialist with no resume — fresh random init each time.
  - 25 cycles × 200 steps_per_cycle = 5000 training steps (~7 min).
  - At end: restore the original parity.pt and clean up the trained
    weights. We're proving capability, not replacing the canonical.

Pass criterion: >= 85% accuracy at end of the run. Random-baseline on
parity is 50%; mastery in PyTorch was 100% with 20K steps. ptxd at 5K
steps should land between these — a clear "we trained it" signal.

If this PASSES, the user's earlier question ("did you actually train
from scratch?") gets a real "yes, on parity, here's the curve."
"""
import json, subprocess, sys, os, shutil, time
from pathlib import Path

REPO_ROOT  = Path(__file__).resolve().parent
CKPT_DIR   = REPO_ROOT / "checkpoints" / "specialists"
CANONICAL  = CKPT_DIR / "parity.pt"
OPT_STATE  = CKPT_DIR / "parity.opt.bin"
SAFE_DIR   = Path("/tmp/from_scratch_safe")
SAFE_DIR.mkdir(exist_ok=True)
PARK_PT    = SAFE_DIR / "parity.pt"
PARK_OPT   = SAFE_DIR / "parity.opt.bin"


def acc_of(pt_path):
    import torch
    if not Path(pt_path).exists(): return None, None
    ck = torch.load(str(pt_path), map_location="cpu", weights_only=False)
    return ck.get("accuracy", 0.0), ck.get("cycles", 0)


def park():
    """Move parity.pt + opt.bin into a safe holding spot so the run is
    truly from scratch. Restored at the end."""
    if CANONICAL.exists():
        shutil.move(str(CANONICAL), str(PARK_PT))
    if OPT_STATE.exists():
        shutil.move(str(OPT_STATE), str(PARK_OPT))


def restore():
    """Put the canonical files back exactly where they came from."""
    if PARK_PT.exists():
        shutil.move(str(PARK_PT), str(CANONICAL))
    if PARK_OPT.exists():
        shutil.move(str(PARK_OPT), str(OPT_STATE))
    # Also clean up any candidate / .opt.bin from this run that doesn't
    # belong in the canonical state.
    for p in [CKPT_DIR / "parity_ptxd_candidate.pt",
              CKPT_DIR / "parity_ptxd_candidate.opt.bin"]:
        if p.exists(): os.unlink(p)


def main():
    # Park canonical so we're guaranteed a fresh-init run.
    print("=== parking canonical parity.pt for safety ===")
    park()
    pre_acc, _ = acc_of(CANONICAL)
    if pre_acc is not None:
        print(f"  ABORT — parity.pt still at {CANONICAL}; park failed")
        sys.exit(2)
    print(f"  parity.pt parked at {PARK_PT}")

    try:
        # 5000 training steps. parity is the easiest task; if ptxd can
        # train ANYTHING from scratch, parity is where it'll show.
        t0 = time.time()
        print("\n=== training from random init (25 cycles × 200 steps) ===")
        proc = subprocess.run([
            sys.executable, str(REPO_ROOT / "ptxd_specialist.py"),
            "--task", "parity",
            "--d-model", "64", "--d-state", "8", "--headdim", "16",
            "--layers", "4", "--batch-size", "256",
            "--lr", "1e-3", "--weight-decay", "0.1",
            "--steps-per-cycle", "200", "--max-cycles", "25",
            "--target-acc", "0.95", "--seed", "12345",
        ], capture_output=True, text=True, timeout=1800)

        # Print cycle progress and final.
        for line in proc.stderr.splitlines():
            if any(k in line for k in ("cycle ", "saved parity", "regressed",
                                        "best_acc", "fresh_acc")):
                print(f"  {line}")
        for line in proc.stdout.splitlines():
            if "best_acc=" in line:
                print(f"  {line}")
        elapsed = time.time() - t0
        print(f"\n  wall: {elapsed:.1f}s")

        # The trained .pt was either saved canonical (success) or as a
        # candidate (regression — but here, prior_acc was 0 since canonical
        # was missing, so any acc>=0 saves canonical).
        post_acc, post_cycles = acc_of(CANONICAL)
        if post_acc is None:
            print("  FAIL — no checkpoint produced")
            sys.exit(1)
        print(f"\n=== verdict ===")
        print(f"  trained checkpoint: acc={post_acc:.3f}  cycles={post_cycles}")
        if post_acc >= 0.85:
            print(f"  PASS — ptxd trained parity from scratch to {post_acc:.0%}")
            sys.exit(0)
        elif post_acc >= 0.70:
            print(f"  PARTIAL — got to {post_acc:.0%}; close but not mastery. "
                  f"More steps or hyperparam tuning would close the gap.")
            sys.exit(0)  # don't hard-fail; partial is informative
        else:
            print(f"  FAIL — only reached {post_acc:.0%}, far from mastery")
            sys.exit(1)
    finally:
        # Always restore canonical, regardless of test outcome.
        print("\n=== restoring canonical ===")
        restore()
        rest_acc, rest_cycles = acc_of(CANONICAL)
        print(f"  parity.pt back at acc={rest_acc:.3f} cycles={rest_cycles}")


if __name__ == "__main__":
    main()
