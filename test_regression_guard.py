"""Regression-guard test — proves we can't accidentally clobber a
mastered checkpoint with a worse run.

Setup:
  - Start with parity.pt at 100%
  - Run ptxd_specialist with deliberately broken hyperparams (huge LR,
    a single short cycle) so the eval will regress
  - After the run, parity.pt MUST still be at 100%
  - A {task}_ptxd_candidate.pt MUST exist with the regressed weights
"""
import json, subprocess, sys, os, shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
CKPT_DIR  = REPO_ROOT / "checkpoints" / "specialists"
CANONICAL = CKPT_DIR / "parity.pt"
CANDIDATE = CKPT_DIR / "parity_ptxd_candidate.pt"
BACKUP    = Path("/tmp/parity.pt.backup")

def acc_of(pt_path):
    import torch
    ck = torch.load(str(pt_path), map_location="cpu", weights_only=False)
    return ck.get("accuracy", 0.0), ck.get("cycles", 0)


def main():
    # Restore canonical to known-good 100% state.
    if BACKUP.exists():
        shutil.copy(BACKUP, CANONICAL)
    if CANDIDATE.exists():
        os.unlink(CANDIDATE)
    # Clear opt state and any candidate opt.
    for p in [CKPT_DIR / "parity.opt.bin", CKPT_DIR / "parity_ptxd_candidate.opt.bin"]:
        if p.exists(): os.unlink(p)

    pre_acc, pre_cycles = acc_of(CANONICAL)
    print(f"=== before ===")
    print(f"  canonical: acc={pre_acc:.3f} cycles={pre_cycles}")
    if pre_acc < 0.99:
        print(f"  ABORT — canonical isn't at 100% to start, can't test regression cleanly")
        sys.exit(2)

    # Run with deliberately broken hyperparams — huge LR, fresh init bins
    # that hammer the model. Should regress hard.
    print("\n=== running ptxd_specialist with LR=1e-1 (deliberately too high) ===")
    proc = subprocess.run([
        sys.executable, str(REPO_ROOT / "ptxd_specialist.py"),
        "--task", "parity",
        "--d-model", "64", "--d-state", "8", "--headdim", "16",
        "--layers", "4", "--batch-size", "256",
        "--lr", "1e-1",   # ← way too high, should blow up the model
        "--steps-per-cycle", "100", "--max-cycles", "1",
    ], capture_output=True, text=True, timeout=180)
    # Show the relevant stderr lines
    for line in proc.stderr.splitlines():
        if any(k in line for k in ("regressed", "saved parity", "WARNING", "candidate")):
            print(f"  {line}")

    post_acc, post_cycles = acc_of(CANONICAL)
    print(f"\n=== after ===")
    print(f"  canonical: acc={post_acc:.3f} cycles={post_cycles}")
    if CANDIDATE.exists():
        cand_acc, cand_cycles = acc_of(CANDIDATE)
        print(f"  candidate: acc={cand_acc:.3f} cycles={cand_cycles}")
    else:
        print(f"  candidate: (none)")

    # Verdict.
    print(f"\n=== verdict ===")
    if abs(post_acc - pre_acc) < 1e-3 and post_cycles == pre_cycles:
        print(f"  PASS — canonical preserved at acc={post_acc:.3f} cycles={post_cycles}")
        if CANDIDATE.exists():
            print(f"         candidate written to {CANDIDATE.name} (acc={cand_acc:.3f}) for inspection")
        sys.exit(0)
    else:
        print(f"  FAIL — canonical was overwritten")
        print(f"         was acc={pre_acc:.3f} cycles={pre_cycles}")
        print(f"         now acc={post_acc:.3f} cycles={post_cycles}")
        sys.exit(1)


if __name__ == "__main__":
    main()
