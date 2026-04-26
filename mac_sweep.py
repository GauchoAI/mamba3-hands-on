"""mac_sweep — sequential MPS-backend sweep harness for specialist_trainer.

Runs one task at a time on M-series GPUs; specialist_trainer's
regression guard keeps a bad-seed re-train from clobbering a good ckpt.

Designed for the Apple Silicon cluster:
- One M4 Mac mini per process (no parallel — MPS shares one GPU per host).
- Adapt --tasks to slice the 37-task list across nodes.
- Dispatched by an outer SSH-shell harness (analogous to
  three_populations.py's subprocess fan-out).

Usage:
  python3 mac_sweep.py --quick                       # 6 representative tasks
  python3 mac_sweep.py                                # all 37 tasks
  python3 mac_sweep.py --tasks logic_gate modus_ponens
  python3 mac_sweep.py --max-cycles 15 --per-task-timeout 600
"""
import argparse, json, os, re, subprocess, sys, time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
LOG_DIR   = Path("/tmp/mac_sweep_logs")
LOG_DIR.mkdir(exist_ok=True)
OUT_PATH  = Path("/tmp/mac_sweep_summary.json")
CKPT_DIR  = REPO_ROOT / "checkpoints" / "specialists"

QUICK_TASKS = [
    "logic_gate", "modus_ponens", "duplicate_detect",
    "binary_pattern_next", "addition", "multiplication",
]


def acc_of(pt_path):
    """Read accuracy from a saved .pt without spinning a fresh torch session."""
    try:
        import torch
        ck = torch.load(str(pt_path), map_location="cpu", weights_only=False)
        return float(ck.get("accuracy", 0.0)), int(ck.get("cycles", 0))
    except Exception:
        return None, None


def run_task(task, args):
    log_path = LOG_DIR / f"{task}.log"
    cmd = [
        sys.executable, str(REPO_ROOT / "specialist_trainer.py"),
        "--task", task,
        "--device", args.device,
        "--d-model", str(args.d_model), "--d-state", str(args.d_state),
        "--headdim", str(args.headdim), "--layers", str(args.layers),
        "--batch-size", str(args.batch_size),
        "--lr", str(args.lr), "--weight-decay", str(args.weight_decay),
        "--steps-per-cycle", str(args.steps_per_cycle),
        "--max-cycles", str(args.max_cycles),
        "--target-acc", str(args.target_acc),
    ]
    t0 = time.time()
    with open(log_path, "w") as f:
        try:
            proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                                  timeout=args.per_task_timeout, text=True)
            rc = proc.returncode
        except subprocess.TimeoutExpired:
            rc = -1
    wall = time.time() - t0

    log_text = log_path.read_text()
    saved_match = re.search(r"Saved specialist.*\((\d+)%\)", log_text)
    guarded_match = re.search(r"REGRESSION GUARD: not saving", log_text)
    best_match = re.findall(r"best=(\d+)%", log_text)
    best = int(best_match[-1]) / 100 if best_match else None
    saved_acc = int(saved_match.group(1)) / 100 if saved_match else None

    pt_acc, pt_cycles = acc_of(CKPT_DIR / f"{task}.pt") if (CKPT_DIR / f"{task}.pt").exists() else (None, None)

    if pt_acc is None:
        verdict = "no_checkpoint"
    elif pt_acc >= 0.95:
        verdict = "mastered"
    elif pt_acc >= 0.90:
        verdict = "close"
    elif pt_acc >= 0.50:
        verdict = "partial"
    else:
        verdict = "stuck"

    return {
        "task": task,
        "verdict": verdict,
        "wall_s": round(wall, 1),
        "rc": rc,
        "best_acc": round(best, 3) if best is not None else None,
        "saved_acc": round(saved_acc, 3) if saved_acc is not None else None,
        "pt_acc": round(pt_acc, 3) if pt_acc is not None else None,
        "pt_cycles": pt_cycles,
        "guarded": bool(guarded_match),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="6-task representative subset")
    ap.add_argument("--tasks", nargs="+", help="Explicit task list (overrides --quick)")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--d-state", type=int, default=16)
    ap.add_argument("--headdim", type=int, default=16)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--steps-per-cycle", type=int, default=100)
    ap.add_argument("--max-cycles", type=int, default=10)
    ap.add_argument("--target-acc", type=float, default=0.95)
    ap.add_argument("--per-task-timeout", type=int, default=400,
                    help="Wall-time cap per task in seconds (default 400)")
    args = ap.parse_args()

    if args.tasks:
        tasks = args.tasks
    elif args.quick:
        tasks = QUICK_TASKS
    else:
        tasks = sorted(p.name for p in (REPO_ROOT / "problems").iterdir() if p.is_dir())

    print(f"Sweeping {len(tasks)} tasks on device={args.device}, "
          f"max_cycles={args.max_cycles}, per_task_timeout={args.per_task_timeout}s")
    t0 = time.time()
    results = {}
    for i, task in enumerate(tasks, 1):
        print(f"[{i:2d}/{len(tasks)}] {task} ...", flush=True)
        r = run_task(task, args)
        results[task] = r
        guarded_tag = "  (guarded)" if r["guarded"] else ""
        print(f"     verdict={r['verdict']:12s} wall={r['wall_s']:6.1f}s "
              f"pt_acc={r.get('pt_acc')!s:>5} best={r.get('best_acc')!s:>5}"
              f"{guarded_tag}", flush=True)
    total = time.time() - t0

    by_verdict = {}
    for r in results.values():
        by_verdict.setdefault(r["verdict"], []).append(r["task"])
    print(f"\n=== summary (total {total:.1f}s, sequential) ===")
    for verdict, names in sorted(by_verdict.items(), key=lambda kv: -len(kv[1])):
        print(f"  {verdict:14s} ({len(names):2d}): {', '.join(sorted(names))}")

    OUT_PATH.write_text(json.dumps({
        "total_wall_s": total,
        "device": args.device,
        "results": [results[t] for t in sorted(results)],
        "by_verdict": by_verdict,
    }, indent=2))
    print(f"\nSummary at {OUT_PATH}")


if __name__ == "__main__":
    main()
