"""Parallel multi-task sweep — runs N ptxd_specialist invocations
concurrently. Default 4-way: ~4× wall-time speedup over serial.

Each spawned ptxd_specialist parks the canonical {task}.pt before
training (curriculum mode), restores it after. Different tasks use
different files so there's no cross-task interference. The H100's
~80GB HBM easily holds 4 small models concurrently.

Usage:
  python3 test_parallel_sweep.py             # all 37 tasks, 4-way parallel
  python3 test_parallel_sweep.py --quick     # 6 representative tasks
  python3 test_parallel_sweep.py --parallel 8  # 8-way
"""
import argparse, json, os, shutil, subprocess, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
CKPT_DIR  = REPO_ROOT / "checkpoints" / "specialists"
PARK_DIR  = Path("/tmp/parsweep_parked")
PARK_DIR.mkdir(exist_ok=True)
OUT_PATH  = Path("/tmp/parallel_sweep_summary.json")

PER_TASK_TIMEOUT = 600  # 10 min cap. Curriculum mode runs each stage with the
# FULL --max-cycles budget (n_stages × max_cycles × steps_per_cycle worst case)
# and relies on per-stage early-exit to keep wall time bounded. At parallel=4
# with 4-stage curricula, the prior 240s cap was tight enough that even
# previously-fast tasks (logic_gate solo: 33s) hit it under GPU contention.
# Override at the CLI with --per-task-timeout if you need longer.

# Representative subset for iterating on changes — covers a mastered, a
# mid-tier, and a stuck task from the prior sweep.
QUICK_TASKS = [
    "logic_gate",          # mastered fast (33s)
    "modus_ponens",        # mastered fast (80s)
    "duplicate_detect",    # close-call (0.925)
    "binary_pattern_next", # mid-tier (0.735)
    "addition",            # stuck (0.23)
    "multiplication",      # very stuck (0.075)
]


def acc_of(pt_path):
    import torch
    if not Path(pt_path).exists():
        return None, None
    ck = torch.load(str(pt_path), map_location="cpu", weights_only=False)
    return ck.get("accuracy", 0.0), ck.get("cycles", 0)


def run_task(task, kd_weight=0.0, per_task_timeout=PER_TASK_TIMEOUT):
    """Run a single task. Park canonical, train fresh, restore.

    Returns a dict with task / verdict / wall_s / best_acc / etc.
    """
    # Park PER-TASK files. With 4-way parallel, each task touches only
    # its own files so no inter-task collision.
    parked = []
    for suf in [".pt", ".opt.bin"]:
        src = CKPT_DIR / f"{task}{suf}"
        if src.exists():
            dst = PARK_DIR / f"{task}{suf}"
            shutil.move(str(src), str(dst))
            parked.append((src, dst))

    try:
        t0 = time.time()
        cmd = [
            sys.executable, str(REPO_ROOT / "ptxd_specialist.py"),
            "--task", task,
            "--d-model", "64", "--d-state", "16", "--headdim", "16",
            "--layers", "2", "--batch-size", "64",
            "--lr", "1e-3", "--weight-decay", "0.1",
            "--steps-per-cycle", "200", "--max-cycles", "25",
            "--target-acc", "0.95", "--seed", "42",
        ]
        if kd_weight > 0 and parked:
            # Use the parked teacher .pt for distillation
            teacher_pt = str(parked[0][1])  # parked .pt path
            cmd.extend([
                "--kd-weight", str(kd_weight),
                "--kd-temperature", "3.0",
                "--teacher-pt", teacher_pt,
            ])

        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=per_task_timeout)
        wall = time.time() - t0
        post_acc, post_cycles = acc_of(CKPT_DIR / f"{task}.pt")
        stages_cleared = sum(1 for line in proc.stderr.splitlines()
                             if "✓ stage" in line and "cleared" in line)
        max_stage_acc = 0.0
        for line in proc.stderr.splitlines():
            if "best_acc=" in line:
                try:
                    chunk = line.split("best_acc=")[1].split(",")[0]
                    max_stage_acc = max(max_stage_acc, float(chunk))
                except Exception:
                    pass

        if proc.returncode != 0 and post_acc is None:
            verdict = "error"
        elif post_acc is None:
            verdict = "no_checkpoint"
        elif post_acc >= 0.95:
            verdict = "mastered"
        elif post_acc >= 0.92:
            verdict = "close"
        elif post_acc >= 0.60:
            verdict = "partial"
        else:
            verdict = "stuck"

        return {
            "task": task,
            "verdict": verdict,
            "wall_s": round(wall, 1),
            "best_acc": round(post_acc, 3) if post_acc is not None else None,
            "cycles": post_cycles,
            "stages_cleared": stages_cleared,
            "max_stage_acc": round(max_stage_acc, 3),
            "kd_weight": kd_weight,
        }
    except subprocess.TimeoutExpired:
        return {"task": task, "verdict": "timeout",
                "wall_s": per_task_timeout, "best_acc": None,
                "cycles": 0, "stages_cleared": 0, "max_stage_acc": 0.0,
                "kd_weight": kd_weight}
    except Exception as e:
        return {"task": task, "verdict": "error", "error": str(e),
                "wall_s": 0, "best_acc": None, "cycles": 0,
                "stages_cleared": 0, "max_stage_acc": 0.0, "kd_weight": kd_weight}
    finally:
        for src, dst in parked:
            if dst.exists():
                shutil.move(str(dst), str(src))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="Run the 6-task representative subset only")
    ap.add_argument("--parallel", type=int, default=4,
                    help="Number of concurrent tasks (default 4)")
    ap.add_argument("--kd-weight", type=float, default=0.0,
                    help="If >0, use the parked canonical .pt as teacher "
                         "for distillation (matches GA's resume-from-master "
                         "scenario but with fresh student weights)")
    ap.add_argument("--per-task-timeout", type=int, default=PER_TASK_TIMEOUT,
                    help=f"Wall-time cap per task in seconds (default {PER_TASK_TIMEOUT}). "
                         "Bump this if curriculum tasks at parallel=4 hit the cap.")
    args = ap.parse_args()

    if args.quick:
        tasks = QUICK_TASKS
    else:
        tasks = sorted(p.name for p in (REPO_ROOT / "problems").iterdir() if p.is_dir())

    print(f"Sweeping {len(tasks)} tasks with parallel={args.parallel}, "
          f"kd_weight={args.kd_weight}.")
    print(f"Per-task timeout: {args.per_task_timeout}s.\n")

    results = {}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.parallel) as pool:
        futures = {pool.submit(run_task, t, args.kd_weight, args.per_task_timeout): t for t in tasks}
        for i, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            results[r["task"]] = r
            print(f"[{i:2d}/{len(tasks)}] {r['task']:30s} "
                  f"verdict={r['verdict']:12s} wall={r['wall_s']:6.1f}s  "
                  f"best_acc={r.get('best_acc')!s:>6}  "
                  f"stages={r['stages_cleared']}", flush=True)
    total = time.time() - t0

    by_verdict = {}
    for r in results.values():
        by_verdict.setdefault(r["verdict"], []).append(r["task"])

    print(f"\n=== summary (total {total:.1f}s, {args.parallel}x parallel) ===")
    for verdict, names in sorted(by_verdict.items(), key=lambda kv: -len(kv[1])):
        print(f"  {verdict:12s} ({len(names):2d}): {', '.join(sorted(names))}")

    OUT_PATH.write_text(json.dumps({
        "total_wall_s": total,
        "parallel": args.parallel,
        "kd_weight": args.kd_weight,
        "results": [results[t] for t in sorted(results)],
        "by_verdict": by_verdict,
    }, indent=2))
    print(f"\nFull summary at {OUT_PATH}")


if __name__ == "__main__":
    main()
