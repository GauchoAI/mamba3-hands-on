"""Multi-task sweep — exercise every task in problems/ through
ptxd_specialist's curriculum mode and record outcomes.

This is the empirical baseline that should drive what kernel work
matters. Tasks that master cleanly: production-ready today. Tasks
that plateau: candidates for GA hyperparameter search OR kernel
fixes. Tasks that error: real bugs to chase.

Each task is run from scratch with a sensible default config
(d=64 L=2 dS=16 hd=16 batch=64 lr=1e-3) — not GA-tuned, just a
reasonable starting point. Existing canonical .pt is parked and
restored regardless of outcome so we don't damage the 82 trained
checkpoints.

Outputs:
  - Per-task line in stderr as it runs
  - JSON summary at /tmp/sweep_summary.json
"""
import json, os, shutil, subprocess, sys, time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
CKPT_DIR  = REPO_ROOT / "checkpoints" / "specialists"
PARK_DIR  = Path("/tmp/sweep_parked")
PARK_DIR.mkdir(exist_ok=True)
OUT_PATH  = Path("/tmp/sweep_summary.json")

# Per-task wall-time cap. With curriculum + auto-tuner bail, most tasks
# finish in 30-90s. Anything that runs longer than this is plateauing
# and we move on.
PER_TASK_TIMEOUT = 240  # 4 min


def acc_of(pt_path):
    import torch
    if not Path(pt_path).exists():
        return None, None
    ck = torch.load(str(pt_path), map_location="cpu", weights_only=False)
    return ck.get("accuracy", 0.0), ck.get("cycles", 0)


def park(task):
    """Move {task}.pt and {task}.opt.bin out of the way for a clean run."""
    parked = []
    for suf in [".pt", ".opt.bin"]:
        src = CKPT_DIR / f"{task}{suf}"
        if src.exists():
            dst = PARK_DIR / src.name
            shutil.move(str(src), str(dst))
            parked.append((src, dst))
    return parked


def restore(parked):
    for src, dst in parked:
        if dst.exists():
            shutil.move(str(dst), str(src))


def run_task(task):
    parked = park(task)
    try:
        t0 = time.time()
        proc = subprocess.run([
            sys.executable, str(REPO_ROOT / "ptxd_specialist.py"),
            "--task", task,
            "--d-model", "64", "--d-state", "16", "--headdim", "16",
            "--layers", "2", "--batch-size", "64",
            "--lr", "1e-3", "--weight-decay", "0.1",
            "--steps-per-cycle", "200", "--max-cycles", "25",
            "--target-acc", "0.95", "--seed", "42",
        ], capture_output=True, text=True, timeout=PER_TASK_TIMEOUT)
        wall = time.time() - t0

        # Read the saved .pt's reported accuracy. ptxd_specialist's
        # regression guard means a successful save is what landed.
        post_acc, post_cycles = acc_of(CKPT_DIR / f"{task}.pt")
        # Stage transitions are the most useful signal — count them.
        stages_cleared = 0
        max_stage_acc = 0.0
        for line in proc.stderr.splitlines():
            if "✓ stage" in line and "cleared" in line:
                stages_cleared += 1
            if "best_acc=" in line:
                # Try to parse the per-stage best_acc lines.
                try:
                    chunk = line.split("best_acc=")[1].split(",")[0]
                    max_stage_acc = max(max_stage_acc, float(chunk))
                except Exception:
                    pass

        verdict = "ok"
        if proc.returncode != 0:
            verdict = "exit_nonzero"
        elif post_acc is None:
            verdict = "no_checkpoint"
        elif post_acc >= 0.95:
            verdict = "mastered"
        elif post_acc >= 0.80:
            verdict = "close"
        elif post_acc >= 0.60:
            verdict = "partial"
        else:
            verdict = "plateaued"

        result = {
            "task": task,
            "verdict": verdict,
            "wall_s": round(wall, 1),
            "best_acc": round(post_acc, 3) if post_acc is not None else None,
            "cycles": post_cycles,
            "stages_cleared": stages_cleared,
            "max_stage_acc": round(max_stage_acc, 3),
            "exit_code": proc.returncode,
        }
        return result
    except subprocess.TimeoutExpired:
        return {"task": task, "verdict": "timeout",
                "wall_s": PER_TASK_TIMEOUT, "best_acc": None,
                "cycles": 0, "stages_cleared": 0, "max_stage_acc": 0.0,
                "exit_code": -1}
    except Exception as e:
        return {"task": task, "verdict": "error", "error": str(e),
                "wall_s": 0, "best_acc": None, "cycles": 0,
                "stages_cleared": 0, "max_stage_acc": 0.0, "exit_code": -2}
    finally:
        restore(parked)


def main():
    tasks = sorted(p.name for p in (REPO_ROOT / "problems").iterdir() if p.is_dir())
    print(f"Sweeping {len(tasks)} tasks. Per-task timeout: {PER_TASK_TIMEOUT}s.\n")
    results = []
    for i, task in enumerate(tasks, 1):
        print(f"[{i:2d}/{len(tasks)}] {task} ...", flush=True)
        r = run_task(task)
        results.append(r)
        print(f"     verdict={r['verdict']:12s} wall={r['wall_s']:6.1f}s  "
              f"best_acc={r.get('best_acc')!s:>6}  cycles={r.get('cycles', 0)}  "
              f"stages={r['stages_cleared']}/{(r.get('max_stage_acc',0)>0 and 'pos') or 0}",
              flush=True)

    # Aggregate
    by_verdict = {}
    for r in results:
        by_verdict.setdefault(r["verdict"], []).append(r["task"])

    print("\n=== summary ===")
    for verdict, names in sorted(by_verdict.items(), key=lambda kv: -len(kv[1])):
        print(f"  {verdict:12s} ({len(names):2d}): {', '.join(names)}")

    OUT_PATH.write_text(json.dumps({
        "results": results,
        "by_verdict": {k: v for k, v in by_verdict.items()},
    }, indent=2))
    print(f"\nFull summary at {OUT_PATH}")


if __name__ == "__main__":
    main()
