"""sweep_and_diff — run the parallel sweep, diff vs prior, emit only on
verdict transitions. This is the foundation of the recurring/scheduled
task; cron this every N hours and silence is success.

Wire:
  1. Reads the prior summary from /tmp/parallel_sweep_summary.json (kept
     by test_parallel_sweep, format: {results: [...], by_verdict: {...}}).
  2. Saves a copy as /tmp/parallel_sweep_prev.json before running.
  3. Runs test_parallel_sweep.py with the requested args.
  4. Diffs new vs prev and prints one line per CHANGED task. Mastered
     stays silent unless it regressed; stuck stays silent unless it
     improved.
  5. Exits 0 on no-change, 1 on regression, 2 on improvement-only. The
     distinct codes let cron / systemd hooks decide whether to notify.

CLI mirrors test_parallel_sweep.py so this is a drop-in replacement:
    python3 sweep_and_diff.py --quick --parallel 4 --kd-weight 0.0
    python3 sweep_and_diff.py --parallel 4 --per-task-timeout 600
"""
import argparse, json, os, shutil, subprocess, sys, time
from pathlib import Path

REPO_ROOT     = Path(__file__).resolve().parent
SUMMARY_PATH  = Path("/tmp/parallel_sweep_summary.json")
PREV_PATH     = Path("/tmp/parallel_sweep_prev.json")
HISTORY_DIR   = Path("/tmp/parallel_sweep_history")
HISTORY_DIR.mkdir(exist_ok=True)


# Verdict ordering: lower = worse. Used to decide regression vs improvement.
VERDICT_RANK = {
    "error":        0,
    "no_checkpoint": 0,
    "timeout":      1,
    "stuck":        2,
    "partial":      3,
    "close":        4,
    "mastered":     5,
}


def load_summary(path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception as e:
        sys.stderr.write(f"[sweep_diff] could not parse {path}: {e}\n")
        return None


def to_dict(summary):
    """Map task → result for easier diffing. Tolerant to schema drift."""
    if not summary or "results" not in summary:
        return {}
    return {r["task"]: r for r in summary["results"]}


def acc_delta(new_acc, old_acc):
    """Return signed delta string ('+0.05', '-0.10', '—' if either is None)."""
    if new_acc is None or old_acc is None:
        return "—"
    delta = new_acc - old_acc
    return f"{delta:+.3f}"


def diff_summaries(prev, curr):
    """Yield one (severity, line) per changed task.

    severity: 'regress' | 'improve' | 'flap' (verdict swap inside same band).
    Tasks present only in `curr` (e.g. brand new) are reported as 'new'.
    Tasks only in `prev` (e.g. removed problem dir) as 'removed'.
    """
    p = to_dict(prev)
    c = to_dict(curr)
    all_tasks = sorted(set(p) | set(c))
    for task in all_tasks:
        pr, cr = p.get(task), c.get(task)
        if pr and not cr:
            yield ("removed", f"  removed     {task}")
            continue
        if cr and not pr:
            yield ("new", f"  new         {task:30s} verdict={cr['verdict']:10s} acc={cr.get('best_acc')}")
            continue
        pv, cv = pr["verdict"], cr["verdict"]
        pa, ca = pr.get("best_acc"), cr.get("best_acc")
        if pv == cv:
            # same verdict — emit only if accuracy moved noticeably
            if pa is not None and ca is not None and abs(ca - pa) >= 0.05:
                sev = "regress" if ca < pa else "improve"
                yield (sev, f"  {sev:10s}  {task:30s} {pv} acc {pa:.3f} → {ca:.3f} ({acc_delta(ca,pa)})")
            continue
        # verdict changed
        pr_rank = VERDICT_RANK.get(pv, -1)
        cr_rank = VERDICT_RANK.get(cv, -1)
        if cr_rank > pr_rank:
            yield ("improve", f"  improve     {task:30s} {pv} → {cv}  acc {pa} → {ca}  ({acc_delta(ca,pa)})")
        elif cr_rank < pr_rank:
            yield ("regress", f"  REGRESS     {task:30s} {pv} → {cv}  acc {pa} → {ca}  ({acc_delta(ca,pa)})")
        else:
            yield ("flap", f"  flap        {task:30s} {pv} → {cv}  acc {pa} → {ca}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--kd-weight", type=float, default=0.0)
    ap.add_argument("--per-task-timeout", type=int, default=600)
    ap.add_argument("--keep-history", action="store_true",
                    help="Archive each summary into /tmp/parallel_sweep_history/")
    args = ap.parse_args()

    # Snapshot prior summary BEFORE the new run overwrites it.
    if SUMMARY_PATH.exists():
        shutil.copyfile(SUMMARY_PATH, PREV_PATH)
    prev = load_summary(PREV_PATH)

    cmd = [sys.executable, str(REPO_ROOT / "test_parallel_sweep.py"),
           "--parallel", str(args.parallel),
           "--per-task-timeout", str(args.per_task_timeout)]
    if args.quick:
        cmd.append("--quick")
    if args.kd_weight > 0:
        cmd.extend(["--kd-weight", str(args.kd_weight)])

    sys.stderr.write(f"[sweep_diff] launching: {' '.join(cmd)}\n")
    t0 = time.time()
    rc = subprocess.call(cmd)
    wall = time.time() - t0
    sys.stderr.write(f"[sweep_diff] sweep wall={wall:.1f}s rc={rc}\n")

    curr = load_summary(SUMMARY_PATH)
    if curr is None:
        sys.stderr.write(f"[sweep_diff] no summary at {SUMMARY_PATH} — sweep failed?\n")
        return 3

    if args.keep_history:
        ts = time.strftime("%Y%m%d_%H%M%S")
        kd_tag = f"kd{args.kd_weight}" if args.kd_weight > 0 else "base"
        hist_path = HISTORY_DIR / f"sweep_{ts}_{kd_tag}.json"
        shutil.copyfile(SUMMARY_PATH, hist_path)
        sys.stderr.write(f"[sweep_diff] archived to {hist_path}\n")

    if prev is None:
        # First run — print the verdict summary so the user has a baseline.
        print(f"[sweep_diff] no prior summary; baseline established")
        for verdict, names in sorted((curr.get("by_verdict") or {}).items(),
                                     key=lambda kv: -len(kv[1])):
            print(f"  {verdict:12s} ({len(names):2d}): {', '.join(sorted(names))}")
        return 0

    # Diff
    diffs = list(diff_summaries(prev, curr))
    regressions  = [line for sev, line in diffs if sev == "regress"]
    improvements = [line for sev, line in diffs if sev == "improve"]
    flaps        = [line for sev, line in diffs if sev == "flap"]
    others       = [line for sev, line in diffs if sev in ("new", "removed")]

    if not diffs:
        print("[sweep_diff] no verdict changes vs prior sweep")
        return 0

    print("[sweep_diff] verdict changes:")
    for line in regressions:  print(line)
    for line in improvements: print(line)
    for line in flaps:        print(line)
    for line in others:       print(line)

    if regressions:
        return 1
    if improvements:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
