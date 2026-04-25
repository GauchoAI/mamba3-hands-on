#!/usr/bin/env python3
"""
ptxd_tail — pipe ptxd's stdout through this to forward TickEvents to
Firebase and stream cycle/final events to stdout (or a logfile).

Usage:
    ./ptxd --concurrent 4 < jobs.jsonl | python3 ptxd_tail.py [--gen N]

Or as a wrapper:
    python3 ptxd_tail.py --cmd ./ptxd --concurrent 4 < jobs.jsonl

Design:
  - Tick events (~50 bytes, 1Hz) → Firebase RTDB at
        mamba3/scheduler_history/{generation}
    Path mirrors push_gpu_tick's existing scheme so the UI can plot
    sm_pct + mem_pct + running + queue as a sparkline alongside GPU%.
  - Cycle / Final events stream to stdout unchanged so downstream
    consumers (ptxd_specialist.py's reader, the GA dashboard) see what
    they used to see.
  - Batches Firebase pushes every flush_interval_s (default 5s) with
    up to N ticks at once → ~12 HTTP requests/min, well under Firebase
    free-tier limits and within reasonable bandwidth.

If firebase_push isn't available, falls through silently — the cycle/
final pass-through still works for local debugging.
"""
import argparse, json, os, sys, time, subprocess
from collections import deque

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", type=int, default=int(time.time()),
                    help="generation id for Firebase path (default: unix time)")
    ap.add_argument("--flush-interval", type=float, default=5.0,
                    help="seconds between Firebase pushes (default 5)")
    ap.add_argument("--cmd", nargs="+",
                    help="if provided, run this command and tail its stdout. "
                         "otherwise reads from stdin.")
    ap.add_argument("--no-firebase", action="store_true",
                    help="don't push, just print")
    args = ap.parse_args()

    # Optional Firebase. If the import fails (or dependency isn't installed),
    # we just don't push — the rest of the pipe still works.
    push_tick = None
    if not args.no_firebase:
        try:
            from firebase_push import _post
            def push_tick(gen, ticks):
                """Append-style: each tick becomes a POST to Firebase RTDB.
                POST auto-generates a sortable child key, giving us a
                time series under mamba3/scheduler_history/{gen}/.
                For batching we'd switch to PATCH, but at our cadence
                (~1Hz) the request rate is fine.
                """
                for tk in ticks:
                    _post(f"mamba3/scheduler_history/{gen}", {
                        "t":       round(tk["t"], 2),
                        "mem":     round(tk["mem_pct"], 1),
                        "sm":      round(tk["sm_pct"], 1),
                        "running": tk["running"],
                        "queue":   tk["queue"],
                    })
        except Exception as e:
            sys.stderr.write(f"[ptxd_tail] Firebase disabled: {e}\n")
            push_tick = None

    # Source of events: subprocess pipe or stdin.
    if args.cmd:
        proc = subprocess.Popen(args.cmd, stdout=subprocess.PIPE, text=True, bufsize=1)
        source = proc.stdout
    else:
        source = sys.stdin
        proc = None

    pending_ticks: deque = deque()
    last_flush = time.time()
    n_ticks_pushed = 0
    n_cycles = 0
    n_finals = 0

    def flush():
        nonlocal last_flush, n_ticks_pushed
        if pending_ticks and push_tick is not None:
            batch = list(pending_ticks)
            pending_ticks.clear()
            try:
                push_tick(args.gen, batch)
                n_ticks_pushed += len(batch)
            except Exception as e:
                sys.stderr.write(f"[ptxd_tail] push failed: {e}\n")
        last_flush = time.time()

    try:
        for line in source:
            line = line.strip()
            if not line: continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                # not a JSON line — pass through (e.g., ptxd's stderr-like prints)
                print(line, flush=True)
                continue

            t = ev.get("type")
            if t == "tick":
                pending_ticks.append(ev)
                # Don't pass tick events to stdout — they're noise for
                # downstream consumers expecting cycle/final.
            elif t == "cycle":
                n_cycles += 1
                print(line, flush=True)
            elif t == "final":
                n_finals += 1
                print(line, flush=True)
            else:
                # Unknown event type — pass through.
                print(line, flush=True)

            if time.time() - last_flush >= args.flush_interval:
                flush()
    except KeyboardInterrupt:
        pass

    flush()
    sys.stderr.write(
        f"[ptxd_tail] done. ticks_pushed={n_ticks_pushed}  "
        f"cycles={n_cycles}  finals={n_finals}\n"
    )
    if proc is not None:
        proc.wait()


if __name__ == "__main__":
    main()
