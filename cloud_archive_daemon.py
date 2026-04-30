"""cloud_archive_daemon.py — long-lived process that mirrors local
artifact directories to the HF bucket continuously.

The pattern: each `CloudArchive` instance already runs its own
background thread that re-syncs every HF_ARCHIVE_SYNC_EVERY seconds.
This daemon just spawns one of those per artifact directory and
sleeps forever. So:

  - Files that exist *before* the daemon starts get synced on the
    first tick (because sync_bucket sees them as "new").
  - Files added later (training checkpoints, freshly-generated
    corpora, things you `mv` into the watched directory) get synced
    on the next tick.
  - The bucket is the source of truth for "what we have"; local
    disks are caches.

Default targets cover the standard repo layout. Override with
--config to point at a JSON file for non-standard layouts.

Run:
    python cloud_archive_daemon.py
    python cloud_archive_daemon.py --once    # single sync, then exit
    python cloud_archive_daemon.py --config archive_targets.json
    nohup python cloud_archive_daemon.py > archive_daemon.log 2>&1 &

Companion to eval_daemon.py (jepa/eval_daemon.py) — same shape:
long-lived watcher, fault-tolerant, never blocks anything.
"""
from __future__ import annotations
import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path

from cloud_archive import CloudArchive


# ─────────────────────────────────────────────────────────────────────
# Default watch list. Each entry says "mirror <local_dir> to
# <experiment_kind>/<run_name>/* in the bucket." Glob expansion is
# applied to local_dir at startup so the daemon picks up sibling
# directories under checkpoints/ without needing a config update for
# every new run.
# ─────────────────────────────────────────────────────────────────────

DEFAULT_TARGETS = [
    # Generated corpora.
    {
        "local_dir": "data",
        "experiment_kind": "corpus",
        "run_name": "data-root",
        "exclude": ["*.zip", "*.tar.gz"],   # belt-and-suspenders; defaults already cover _* caches
    },
    # All bilingual cortex experiment checkpoints. Each subdirectory
    # of checkpoints/ that matches lm* or cortex* gets its own
    # CloudArchive.
    {
        "local_dir_glob": "checkpoints/lm*",
        "experiment_kind": "cortex_bilingual",
    },
    {
        "local_dir_glob": "checkpoints/cortex*",
        "experiment_kind": "cortex",
    },
    # JEPA + RLF runs (each ships its own experiment_pusher already;
    # archive picks up checkpoints + sample logs).
    {
        "local_dir_glob": "jepa/runs/*",
        "experiment_kind": "jepa",
    },
    {
        "local_dir_glob": "rlf_cortex/runs/*",
        "experiment_kind": "rlf_cortex",
    },
]


def expand_targets(raw: list[dict]) -> list[dict]:
    """Resolve `local_dir_glob` into one entry per matching directory."""
    out = []
    for t in raw:
        if "local_dir" in t:
            out.append(t)
            continue
        pattern = t.get("local_dir_glob")
        if not pattern:
            continue
        from glob import glob
        for p in sorted(glob(pattern)):
            if not os.path.isdir(p):
                continue
            entry = dict(t)
            entry.pop("local_dir_glob", None)
            entry["local_dir"] = p
            entry.setdefault("run_name", os.path.basename(p))
            out.append(entry)
    return out


def build_archives(targets: list[dict]) -> list[CloudArchive]:
    archives = []
    for t in targets:
        local_dir = t["local_dir"]
        if not os.path.isdir(local_dir):
            print(f"[daemon] skip missing dir: {local_dir}", flush=True)
            continue
        kwargs = {
            "experiment_kind": t["experiment_kind"],
            "run_name": t.get("run_name", os.path.basename(local_dir)),
            "local_dir": local_dir,
        }
        if "exclude" in t:
            kwargs["exclude"] = t["exclude"]
        if "include" in t:
            kwargs["include"] = t["include"]
        archives.append(CloudArchive(**kwargs))
    return archives


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None,
                    help="JSON file with archive targets (overrides defaults)")
    ap.add_argument("--once", action="store_true",
                    help="run a single sync and exit")
    ap.add_argument("--sync-every", type=int, default=None,
                    help="override HF_ARCHIVE_SYNC_EVERY for this daemon")
    args = ap.parse_args()

    if args.sync_every is not None:
        os.environ["HF_ARCHIVE_SYNC_EVERY"] = str(args.sync_every)

    if args.config:
        raw = json.loads(Path(args.config).read_text())
    else:
        raw = DEFAULT_TARGETS

    targets = expand_targets(raw)
    if not targets:
        print("[daemon] no targets resolved — nothing to do.", flush=True)
        return 0

    print(f"[daemon] resolving {len(targets)} archive target(s):", flush=True)
    for t in targets:
        print(f"  {t['local_dir']:40s} → {t['experiment_kind']}/"
              f"{t.get('run_name', os.path.basename(t['local_dir']))}",
              flush=True)

    archives = build_archives(targets)
    if not archives:
        print("[daemon] no archives created — check HF_TOKEN / dir existence.",
              flush=True)
        return 1

    if args.once:
        print("[daemon] one-shot sync...", flush=True)
        for a in archives:
            a.sync_now()
            a.complete(timeout=300.0)
        print("[daemon] one-shot complete.", flush=True)
        return 0

    # Long-lived: each CloudArchive's own daemon thread does the
    # periodic syncing. Main thread sleeps until a signal.
    stop = {"flag": False}
    def _on_signal(signo, frame):
        print(f"\n[daemon] caught signal {signo}, stopping…", flush=True)
        stop["flag"] = True
    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    print(f"[daemon] running. Each target re-syncs every "
          f"{os.environ.get('HF_ARCHIVE_SYNC_EVERY', '60')}s. "
          f"Ctrl-C / SIGTERM to stop.", flush=True)

    # Initial sync immediately so the user sees movement on startup
    # without waiting one full tick.
    for a in archives:
        a.sync_now()

    while not stop["flag"]:
        time.sleep(1)

    print("[daemon] flushing final syncs…", flush=True)
    for a in archives:
        a.complete(timeout=120.0)
    print("[daemon] done.", flush=True)
    return 0


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    sys.exit(main())
