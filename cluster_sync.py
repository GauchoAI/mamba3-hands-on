"""cluster_sync — rsync the repo to every non-local node.

Excludes .venv, runs, checkpoints, three_pop, .git, __pycache__,
node_modules — everything per-node should be local. The .venv on
each node is set up once manually (torch + MPS).

Usage:
  python cluster_sync.py                       # one-shot sync to all remotes
  python cluster_sync.py --dry-run
  python cluster_sync.py --node m4-mini
  python cluster_sync.py --watch               # poll forever, sync on change
  python cluster_sync.py --watch --interval 30 --node m4-mini

`--watch` mode: every `--interval` seconds, scan source-file mtimes;
if anything is newer than the last successful sync, rsync. Quietly
skips when remote hosts are offline. Replaces the manual
'edit-then-cluster_sync' loop with an automatic propagation pipe.
"""
import argparse, json, os, subprocess, sys, time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_EXCLUDES = [
    ".venv/", "runs/", "checkpoints/", "three_pop/", ".git/",
    "__pycache__/", "node_modules/", ".pytest_cache/",
    "*.pyc", "*.pt", "*.pkl", ".DS_Store",
    "/tmp/", "models_cache/",
]


def rsync_to(node, dry_run=False, quiet=False):
    user = node["user"]
    host = node["host"]
    port = node.get("port", 22)
    remote_path = node.get("repo_path", "~/mamba3-hands-on")
    # rsync needs trailing slash on src to copy contents not parent dir
    src = str(REPO_ROOT) + "/"
    dst = f"{user}@{host}:{remote_path}/"
    # Fast-fail SSH connect so the watch loop doesn't stall when m4-mini
    # is asleep. ServerAliveInterval keeps long transfers alive once
    # connected.
    rsh = (f"ssh -o ConnectTimeout=4 -o BatchMode=yes "
           f"-o ServerAliveInterval=30 "
           f"-o StrictHostKeyChecking=accept-new -p {port}")
    cmd = ["rsync", "-az", "--delete", "-e", rsh]
    for ex in DEFAULT_EXCLUDES:
        cmd += ["--exclude", ex]
    if dry_run:
        cmd.append("--dry-run")
    cmd += [src, dst]
    if not quiet:
        print(f"  rsync -> {node['name']} ({host}:{remote_path})", flush=True)
    return subprocess.run(cmd, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT).returncode


# ─────────────────────────────────────────────────────────────────────
# Watch mode — poll for source-file changes and auto-sync.
# ─────────────────────────────────────────────────────────────────────

def _latest_source_mtime(root: Path) -> float:
    """Walk root, ignoring excluded dirs/files, return the newest mtime."""
    skip_dirs = {".venv", "runs", "checkpoints", "three_pop", ".git",
                 "__pycache__", "node_modules", ".pytest_cache",
                 "models_cache"}
    skip_exts = {".pyc", ".pt", ".pkl"}
    latest = 0.0
    for r, dirs, files in os.walk(root):
        # Prune excluded directories in-place so os.walk skips them.
        dirs[:] = [d for d in dirs if d not in skip_dirs and not d.startswith(".")]
        for f in files:
            if f == ".DS_Store":
                continue
            ext = os.path.splitext(f)[1]
            if ext in skip_exts:
                continue
            try:
                m = os.path.getmtime(os.path.join(r, f))
                if m > latest:
                    latest = m
            except OSError:
                pass
    return latest


def watch_loop(targets, interval: int) -> int:
    """Poll source-tree mtime; rsync when something newer than last
    successful sync. Per-target last-sync timestamp so each remote
    catches up independently.
    """
    last_sync = {n["name"]: 0.0 for n in targets}
    print(f"[watch] {len(targets)} target(s), interval={interval}s. "
          f"Ctrl-C to stop.", flush=True)
    try:
        while True:
            current = _latest_source_mtime(REPO_ROOT)
            for n in targets:
                if current <= last_sync[n["name"]]:
                    continue
                # Source changed since last sync to this target — try.
                rc = rsync_to(n, quiet=True)
                if rc == 0:
                    last_sync[n["name"]] = current
                    print(f"[watch] {time.strftime('%H:%M:%S')}  "
                          f"synced → {n['name']}", flush=True)
                # Non-zero: probably offline. Stay silent so the loop
                # doesn't spam logs; we'll retry next tick.
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n[watch] stopped.", flush=True)
        return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", default="cluster_nodes.json")
    ap.add_argument("--node", default=None,
                    help="Sync only this named node (default: all non-local)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--watch", action="store_true",
                    help="poll forever, auto-sync when source changes")
    ap.add_argument("--interval", type=int, default=15,
                    help="watch poll interval in seconds (default 15)")
    args = ap.parse_args()

    nodes = json.loads(Path(args.nodes).read_text())
    targets = [n for n in nodes
               if not n.get("local")
               and (args.node is None or n["name"] == args.node)]
    if not targets:
        print("No remote nodes to sync.", flush=True)
        return 0

    if args.watch:
        return watch_loop(targets, args.interval)

    print(f"Syncing repo to {len(targets)} node(s){' [DRY RUN]' if args.dry_run else ''}:",
          flush=True)
    fails = 0
    for n in targets:
        rc = rsync_to(n, dry_run=args.dry_run)
        if rc != 0:
            fails += 1
            print(f"    FAIL rc={rc}", flush=True)
        else:
            print(f"    OK", flush=True)
    return fails


if __name__ == "__main__":
    sys.exit(main())
