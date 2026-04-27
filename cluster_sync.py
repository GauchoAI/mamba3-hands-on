"""cluster_sync — rsync the repo to every non-local node.

Excludes .venv, runs, checkpoints, three_pop, .git, __pycache__,
node_modules — everything per-node should be local. The .venv on
each node is set up once manually (torch + MPS).

Usage:
  python cluster_sync.py
  python cluster_sync.py --dry-run
  python cluster_sync.py --node m4-mini
"""
import argparse, json, subprocess, sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_EXCLUDES = [
    ".venv/", "runs/", "checkpoints/", "three_pop/", ".git/",
    "__pycache__/", "node_modules/", ".pytest_cache/",
    "*.pyc", "*.pt", "*.pkl", ".DS_Store",
    "/tmp/", "models_cache/",
]


def rsync_to(node, dry_run=False):
    user = node["user"]
    host = node["host"]
    port = node.get("port", 22)
    remote_path = node.get("repo_path", "~/mamba3-hands-on")
    # rsync needs trailing slash on src to copy contents not parent dir
    src = str(REPO_ROOT) + "/"
    dst = f"{user}@{host}:{remote_path}/"
    cmd = [
        "rsync", "-avz", "--delete",
        "-e", f"ssh -o ServerAliveInterval=30 -p {port}",
    ]
    for ex in DEFAULT_EXCLUDES:
        cmd += ["--exclude", ex]
    if dry_run:
        cmd.append("--dry-run")
    cmd += [src, dst]
    print(f"  rsync -> {node['name']} ({host}:{remote_path})", flush=True)
    rc = subprocess.run(cmd, stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT).returncode
    return rc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", default="cluster_nodes.json")
    ap.add_argument("--node", default=None,
                    help="Sync only this named node (default: all non-local)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    nodes = json.loads(Path(args.nodes).read_text())
    targets = [n for n in nodes
               if not n.get("local")
               and (args.node is None or n["name"] == args.node)]
    if not targets:
        print("No remote nodes to sync.", flush=True)
        return 0
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
