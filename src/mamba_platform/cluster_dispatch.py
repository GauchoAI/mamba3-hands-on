"""cluster_dispatch — fan a job manifest out to N Apple-silicon nodes.

A job manifest is a JSON file: a list of dicts with shape

    [
      {"node": "m4-pro",  "name": "fibd-lr1e-4",
       "cmd": ".venv/bin/python specialist_trainer.py --task fib_decimal ..."},
      {"node": "m4-mini", "name": "hanoi-rerun",
       "cmd": ".venv/bin/python specialist_trainer.py --task tower_of_hanoi_binary ..."},
    ]

The dispatcher pulls one job per node at a time (MPS owns the whole
GPU, so concurrency on a single node is wasted), runs them in
parallel via SSH for remote nodes / subprocess for local, streams
stdout to per-job log files, and prints a summary at the end.

Two ways to give it work:
  - manifest:  --manifest jobs.json
  - shorthand: --node m4-mini --cmd '...' --name foo
               (single ad-hoc job; useful from CLI)

Logs land in /tmp/cluster_dispatch_logs/{name}.log
"""
import argparse, json, queue, subprocess, sys, threading, time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
LOG_DIR = Path("/tmp/cluster_dispatch_logs")


def _node_by_name(nodes, name):
    for n in nodes:
        if n["name"] == name:
            return n
    raise KeyError(f"node {name!r} not in cluster_nodes.json")


def build_cmd(node, raw_cmd):
    """Build the local subprocess argv for a job on `node`.

    For a local node we run the cmd via `bash -lc` so PATH/cd work.
    For a remote node we ssh in, cd into the repo, and run the cmd
    in a bash login shell.
    """
    if node.get("local"):
        # Local execution; cd to repo root for parity with remote.
        repo = node.get("repo_path", str(REPO_ROOT))
        return ["bash", "-lc", f"cd {repo} && {raw_cmd}"]
    user_at_host = f'{node["user"]}@{node["host"]}'
    repo = node.get("repo_path", "~/mamba3-hands-on")
    remote = f"cd {repo} && {raw_cmd}"
    return [
        "ssh", "-o", "ServerAliveInterval=30",
        "-p", str(node.get("port", 22)),
        user_at_host, remote,
    ]


def worker_loop(node, task_q, results, per_task_timeout):
    while True:
        try:
            job = task_q.get_nowait()
        except queue.Empty:
            return
        cmd = build_cmd(node, job["cmd"])
        log_path = LOG_DIR / f"{job['name']}.log"
        t0 = time.time()
        with open(log_path, "w") as f:
            f.write(f"# node={node['name']} cmd={job['cmd']}\n")
            f.flush()
            try:
                rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                                    timeout=per_task_timeout).returncode
            except subprocess.TimeoutExpired:
                rc = -1
        wall = time.time() - t0
        results.append({"node": node["name"], "name": job["name"],
                        "rc": rc, "wall_s": round(wall, 1),
                        "log": str(log_path)})
        print(f"  [{node['name']}] {job['name']} done in {wall:.0f}s rc={rc}",
              flush=True)
        task_q.task_done()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", default="cluster_nodes.json")
    ap.add_argument("--manifest", default=None,
                    help="Path to JSON manifest of jobs")
    ap.add_argument("--node", default=None,
                    help="(shorthand) run a single ad-hoc job on this node")
    ap.add_argument("--cmd", default=None,
                    help="(shorthand) the command to run for the ad-hoc job")
    ap.add_argument("--name", default="adhoc",
                    help="(shorthand) name for the ad-hoc job (used for log filename)")
    ap.add_argument("--per-task-timeout", type=int, default=14400,
                    help="seconds; default 4h")
    args = ap.parse_args()

    nodes = json.loads(Path(args.nodes).read_text())
    nodes_by_name = {n["name"]: n for n in nodes}

    if args.manifest:
        jobs = json.loads(Path(args.manifest).read_text())
    elif args.cmd and args.node:
        jobs = [{"node": args.node, "name": args.name, "cmd": args.cmd}]
    else:
        ap.error("provide either --manifest or both --node and --cmd")

    LOG_DIR.mkdir(exist_ok=True)
    print(f"Cluster dispatch: {len(jobs)} job(s) across {len(set(j['node'] for j in jobs))} node(s)")
    for j in jobs:
        print(f"  [{j['node']}] {j['name']}: {j['cmd'][:80]}{'...' if len(j['cmd'])>80 else ''}")

    # Group jobs per node (one queue per node — MPS owns its GPU).
    per_node_q = {n["name"]: queue.Queue() for n in nodes}
    for j in jobs:
        if j["node"] not in per_node_q:
            ap.error(f"job references unknown node {j['node']!r}")
        per_node_q[j["node"]].put(j)

    results = []
    threads = []
    for n in nodes:
        q = per_node_q[n["name"]]
        if q.empty():
            continue
        t = threading.Thread(target=worker_loop,
                             args=(n, q, results, args.per_task_timeout),
                             daemon=True)
        threads.append(t)

    t0 = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    total = time.time() - t0

    print(f"\n=== summary (total {total:.1f}s) ===")
    by_node = {}
    for r in results:
        by_node.setdefault(r["node"], []).append(r)
    for node, rs in by_node.items():
        ok = sum(1 for r in rs if r["rc"] == 0)
        print(f"  {node}: {ok}/{len(rs)} ok, total {sum(r['wall_s'] for r in rs):.1f}s")
        for r in rs:
            mark = "✓" if r["rc"] == 0 else "✗"
            print(f"    {mark} {r['name']:<30} {r['wall_s']:>7.1f}s  log={r['log']}")
    out_path = LOG_DIR / "summary.json"
    out_path.write_text(json.dumps({"results": results, "total_wall_s": total},
                                   indent=2))
    print(f"\nSummary at {out_path}")
    return sum(1 for r in results if r["rc"] != 0)


if __name__ == "__main__":
    sys.exit(main())
