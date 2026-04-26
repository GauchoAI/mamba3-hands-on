"""cluster_dispatch — fan a task list out to N Apple-silicon nodes via SSH.

The Apple Silicon cluster plan (project memory) is one M4 Pro now,
2 more M4 Mac minis arriving ~2026-05-06. Each node has one MPS GPU,
so the right grain is one task per node at a time. This module is the
shell that lets `mac_sweep.py` (currently sequential, single-host) run
across multiple Macs.

Wire (intended once cluster lands):
  python3 cluster_dispatch.py --nodes ~/cluster_nodes.json --tasks-file all_tasks.txt

The nodes JSON is a simple list of dicts:
  [
    {"name": "m4-pro",      "host": "192.168.1.10", "port": 22, "user": "miguel"},
    {"name": "m4-mini-a",   "host": "192.168.1.11", "port": 22, "user": "miguel"},
    {"name": "m4-mini-b",   "host": "192.168.1.12", "port": 22, "user": "miguel"}
  ]

Each spawn:
  ssh {user}@{host} -p {port} 'cd ~/mamba3-hands-on && .venv/bin/python \
       specialist_trainer.py --task {task} --device mps ...'

Stdout streams back; we keep one task per node assignment via a queue,
and reassign the next pending task as soon as a node finishes. No
concurrency on a single node — MPS owns the whole GPU.

Usage today (single host, no SSH): the same harness works locally —
pass `--nodes-file -` and it skips SSH for the current host.
"""
import argparse, json, queue, subprocess, sys, threading, time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent


def build_remote_cmd(node, task, args):
    """Build the SSH command that runs specialist_trainer on `node` for `task`."""
    user_at_host = f'{node["user"]}@{node["host"]}'
    remote = (
        f'cd {node.get("repo_path", "~/mamba3-hands-on")} && '
        f'.venv/bin/python -u specialist_trainer.py '
        f'--task {task} --device mps '
        f'--d-model {args.d_model} --d-state {args.d_state} '
        f'--headdim {args.headdim} --layers {args.layers} '
        f'--batch-size {args.batch_size} --lr {args.lr} '
        f'--weight-decay {args.weight_decay} '
        f'--steps-per-cycle {args.steps_per_cycle} '
        f'--max-cycles {args.max_cycles} '
        f'--target-acc {args.target_acc}'
    )
    return ["ssh", "-o", "ServerAliveInterval=30",
            "-p", str(node.get("port", 22)),
            user_at_host, remote]


def build_local_cmd(task, args):
    """Local run on this host — no SSH."""
    return [
        sys.executable, "-u", str(REPO_ROOT / "specialist_trainer.py"),
        "--task", task, "--device", "mps",
        "--d-model", str(args.d_model), "--d-state", str(args.d_state),
        "--headdim", str(args.headdim), "--layers", str(args.layers),
        "--batch-size", str(args.batch_size),
        "--lr", str(args.lr), "--weight-decay", str(args.weight_decay),
        "--steps-per-cycle", str(args.steps_per_cycle),
        "--max-cycles", str(args.max_cycles),
        "--target-acc", str(args.target_acc),
    ]


def worker_loop(node, task_q, results, args):
    """Per-node loop: pull tasks off the queue, run, post result."""
    while True:
        try:
            task = task_q.get_nowait()
        except queue.Empty:
            return
        cmd = (build_local_cmd(task, args) if node.get("local")
               else build_remote_cmd(node, task, args))
        log_path = Path("/tmp/cluster_dispatch_logs") / f"{node['name']}_{task}.log"
        log_path.parent.mkdir(exist_ok=True)
        t0 = time.time()
        with open(log_path, "w") as f:
            try:
                rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                                    timeout=args.per_task_timeout).returncode
            except subprocess.TimeoutExpired:
                rc = -1
        wall = time.time() - t0
        results.append({"node": node["name"], "task": task,
                        "rc": rc, "wall_s": round(wall, 1),
                        "log": str(log_path)})
        print(f"  [{node['name']}] {task} done in {wall:.1f}s rc={rc}", flush=True)
        task_q.task_done()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", default=None,
                    help="Path to nodes JSON; omit to run local-only (this host)")
    ap.add_argument("--tasks", nargs="+", required=True,
                    help="Task names to fan out")
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
    ap.add_argument("--per-task-timeout", type=int, default=600)
    args = ap.parse_args()

    if args.nodes:
        nodes = json.loads(Path(args.nodes).read_text())
    else:
        nodes = [{"name": "local", "local": True}]

    print(f"Dispatching {len(args.tasks)} tasks across {len(nodes)} nodes")
    for n in nodes:
        print(f"  {n['name']}: " + ("local" if n.get('local') else
              f"{n.get('user','')}@{n.get('host','')}:{n.get('port',22)}"))

    task_q = queue.Queue()
    for t in args.tasks:
        task_q.put(t)
    results = []
    threads = [threading.Thread(target=worker_loop,
                                args=(n, task_q, results, args), daemon=True)
               for n in nodes]
    t0 = time.time()
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    total = time.time() - t0

    print(f"\n=== summary (total {total:.1f}s, {len(nodes)} node(s)) ===")
    by_node = {}
    for r in results:
        by_node.setdefault(r["node"], []).append(r)
    for node, rs in by_node.items():
        ok = sum(1 for r in rs if r["rc"] == 0)
        print(f"  {node}: {ok}/{len(rs)} ok, total {sum(r['wall_s'] for r in rs):.1f}s")

    out_path = Path("/tmp/cluster_dispatch_summary.json")
    out_path.write_text(json.dumps({"results": results, "total_wall_s": total}, indent=2))
    print(f"Summary at {out_path}")


if __name__ == "__main__":
    main()
