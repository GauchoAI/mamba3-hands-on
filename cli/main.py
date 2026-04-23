#!/usr/bin/env python3
"""
Mamba CLI — Spark-like job submission for Mamba-3 training.

Usage:
    mamba nodes                          # list registered training nodes
    mamba status [--node NODE]           # show task status across nodes
    mamba submit --target NODE           # deploy problems + start training
    mamba logs --node NODE [--task T]    # stream logs from a node
    mamba pull --node NODE --output DIR  # download checkpoints
    mamba stop --node NODE               # stop training on a node
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def cmd_nodes(args):
    """List all registered training nodes."""
    from server.node_agent import list_nodes
    import time

    nodes = list_nodes()
    if not nodes:
        print("No nodes registered. Run 'python server/node_agent.py --register' on a training machine.")
        return

    now = time.time()
    print(f"{'Node':30s} {'Status':8s} {'Backends':25s} {'VRAM':>8s} {'GPU':20s} {'Last seen':>12s}")
    print("-" * 110)

    for nid, info in sorted(nodes.items()):
        if not isinstance(info, dict):
            continue
        last_hb = info.get("last_heartbeat", 0)
        age = now - last_hb if last_hb else float("inf")

        if age < 60:
            status = "online"
            age_str = f"{int(age)}s ago"
        elif age < 300:
            status = "stale"
            age_str = f"{int(age/60)}m ago"
        else:
            status = "offline"
            age_str = f"{int(age/3600)}h ago" if age < 86400 else "long ago"

        backends = ", ".join(info.get("backends", []))
        vram = f"{info.get('vram_mb', 0):,} MB"
        gpu = info.get("gpu_name", info.get("arch", ""))[:20]

        print(f"{nid:30s} {status:8s} {backends:25s} {vram:>8s} {gpu:20s} {age_str:>12s}")


def cmd_status(args):
    """Show task training status. Reads from Firebase."""
    from server.node_agent import _firebase_get
    import time

    # Get task status from three_pop data
    three_pop = _firebase_get("mamba3/three_pop")
    if not three_pop:
        print("No training data found in Firebase.")
        return

    # Show task leaderboard
    teachers = three_pop.get("teachers", {})
    tasks = three_pop.get("tasks", {})
    worker_lb = three_pop.get("worker_leaderboard", {})

    print("=== Task Status ===")
    print(f"{'Task':25s} {'Accuracy':>8s} {'Status':>10s} {'Config':40s}")
    print("-" * 90)

    # Merge teacher and task data
    all_tasks = set(list(teachers.keys()) + list(tasks.keys()))
    for task in sorted(all_tasks):
        teacher = teachers.get(task, {})
        task_info = tasks.get(task, {})

        acc = teacher.get("accuracy") or task_info.get("best_accuracy", 0)
        if isinstance(acc, (int, float)):
            acc_str = f"{acc:.0%}" if acc <= 1 else f"{acc:.0f}%"
        else:
            acc_str = str(acc)

        status = "mastered" if teacher else task_info.get("status", "unknown")

        cfg = teacher.get("config", task_info.get("current_config", {}))
        if isinstance(cfg, dict):
            cfg_str = f"d={cfg.get('d_model', '?')} L={cfg.get('n_kernel_layers', '?')} lr={cfg.get('lr', '?')}"
        elif isinstance(cfg, str):
            cfg_str = cfg[:40]
        else:
            cfg_str = ""

        print(f"{task:25s} {acc_str:>8s} {status:>10s} {cfg_str:40s}")

    # Show nodes
    print()
    nodes = _firebase_get("mamba3/nodes") or {}
    now = time.time()
    online = [nid for nid, info in nodes.items()
              if isinstance(info, dict) and now - info.get("last_heartbeat", 0) < 120]
    print(f"Nodes: {len(online)} online / {len(nodes)} registered")
    for nid in online:
        info = nodes[nid]
        backends = ", ".join(info.get("backends", []))
        print(f"  {nid}: {backends} ({info.get('gpu_name', info.get('arch', ''))})")


def cmd_submit(args):
    """Deploy problems + code to a target node and start training."""
    from server.node_agent import list_nodes, _firebase_get
    import subprocess
    import time

    # Resolve target node
    nodes = list_nodes()
    target = None

    if args.target:
        # Match by node_id or partial match
        for nid, info in nodes.items():
            if not isinstance(info, dict):
                continue
            if args.target in nid or args.target == info.get("hostname"):
                target = info
                target["node_id"] = nid
                break

    if not target:
        print(f"Node '{args.target}' not found. Available nodes:")
        for nid in sorted(nodes.keys()):
            print(f"  {nid}")
        return

    ssh = target.get("ssh", {})
    if not ssh:
        print(f"Node '{target['node_id']}' has no SSH info. Cannot deploy.")
        return

    host = ssh["host"]
    port = ssh.get("port", 22)
    user = ssh.get("user", "root")
    remote_dir = target.get("working_dir", "/root/mamba3-hands-on")

    print(f"Deploying to {target['node_id']} ({user}@{host}:{port})")
    print(f"  Remote dir: {remote_dir}")
    print(f"  Backends: {', '.join(target.get('backends', []))}")

    # Step 1: rsync code
    problems_dir = args.problems or "problems/"
    generators_dir = args.generators or "generators/"

    src_dir = str(Path(__file__).parent.parent)
    rsync_cmd = [
        "rsync", "-avz", "--delete",
        "-e", f"ssh -p {port}",
        "--include=*.py", "--include=*.yaml", "--include=*/",
        "--include=problems/***", "--include=generators/***",
        "--include=registry/***", "--include=server/***",
        "--exclude=checkpoints/", "--exclude=three_pop/",
        "--exclude=*.log", "--exclude=__pycache__/",
        "--exclude=.git/", "--exclude=*.pt",
        f"{src_dir}/", f"{user}@{host}:{remote_dir}/"
    ]

    print(f"  Syncing code...")
    result = subprocess.run(rsync_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  rsync failed: {result.stderr}")
        return

    # Step 2: Start training
    ssh_cmd = (
        f"cd {remote_dir} && "
        f"nohup {sys.executable} -u three_populations.py "
        f"> three_pop.log 2>&1 & echo $!"
    )
    start_cmd = ["ssh", "-p", str(port), f"{user}@{host}", ssh_cmd]

    print(f"  Starting training...")
    result = subprocess.run(start_cmd, capture_output=True, text=True, timeout=30)
    if result.returncode == 0:
        pid = result.stdout.strip()
        print(f"  Training started (PID: {pid})")
        print(f"  Dashboard: https://gauchoai.github.io/mamba3-hands-on/")
    else:
        print(f"  Failed to start: {result.stderr}")


def cmd_logs(args):
    """Stream logs from a training node."""
    from server.node_agent import list_nodes
    import subprocess

    nodes = list_nodes()
    target = None
    for nid, info in nodes.items():
        if not isinstance(info, dict):
            continue
        if args.node in nid:
            target = info
            target["node_id"] = nid
            break

    if not target:
        print(f"Node '{args.node}' not found.")
        return

    ssh = target.get("ssh", {})
    if not ssh:
        print("No SSH info for this node.")
        return

    host = ssh["host"]
    port = ssh.get("port", 22)
    user = ssh.get("user", "root")
    remote_dir = target.get("working_dir", "/root/mamba3-hands-on")

    log_file = "three_pop.log"
    lines = args.lines or 50

    cmd = ["ssh", "-p", str(port), f"{user}@{host}",
           f"tail -{lines} {remote_dir}/{log_file}"]

    if args.follow:
        cmd[-1] = f"tail -f {remote_dir}/{log_file}"
        print(f"Streaming logs from {target['node_id']} (Ctrl+C to stop)...")
        try:
            subprocess.run(cmd)
        except KeyboardInterrupt:
            pass
    else:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        if result.returncode == 0:
            print(result.stdout)
        else:
            print(f"Error: {result.stderr}")


def cmd_stop(args):
    """Stop training on a node (sends SIGTERM, workers save checkpoints)."""
    from server.node_agent import list_nodes
    import subprocess

    nodes = list_nodes()
    target = None
    for nid, info in nodes.items():
        if not isinstance(info, dict):
            continue
        if args.node in nid:
            target = info
            target["node_id"] = nid
            break

    if not target:
        print(f"Node '{args.node}' not found.")
        return

    ssh = target.get("ssh", {})
    if not ssh:
        print("No SSH info for this node.")
        return

    host = ssh["host"]
    port = ssh.get("port", 22)
    user = ssh.get("user", "root")

    # Confirm with user
    print(f"Stop training on {target['node_id']}? Workers will save checkpoints.")
    confirm = input("Type 'yes' to confirm: ")
    if confirm.lower() != "yes":
        print("Cancelled.")
        return

    cmd = ["ssh", "-p", str(port), f"{user}@{host}",
           "pkill -TERM -f three_populations.py"]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
    if result.returncode == 0:
        print(f"  SIGTERM sent. Workers will checkpoint and exit.")
    else:
        print(f"  Error: {result.stderr}")


def cmd_pull(args):
    """Download checkpoints from a training node."""
    from server.node_agent import list_nodes
    import subprocess

    nodes = list_nodes()
    target = None
    for nid, info in nodes.items():
        if not isinstance(info, dict):
            continue
        if args.node in nid:
            target = info
            target["node_id"] = nid
            break

    if not target:
        print(f"Node '{args.node}' not found.")
        return

    ssh = target.get("ssh", {})
    if not ssh:
        print("No SSH info for this node.")
        return

    host = ssh["host"]
    port = ssh.get("port", 22)
    user = ssh.get("user", "root")
    remote_dir = target.get("working_dir", "/root/mamba3-hands-on")
    output = args.output or "pulled_checkpoints/"

    os.makedirs(output, exist_ok=True)

    rsync_cmd = [
        "rsync", "-avz",
        "-e", f"ssh -p {port}",
        f"{user}@{host}:{remote_dir}/checkpoints/specialists/",
        f"{output}/"
    ]

    print(f"Pulling checkpoints from {target['node_id']} → {output}/")
    result = subprocess.run(rsync_cmd, text=True)
    if result.returncode == 0:
        print("Done.")
    else:
        print("Pull failed.")


def main():
    from pathlib import Path

    parser = argparse.ArgumentParser(
        prog="mamba",
        description="Mamba training platform CLI",
    )
    sub = parser.add_subparsers(dest="command", help="Command")

    # nodes
    p_nodes = sub.add_parser("nodes", help="List registered training nodes")

    # status
    p_status = sub.add_parser("status", help="Show task training status")
    p_status.add_argument("--node", type=str, default=None)

    # submit
    p_submit = sub.add_parser("submit", help="Deploy and start training on a node")
    p_submit.add_argument("--target", type=str, required=True, help="Node ID or hostname")
    p_submit.add_argument("--problems", type=str, default=None, help="Problems directory")
    p_submit.add_argument("--generators", type=str, default=None, help="Generators directory")

    # logs
    p_logs = sub.add_parser("logs", help="View logs from a training node")
    p_logs.add_argument("--node", type=str, required=True)
    p_logs.add_argument("--lines", type=int, default=50)
    p_logs.add_argument("--follow", "-f", action="store_true")

    # pull
    p_pull = sub.add_parser("pull", help="Download checkpoints from a node")
    p_pull.add_argument("--node", type=str, required=True)
    p_pull.add_argument("--output", type=str, default=None)

    # stop
    p_stop = sub.add_parser("stop", help="Stop training on a node")
    p_stop.add_argument("--node", type=str, required=True)

    args = parser.parse_args()

    if args.command == "nodes":
        cmd_nodes(args)
    elif args.command == "status":
        cmd_status(args)
    elif args.command == "submit":
        cmd_submit(args)
    elif args.command == "logs":
        cmd_logs(args)
    elif args.command == "pull":
        cmd_pull(args)
    elif args.command == "stop":
        cmd_stop(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
