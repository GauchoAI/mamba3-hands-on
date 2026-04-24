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
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


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
    """Global overview — all tasks, all nodes, learning rate, bandwidth."""
    from server.node_agent import _firebase_get
    import time

    three_pop = _firebase_get("mamba3/three_pop") or {}
    teachers = three_pop.get("teachers", {})
    tasks = three_pop.get("tasks", {})
    nodes_data = _firebase_get("mamba3/nodes") or {}
    lr_data = _firebase_get("mamba3/learning_rate") or {}
    now = time.time()

    # Header
    n_teachers = len(teachers)
    all_tasks = set(list(teachers.keys()) + list(tasks.keys()))
    n_tasks = len(all_tasks)
    online = [nid for nid, info in nodes_data.items()
              if isinstance(info, dict) and now - info.get("last_heartbeat", 0) < 120]

    print(f"=== Mamba Platform Overview ===")
    print(f"  Tasks: {n_tasks} | Teachers: {n_teachers} | Nodes: {len(online)} online")
    lr = lr_data.get("learning_ratio")
    if lr:
        label = "accelerating" if lr < 0.8 else "steady" if lr < 1.2 else "harder tasks"
        print(f"  Learning rate: {lr:.2f} ({label})")
    print()

    # Task table — grouped by level
    l0, l1, l2, other = [], [], [], []
    for task in sorted(all_tasks):
        teacher = teachers.get(task, {})
        task_info = tasks.get(task, {})
        acc = teacher.get("accuracy") or task_info.get("best_accuracy", 0)
        if isinstance(acc, (int, float)):
            acc_pct = round(acc * 100) if acc <= 1 else round(acc)
        else:
            acc_pct = 0
        status = "mastered" if teacher else task_info.get("status", "training")
        stage = task_info.get("current_stage", 1)
        node = (teacher.get("node") or task_info.get("node") or "")[:15]
        distilled = "D" if task_info.get("distilled_from") else ""
        entry = (task, acc_pct, status, stage, node, distilled)

        # Classify by level
        tags = []
        if task in ["parity","binary_pattern_next","same_different","odd_one_out",
                    "sequence_completion","pattern_period","run_length_next",
                    "mirror_detection","repeat_count","arithmetic_next",
                    "geometric_next","alternating_next","logic_gate","logic_chain","modus_ponens"]:
            l0.append(entry)
        elif task in ["cumulative_sum","max_element","min_element","sort_check",
                     "duplicate_detect","element_position","reverse_sequence",
                     "fibonacci_next","modular_arithmetic","comparison_chain"]:
            l1.append(entry)
        elif task in ["count_above_threshold","second_largest","range_of_sequence",
                     "conditional_sum","majority_element"]:
            l2.append(entry)
        else:
            other.append(entry)

    def _print_group(label, entries):
        if not entries:
            return
        mastered = sum(1 for e in entries if e[2] == "mastered")
        print(f"--- {label} ({mastered}/{len(entries)} mastered) ---")
        print(f"  {'Task':25s} {'Acc':>5s} {'Stg':>3s} {'D':>1s} {'Node':>15s} {'Status':>10s}")
        for task, acc, status, stage, node, dist in entries:
            bar = "█" * (acc // 10) + "░" * (10 - acc // 10)
            st = "✓" if status == "mastered" else " "
            print(f"  {task:25s} {acc:>4d}% {stage:>3d} {dist:>1s} {node:>15s} {bar} {st}")
        print()

    _print_group("Level 0 — Pattern Recognition", l0)
    _print_group("Level 1 — Reasoning", l1)
    _print_group("Level 2 — Composition", l2)
    if other:
        _print_group("Other", other)

    # Nodes
    print(f"--- Nodes ({len(online)}/{len(nodes_data)}) ---")
    for nid, info in sorted(nodes_data.items()):
        if not isinstance(info, dict):
            continue
        age = now - info.get("last_heartbeat", 0)
        status = "online" if age < 60 else "stale" if age < 300 else "offline"
        backends = ", ".join(info.get("backends", []))
        gpu = info.get("gpu_name", "")[:25]
        vram = f"{info.get('vram_mb', 0) // 1024}GB"
        dot = "●" if status == "online" else "○"
        print(f"  {dot} {nid:25s} {status:8s} {backends:20s} {vram:>5s} {gpu}")


def cmd_models(args):
    """Show all models across all nodes — the unified catalog."""
    from server.node_agent import _firebase_get

    models = _firebase_get("mamba3/models") or {}
    teachers = (_firebase_get("mamba3/three_pop/teachers") or {})

    if not models:
        print("No models in catalog. Run 'python server/push_state.py' on a training node.")
        return

    print(f"{'Task':25s} {'Size':>8s} {'Acc':>5s} {'Node':20s} {'Status':>10s}")
    print("-" * 75)
    for task in sorted(models.keys()):
        m = models[task]
        t = teachers.get(task, {})
        acc = t.get("accuracy", 0)
        acc_s = f"{acc:.0%}" if isinstance(acc, (int, float)) and acc > 0 else "  -"
        node = str(m.get("node", "?"))[:20]
        size = f"{m.get('size_kb', 0):.0f}KB"
        status = "teacher" if task in teachers else "training"
        print(f"{task:25s} {size:>8s} {acc_s:>5s} {node:20s} {status:>10s}")
    print(f"\nTotal: {len(models)} models, {len(teachers)} teachers")


def cmd_run(args):
    """Run inference on any model from the catalog."""
    import subprocess

    # Resolve checkpoint — local or fetch from catalog
    ckpt = args.checkpoint
    if not Path(ckpt).exists():
        # Try checkpoints/specialists/<name>.pt
        ckpt = f"checkpoints/specialists/{args.checkpoint}.pt"
    if not Path(ckpt).exists():
        # Try fetching via ModelRegistry
        try:
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from registry.model_registry import ModelRegistry
            reg = ModelRegistry()
            local = reg.ensure_local(args.checkpoint)
            if local:
                ckpt = str(local)
            else:
                print(f"Model '{args.checkpoint}' not found locally or in catalog.")
                return
        except Exception as e:
            print(f"Error: {e}")
            return

    cmd = [sys.executable, str(Path(__file__).parent.parent / "export" / "mamba_inference.py"),
           "--checkpoint", ckpt, "--device", args.device or "cpu",
           "--problems-dir", "problems"]

    if args.eval:
        cmd.extend(["--eval", "--n-examples", str(args.n or 100)])
    elif args.bench:
        cmd.append("--bench")
    elif args.serve:
        cmd.extend(["--serve", "--port", str(args.port or 8090)])
    else:
        cmd.append("--eval")

    subprocess.run(cmd)


def cmd_lr(args):
    """Show learning rate — is the system getting faster at mastering tasks?"""
    from server.node_agent import _firebase_get

    data = _firebase_get("mamba3/learning_rate")
    if not data:
        print("No learning rate data yet. Need at least 2 graduated teachers.")
        return

    ratio = data.get("learning_ratio")
    first = data.get("first_task", "?")
    first_c = data.get("first_cycles", 0)
    last = data.get("last_task", "?")
    last_c = data.get("last_cycles", 0)

    print("=== Learning Rate ===")
    if ratio:
        label = "accelerating" if ratio < 1 else "slowing" if ratio > 1 else "steady"
        print(f"  Ratio: {ratio:.3f} ({label})")
    print(f"  First task: {first} ({first_c} cycles)")
    print(f"  Last task:  {last} ({last_c} cycles)")

    tasks = data.get("tasks", {})
    if tasks:
        ordered = sorted(tasks.items(), key=lambda x: x[1].get("order", 0))
        print(f"\n{'#':>3s} {'Task':25s} {'Cycles':>8s} {'Speedup':>8s}")
        print("-" * 50)
        for task, info in ordered:
            cycles = info.get("cycles", 0)
            speedup = info.get("speedup_vs_first", 1.0)
            bar = "█" * min(int(speedup), 20)
            print(f"{info.get('order',0):>3d} {task:25s} {cycles:>8d} {speedup:>7.1f}x {bar}")


def cmd_teachers(args):
    """Show all graduated teachers across all nodes."""
    from server.node_agent import _firebase_get, list_nodes

    three_pop = _firebase_get("mamba3/three_pop") or {}
    teachers = three_pop.get("teachers", {})

    if not teachers:
        print("No teachers registered yet.")
        return

    print(f"{'Task':25s} {'Accuracy':>8s} {'Node':20s}")
    print("-" * 58)
    for task in sorted(teachers.keys()):
        info = teachers[task]
        acc = info.get("accuracy", 0)
        acc_str = f"{acc:.0%}" if acc <= 1 else f"{acc:.0f}%"
        node = info.get("node_id", "unknown")
        print(f"{task:25s} {acc_str:>8s} {node:20s}")

    print(f"\nTotal: {len(teachers)} teachers")

    # Show which nodes have which checkpoints
    nodes = list_nodes()
    online = {nid: info for nid, info in nodes.items()
              if isinstance(info, dict) and info.get("status") == "online"}
    if online:
        print(f"\nTo sync teachers to a node: mamba sync --source <node> --dest <node>")


def _resolve_node(target_name):
    """Resolve a target name to a node dict with SSH info."""
    from server.node_agent import list_nodes
    nodes = list_nodes()

    for nid, info in nodes.items():
        if not isinstance(info, dict):
            continue
        if target_name in nid or target_name == info.get("hostname"):
            info["node_id"] = nid
            return info

    print(f"Node '{target_name}' not found. Available nodes:")
    for nid in sorted(nodes.keys()):
        if isinstance(nodes[nid], dict):
            print(f"  {nid}")
    return None


def _ssh_cmd(ssh_info, command, timeout=30):
    """Run a command on a remote node via SSH. Returns (returncode, stdout, stderr)."""
    import subprocess
    host = ssh_info["host"]
    port = ssh_info.get("port", 22)
    user = ssh_info.get("user", "root")
    cmd = ["ssh", "-p", str(port), "-o", "StrictHostKeyChecking=no",
           f"{user}@{host}", command]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return result.returncode, result.stdout, result.stderr


def cmd_submit(args):
    """Deploy problems + code to a target node and start training."""
    import subprocess
    import hashlib
    import time

    target = _resolve_node(args.target)
    if not target:
        return

    ssh = target.get("ssh")
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

    # Step 1: Commit and push locally, then pull on remote
    # This is more reliable than rsync with complex filters
    print(f"  Syncing code via git...")
    rc, out, err = _ssh_cmd(ssh, f"cd {remote_dir} && git pull", timeout=60)
    if rc != 0:
        print(f"  git pull failed: {err}")
        # Fallback: rsync essential dirs
        src_dir = str(Path(__file__).parent.parent)
        rsync_cmd = [
            "rsync", "-avz",
            "-e", f"ssh -p {port} -o StrictHostKeyChecking=no",
            f"{src_dir}/problems/", f"{user}@{host}:{remote_dir}/problems/",
        ]
        subprocess.run(rsync_cmd, capture_output=True)
        rsync_cmd[-2] = f"{src_dir}/generators/"
        rsync_cmd[-1] = f"{user}@{host}:{remote_dir}/generators/"
        subprocess.run(rsync_cmd, capture_output=True)
        rsync_cmd[-2] = f"{src_dir}/registry/"
        rsync_cmd[-1] = f"{user}@{host}:{remote_dir}/registry/"
        subprocess.run(rsync_cmd, capture_output=True)
        print(f"  Fell back to rsync")
    else:
        print(f"  {out.strip().splitlines()[-1] if out.strip() else 'Up to date'}")

    # Step 2: Generate job ID
    job_id = f"j-{time.strftime('%Y%m%d-%H%M%S')}-{hashlib.md5(target['node_id'].encode()).hexdigest()[:6]}"

    # Step 3: Detect remote Python (venv or system)
    rc, py_out, _ = _ssh_cmd(ssh, f"test -f {remote_dir}/.venv/bin/python && echo {remote_dir}/.venv/bin/python || which python3")
    remote_python = py_out.strip().splitlines()[-1] if py_out.strip() else "python3"

    # Step 4: Start training
    problems_dir = args.problems or "problems"
    start_cmd = (
        f"cd {remote_dir} && "
        f"nohup {remote_python} -u three_populations.py "
        f"--problems-dir {problems_dir} "
        f"--job-id {job_id} "
        f"> three_pop.log 2>&1 & echo $!"
    )

    print(f"  Starting training (job: {job_id})...")
    rc, out, err = _ssh_cmd(ssh, start_cmd)
    if rc == 0:
        pid = out.strip().splitlines()[-1] if out.strip() else "?"
        print(f"  Training started (PID: {pid})")
        print(f"  Job ID: {job_id}")
        print(f"  Monitor: mamba logs --node {target['node_id']}")
        print(f"  Dashboard: https://gauchoai.github.io/mamba3-hands-on/")

        # Register job in Firebase
        from server.node_agent import _firebase_put
        _firebase_put(f"mamba3/jobs/{job_id}", {
            "node_id": target["node_id"],
            "pid": pid,
            "problems_dir": problems_dir,
            "submitted_at": time.time(),
            "status": "running",
        })
    else:
        print(f"  Failed to start: {err}")


def cmd_logs(args):
    """Stream logs from a training node."""
    import subprocess

    target = _resolve_node(args.node)
    if not target:
        return

    ssh = target.get("ssh")
    if not ssh:
        print("No SSH info for this node.")
        return

    host = ssh["host"]
    port = ssh.get("port", 22)
    user = ssh.get("user", "root")
    remote_dir = target.get("working_dir", "/root/mamba3-hands-on")
    lines = args.lines or 50

    cmd = ["ssh", "-p", str(port), "-o", "StrictHostKeyChecking=no",
           f"{user}@{host}",
           f"tail -{lines} {remote_dir}/three_pop.log"]

    if args.follow:
        cmd[-1] = f"tail -f {remote_dir}/three_pop.log"
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
    target = _resolve_node(args.node)
    if not target:
        return

    ssh = target.get("ssh")
    if not ssh:
        print("No SSH info for this node.")
        return

    print(f"Stop training on {target['node_id']}? Workers will save checkpoints.")
    confirm = input("Type 'yes' to confirm: ")
    if confirm.lower() != "yes":
        print("Cancelled.")
        return

    rc, out, err = _ssh_cmd(ssh, "pkill -TERM -f three_populations.py")
    if rc == 0:
        print(f"  SIGTERM sent. Workers will checkpoint and exit.")
    else:
        print(f"  Error: {err}")


def cmd_pull(args):
    """Download checkpoints from a training node."""
    import subprocess

    target = _resolve_node(args.node)
    if not target:
        return

    ssh = target.get("ssh")
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
        "-e", f"ssh -p {port} -o StrictHostKeyChecking=no",
        f"{user}@{host}:{remote_dir}/checkpoints/specialists/",
        f"{output}/"
    ]

    print(f"Pulling checkpoints from {target['node_id']} → {output}/")
    result = subprocess.run(rsync_cmd, text=True)
    if result.returncode == 0:
        print("Done.")
    else:
        print("Pull failed.")


def cmd_card(args):
    """Show full model card with hardware provenance, teachers, lineage."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from state_db import StateDB

    db = StateDB(args.db)
    card = db.build_model_card(args.task)
    db.close()

    if not card.get("config"):
        print(f"No data for task '{args.task}'")
        return

    cfg = card["config"]
    print(f"=== Model Card: {card['task']} ===")
    print(f"  Best accuracy: {card['best_accuracy']:.0%}")
    print(f"  Total rounds:  {card['total_rounds']}")
    print()

    # Confidence
    conf = card.get("confidence", {})
    if conf.get("n_samples", 0) > 0:
        print(f"  Confidence: {conf['score']:.2%} (mean={conf['mean']:.0%} std={conf['std']:.2f} n={conf['n_samples']})")

    # Architecture
    print(f"\n--- Architecture ---")
    print(f"  d_model:    {cfg.get('d_model', '?')}")
    print(f"  layers:     {cfg.get('n_kernel_layers', '?')}")
    print(f"  d_state:    {cfg.get('d_state', '?')}")
    print(f"  headdim:    {cfg.get('headdim', '?')}")
    n_params = cfg.get("d_model", 64) * 260 * 2  # rough estimate
    print(f"  optimizer:  {cfg.get('optimizer', '?')}")
    print(f"  loss_fn:    {cfg.get('loss_fn', '?')}")
    print(f"  lr:         {cfg.get('lr', '?')}")

    # Hardware provenance
    hw = card.get("hardware", {})
    if hw:
        print(f"\n--- Hardware Provenance ---")
        print(f"  Device:       {hw.get('device', '?')}")
        print(f"  Backend:      {hw.get('scan_backend', '?')}")
        print(f"  Total cycles: {hw.get('total_cycles', '?')}")
        if hw.get("total_time_s"):
            t = hw["total_time_s"]
            if t > 3600:
                print(f"  Training time: {t/3600:.1f} hours")
            elif t > 60:
                print(f"  Training time: {t/60:.1f} minutes")
            else:
                print(f"  Training time: {t:.0f} seconds")

    # Distillation
    prov = card.get("provenance", {})
    distilled = prov.get("distilled_from", {})
    if distilled:
        print(f"\n--- Distillation ---")
        print(f"  Distilled from: {distilled.get('value', '?')}")

    # Teachers
    teachers = card.get("teachers", [])
    if teachers:
        print(f"\n--- Teachers ({len(teachers)}) ---")
        for t in teachers[:10]:
            print(f"  {t['model']:30s} weight={t['weight']:.3f} (from round {t['from_round']})")

    # Diagnostics
    diag = card.get("diagnostics", {}).get("stats", [])
    if diag:
        print(f"\n--- Diagnostics ---")
        for d in diag[:5]:
            print(f"  {d.get('signal', '?'):20s} tried={d.get('tries', 0)} wins={d.get('wins', 0)}")


def cmd_sync(args):
    """Sync teacher checkpoints between nodes for cross-node distillation."""
    import subprocess

    source = _resolve_node(args.source)
    if not source:
        return

    ssh_src = source.get("ssh")
    if not ssh_src:
        print(f"Source node has no SSH info.")
        return

    src_host = ssh_src["host"]
    src_port = ssh_src.get("port", 22)
    src_user = ssh_src.get("user", "root")
    src_dir = source.get("working_dir", "/root/mamba3-hands-on")

    if args.dest:
        dest = _resolve_node(args.dest)
        if not dest:
            return
        ssh_dst = dest.get("ssh")
        if not ssh_dst:
            print("Dest node has no SSH info.")
            return
        dst_host = ssh_dst["host"]
        dst_port = ssh_dst.get("port", 22)
        dst_user = ssh_dst.get("user", "root")
        dst_dir = dest.get("working_dir", "/root/mamba3-hands-on")

        # Pull from source to local temp, then push to dest
        local_tmp = "/tmp/mamba_sync_checkpoints/"
        os.makedirs(local_tmp, exist_ok=True)

        print(f"Syncing teachers: {source['node_id']} → {dest['node_id']}")
        # Pull
        pull_cmd = [
            "rsync", "-avz",
            "-e", f"ssh -p {src_port} -o StrictHostKeyChecking=no",
            f"{src_user}@{src_host}:{src_dir}/checkpoints/specialists/",
            local_tmp
        ]
        r = subprocess.run(pull_cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"  Pull failed: {r.stderr}")
            return
        n_files = len([f for f in os.listdir(local_tmp) if f.endswith(".pt")])
        print(f"  Pulled {n_files} checkpoints")

        # Push
        push_cmd = [
            "rsync", "-avz",
            "-e", f"ssh -p {dst_port} -o StrictHostKeyChecking=no",
            local_tmp,
            f"{dst_user}@{dst_host}:{dst_dir}/checkpoints/specialists/"
        ]
        r = subprocess.run(push_cmd, capture_output=True, text=True)
        if r.returncode == 0:
            print(f"  Pushed to {dest['node_id']}. Teachers available for distillation.")
        else:
            print(f"  Push failed: {r.stderr}")
    else:
        # Pull to local
        local_dir = args.output or "checkpoints/specialists/"
        os.makedirs(local_dir, exist_ok=True)
        print(f"Pulling teachers from {source['node_id']} → {local_dir}")
        pull_cmd = [
            "rsync", "-avz",
            "-e", f"ssh -p {src_port} -o StrictHostKeyChecking=no",
            f"{src_user}@{src_host}:{src_dir}/checkpoints/specialists/",
            f"{local_dir}/"
        ]
        r = subprocess.run(pull_cmd, text=True)
        if r.returncode == 0:
            n = len([f for f in os.listdir(local_dir) if f.endswith(".pt")])
            print(f"  Done. {n} checkpoints available locally.")


def main():

    parser = argparse.ArgumentParser(
        prog="mamba",
        description="Mamba training platform CLI",
    )
    sub = parser.add_subparsers(dest="command", help="Command")

    # nodes
    p_nodes = sub.add_parser("nodes", help="List registered training nodes")

    # models
    p_models = sub.add_parser("models", help="Show all models in unified catalog")

    # run
    p_run = sub.add_parser("run", help="Run inference on any model from catalog")
    p_run.add_argument("checkpoint", help="Task name or path to .pt checkpoint")
    p_run.add_argument("--device", default="cpu")
    p_run.add_argument("--eval", action="store_true", help="Evaluate on fresh examples")
    p_run.add_argument("--bench", action="store_true", help="Benchmark throughput")
    p_run.add_argument("--serve", action="store_true", help="Start HTTP server")
    p_run.add_argument("--port", type=int, default=8090)
    p_run.add_argument("-n", type=int, default=100, help="Number of examples")

    # lr
    p_lr = sub.add_parser("lr", help="Show learning rate — is the system accelerating?")

    # teachers
    p_teachers = sub.add_parser("teachers", help="List all graduated teachers across nodes")

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

    # card
    p_card = sub.add_parser("card", help="Show model card with full provenance")
    p_card.add_argument("--task", type=str, required=True)
    p_card.add_argument("--db", type=str, default="three_pop/training.db")

    # sync
    p_sync = sub.add_parser("sync", help="Sync teacher checkpoints between nodes")
    p_sync.add_argument("--source", type=str, required=True, help="Source node ID")
    p_sync.add_argument("--dest", type=str, default=None, help="Destination node (omit for local)")
    p_sync.add_argument("--output", type=str, default=None, help="Local output dir")

    args = parser.parse_args()

    if args.command == "nodes":
        cmd_nodes(args)
    elif args.command == "models":
        cmd_models(args)
    elif args.command == "run":
        cmd_run(args)
    elif args.command == "lr":
        cmd_lr(args)
    elif args.command == "teachers":
        cmd_teachers(args)
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
    elif args.command == "sync":
        cmd_sync(args)
    elif args.command == "card":
        cmd_card(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
