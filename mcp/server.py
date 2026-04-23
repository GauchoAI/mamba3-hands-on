#!/usr/bin/env python3
"""
Mamba MCP Server — expose training platform as MCP tools for Claude Code.

Tools:
  - mamba_nodes: List registered training nodes
  - mamba_status: Show task training status across nodes
  - mamba_register_problem: Register a new problem (generator or dataset)
  - mamba_submit: Deploy and start training on a node
  - mamba_logs: View recent logs from a training node
  - mamba_pull: Download checkpoints from a training node

Usage:
  python mcp/server.py                    # stdio transport (for Claude Code)

Configure in Claude Code:
  Add to ~/.claude/claude_desktop_config.json or project .mcp.json
"""

import json
import sys
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = str(Path(__file__).parent.parent)
sys.path.insert(0, PROJECT_ROOT)


def _read_stdin():
    """Read a JSON-RPC message from stdin."""
    # Read Content-Length header
    headers = {}
    while True:
        line = sys.stdin.readline()
        if not line or line.strip() == "":
            break
        if ":" in line:
            key, _, val = line.partition(":")
            headers[key.strip().lower()] = val.strip()

    content_length = int(headers.get("content-length", 0))
    if content_length == 0:
        return None

    body = sys.stdin.read(content_length)
    return json.loads(body)


def _write_stdout(msg):
    """Write a JSON-RPC message to stdout."""
    body = json.dumps(msg)
    sys.stdout.write(f"Content-Length: {len(body)}\r\n\r\n{body}")
    sys.stdout.flush()


def _success(id, result):
    return {"jsonrpc": "2.0", "id": id, "result": result}


def _error(id, code, message):
    return {"jsonrpc": "2.0", "id": id, "error": {"code": code, "message": message}}


# ── Tool implementations ──────────────────────────────────────────

def tool_mamba_nodes():
    """List all registered training nodes with their capabilities."""
    from server.node_agent import list_nodes
    import time

    nodes = list_nodes()
    now = time.time()
    result = []

    for nid, info in sorted(nodes.items()):
        if not isinstance(info, dict):
            continue
        last_hb = info.get("last_heartbeat", 0)
        age = now - last_hb

        result.append({
            "node_id": nid,
            "status": "online" if age < 60 else ("stale" if age < 300 else "offline"),
            "backends": info.get("backends", []),
            "vram_mb": info.get("vram_mb", 0),
            "gpu_name": info.get("gpu_name", ""),
            "hostname": info.get("hostname", ""),
            "last_seen_seconds_ago": int(age),
        })

    return result


def tool_mamba_status():
    """Show task training status from Firebase."""
    from server.node_agent import _firebase_get

    three_pop = _firebase_get("mamba3/three_pop") or {}
    teachers = three_pop.get("teachers", {})
    tasks_data = three_pop.get("tasks", {})

    tasks = []
    all_names = set(list(teachers.keys()) + list(tasks_data.keys()))
    for name in sorted(all_names):
        teacher = teachers.get(name, {})
        task_info = tasks_data.get(name, {})
        acc = teacher.get("accuracy") or task_info.get("best_accuracy", 0)
        status = "mastered" if teacher else task_info.get("status", "unknown")
        tasks.append({"task": name, "accuracy": acc, "status": status})

    nodes = tool_mamba_nodes()
    online = [n for n in nodes if n["status"] == "online"]

    return {
        "tasks": tasks,
        "mastered": sum(1 for t in tasks if t["status"] == "mastered"),
        "total": len(tasks),
        "nodes_online": len(online),
        "nodes_total": len(nodes),
    }


def tool_mamba_register_problem(name, generator_module, generator_function,
                                 target_accuracy=0.95, tags=None, description="",
                                 problem_type="generator", dataset_path=None):
    """Register a new problem by creating its YAML manifest.

    After registering, submit a job to start training on it.
    """
    problems_dir = Path(PROJECT_ROOT) / "problems" / name
    problems_dir.mkdir(parents=True, exist_ok=True)

    if problem_type == "generator":
        yaml_content = (
            f"name: {name}\n"
            f"type: generator\n"
            f"generator: {generator_module}:{generator_function}\n"
            f"target_accuracy: {target_accuracy}\n"
            f"tags: [{', '.join(tags or ['level0'])}]\n"
            f'description: "{description}"\n'
        )
    else:
        yaml_content = (
            f"name: {name}\n"
            f"type: dataset\n"
            f"dataset: {dataset_path}\n"
            f"target_accuracy: {target_accuracy}\n"
            f"tags: [{', '.join(tags or ['level0'])}]\n"
            f'description: "{description}"\n'
        )

    yaml_path = problems_dir / "problem.yaml"
    yaml_path.write_text(yaml_content)

    # Verify the generator can be loaded
    if problem_type == "generator":
        try:
            from registry.problem_registry import ProblemRegistry
            reg = ProblemRegistry()
            reg.discover([str(Path(PROJECT_ROOT) / "problems")])
            gen = reg.get_generator(name)
            example = gen()
            return {
                "status": "registered",
                "path": str(yaml_path),
                "example": example,
                "total_problems": len(reg.problems),
            }
        except Exception as e:
            return {"status": "error", "message": str(e), "path": str(yaml_path)}
    else:
        return {"status": "registered", "path": str(yaml_path)}


def tool_mamba_submit(target_node, problems_dir="problems"):
    """Deploy code and start training on a target node."""
    import subprocess
    import hashlib
    import time

    from server.node_agent import list_nodes, _firebase_put

    nodes = list_nodes()
    target = None
    for nid, info in nodes.items():
        if not isinstance(info, dict):
            continue
        if target_node in nid:
            target = info
            target["node_id"] = nid
            break

    if not target:
        return {"status": "error", "message": f"Node '{target_node}' not found",
                "available": [nid for nid in nodes.keys()]}

    ssh = target.get("ssh")
    if not ssh:
        return {"status": "error", "message": "Node has no SSH info"}

    host = ssh["host"]
    port = ssh.get("port", 22)
    user = ssh.get("user", "root")
    remote_dir = target.get("working_dir", "/root/mamba3-hands-on")

    # Git pull on remote
    ssh_cmd = ["ssh", "-p", str(port), "-o", "StrictHostKeyChecking=no",
               f"{user}@{host}", f"cd {remote_dir} && git pull"]
    subprocess.run(ssh_cmd, capture_output=True, timeout=60)

    # Detect remote Python
    ssh_cmd = ["ssh", "-p", str(port), "-o", "StrictHostKeyChecking=no",
               f"{user}@{host}",
               f"test -f {remote_dir}/.venv/bin/python && echo {remote_dir}/.venv/bin/python || which python3"]
    result = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=15)
    remote_python = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else "python3"

    # Generate job ID and start
    job_id = f"j-{time.strftime('%Y%m%d-%H%M%S')}-{hashlib.md5(target['node_id'].encode()).hexdigest()[:6]}"

    start_cmd = (
        f"cd {remote_dir} && "
        f"nohup {remote_python} -u three_populations.py "
        f"--problems-dir {problems_dir} "
        f"--job-id {job_id} "
        f"> three_pop.log 2>&1 & echo $!"
    )
    ssh_cmd = ["ssh", "-p", str(port), "-o", "StrictHostKeyChecking=no",
               f"{user}@{host}", start_cmd]
    result = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=30)

    if result.returncode == 0:
        pid = result.stdout.strip().splitlines()[-1]
        _firebase_put(f"mamba3/jobs/{job_id}", {
            "node_id": target["node_id"],
            "pid": pid,
            "problems_dir": problems_dir,
            "submitted_at": time.time(),
            "status": "running",
        })
        return {
            "status": "submitted",
            "job_id": job_id,
            "node": target["node_id"],
            "pid": pid,
            "monitor": f"mamba logs --node {target['node_id']}",
        }
    else:
        return {"status": "error", "message": result.stderr}


def tool_mamba_logs(node_name, lines=30):
    """View recent training logs from a node."""
    import subprocess

    from server.node_agent import list_nodes
    nodes = list_nodes()
    target = None
    for nid, info in nodes.items():
        if not isinstance(info, dict):
            continue
        if node_name in nid:
            target = info
            break

    if not target:
        return {"status": "error", "message": f"Node '{node_name}' not found"}

    ssh = target.get("ssh")
    if not ssh:
        return {"status": "error", "message": "No SSH info"}

    host = ssh["host"]
    port = ssh.get("port", 22)
    user = ssh.get("user", "root")
    remote_dir = target.get("working_dir", "/root/mamba3-hands-on")

    cmd = ["ssh", "-p", str(port), "-o", "StrictHostKeyChecking=no",
           f"{user}@{host}", f"tail -{lines} {remote_dir}/three_pop.log"]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)

    if result.returncode == 0:
        return {"status": "ok", "logs": result.stdout}
    else:
        return {"status": "error", "message": result.stderr}


# ── MCP Protocol Handler ──────────────────────────────────────────

TOOLS = [
    {
        "name": "mamba_nodes",
        "description": "List all registered Mamba training nodes with capabilities (GPU, backends, VRAM)",
        "inputSchema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "mamba_status",
        "description": "Show training status: which tasks are mastered, accuracy levels, active nodes",
        "inputSchema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "mamba_register_problem",
        "description": "Register a new training problem (task). Creates a YAML manifest pointing to a generator function or dataset. After registering, use mamba_submit to start training.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Problem name (e.g., 'tower_of_hanoi')"},
                "generator_module": {"type": "string", "description": "Python module path (e.g., 'generators.level1_reasoning')"},
                "generator_function": {"type": "string", "description": "Function name (e.g., 'gen_tower_of_hanoi')"},
                "target_accuracy": {"type": "number", "description": "Target accuracy for mastery (default 0.95)"},
                "tags": {"type": "array", "items": {"type": "string"}, "description": "Tags for categorization"},
                "description": {"type": "string", "description": "Human-readable description"},
            },
            "required": ["name", "generator_module", "generator_function"],
        },
    },
    {
        "name": "mamba_submit",
        "description": "Deploy code and start training on a target node. Syncs via git pull, starts three_populations.py orchestrator.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "target_node": {"type": "string", "description": "Node ID or partial match (e.g., 'h100')"},
                "problems_dir": {"type": "string", "description": "Problems directory (default 'problems')"},
            },
            "required": ["target_node"],
        },
    },
    {
        "name": "mamba_logs",
        "description": "View recent training logs from a node",
        "inputSchema": {
            "type": "object",
            "properties": {
                "node_name": {"type": "string", "description": "Node ID or partial match"},
                "lines": {"type": "integer", "description": "Number of log lines to return (default 30)"},
            },
            "required": ["node_name"],
        },
    },
]


def handle_request(msg):
    """Handle a JSON-RPC request."""
    method = msg.get("method", "")
    id = msg.get("id")
    params = msg.get("params", {})

    if method == "initialize":
        return _success(id, {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "mamba-platform", "version": "0.1.0"},
        })

    elif method == "notifications/initialized":
        return None  # no response needed

    elif method == "tools/list":
        return _success(id, {"tools": TOOLS})

    elif method == "tools/call":
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})

        try:
            if tool_name == "mamba_nodes":
                result = tool_mamba_nodes()
            elif tool_name == "mamba_status":
                result = tool_mamba_status()
            elif tool_name == "mamba_register_problem":
                result = tool_mamba_register_problem(**arguments)
            elif tool_name == "mamba_submit":
                result = tool_mamba_submit(**arguments)
            elif tool_name == "mamba_logs":
                result = tool_mamba_logs(**arguments)
            else:
                return _error(id, -32601, f"Unknown tool: {tool_name}")

            return _success(id, {
                "content": [{"type": "text", "text": json.dumps(result, indent=2, default=str)}]
            })
        except Exception as e:
            return _success(id, {
                "content": [{"type": "text", "text": f"Error: {e}"}],
                "isError": True,
            })

    elif method == "ping":
        return _success(id, {})

    else:
        return _error(id, -32601, f"Method not found: {method}")


def main():
    """Run MCP server on stdio."""
    sys.stderr.write("Mamba MCP server starting (stdio transport)\n")

    while True:
        try:
            msg = _read_stdin()
            if msg is None:
                break

            response = handle_request(msg)
            if response is not None:
                _write_stdout(response)

        except (EOFError, KeyboardInterrupt):
            break
        except Exception as e:
            sys.stderr.write(f"MCP error: {e}\n")


if __name__ == "__main__":
    main()
