"""Node Agent — auto-probes hardware, registers to Firebase, runs heartbeat.

Each training node runs this agent. It:
1. Probes local hardware (GPU, CPU, backends available)
2. Registers itself to Firebase so the CLI and other nodes can discover it
3. Sends heartbeats every 30s so stale nodes get pruned
4. Optionally starts the three_populations.py orchestrator

Usage:
    python server/node_agent.py --probe          # just print capabilities
    python server/node_agent.py --register       # register + heartbeat loop
    python server/node_agent.py --start          # register + start training
"""

import json
import os
import platform
import socket
import subprocess
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

FIREBASE_URL = "https://signaling-dcfad-default-rtdb.europe-west1.firebasedatabase.app"
HEARTBEAT_INTERVAL = 30  # seconds


def _firebase_put(path, data):
    """PUT data to Firebase Realtime DB."""
    url = f"{FIREBASE_URL}/{path}.json"
    body = json.dumps(data).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="PUT",
                                headers={"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(req, timeout=5)
        return True
    except Exception as e:
        print(f"  Firebase PUT failed: {e}", flush=True)
        return False


def _firebase_get(path):
    """GET data from Firebase Realtime DB."""
    url = f"{FIREBASE_URL}/{path}.json"
    try:
        resp = urllib.request.urlopen(url, timeout=5)
        return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None


def _firebase_delete(path):
    """DELETE data from Firebase Realtime DB."""
    url = f"{FIREBASE_URL}/{path}.json"
    req = urllib.request.Request(url, method="DELETE")
    try:
        urllib.request.urlopen(req, timeout=5)
        return True
    except Exception:
        return False


def probe_capabilities() -> dict:
    """Auto-detect hardware capabilities of this machine."""
    caps = {
        "hostname": socket.gethostname(),
        "platform": platform.system().lower(),
        "arch": platform.machine(),
        "python": platform.python_version(),
        "backends": [],
        "vram_mb": 0,
        "cpu_cores": os.cpu_count() or 1,
        "max_workers": max(1, (os.cpu_count() or 2) // 2),
    }

    # Check CUDA
    try:
        import torch
        if torch.cuda.is_available():
            caps["backends"].append("cuda")
            caps["vram_mb"] = int(torch.cuda.get_device_properties(0).total_mem / 1024 / 1024)
            caps["gpu_name"] = torch.cuda.get_device_name(0)
            # Check Triton
            try:
                import triton
                caps["backends"].append("triton")
            except ImportError:
                pass
    except ImportError:
        pass

    # Check MPS (Apple Silicon)
    try:
        import torch
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            caps["backends"].append("mps")
            # Apple Silicon shared memory — report system RAM
            try:
                import psutil
                caps["vram_mb"] = int(psutil.virtual_memory().total / 1024 / 1024)
            except ImportError:
                # Fallback: read from sysctl
                try:
                    out = subprocess.check_output(["/usr/sbin/sysctl", "-n", "hw.memsize"]).decode().strip()
                    caps["vram_mb"] = int(int(out) / 1024 / 1024)
                except Exception:
                    pass
    except ImportError:
        pass

    # On macOS without torch, still detect RAM
    if caps["vram_mb"] == 0 and caps["platform"] == "darwin":
        try:
            out = subprocess.check_output(["/usr/sbin/sysctl", "-n", "hw.memsize"]).decode().strip()
            caps["vram_mb"] = int(int(out) / 1024 / 1024)
        except Exception:
            pass

    # JIT always available (PyTorch native)
    caps["backends"].append("jit")

    # CPU always available
    if "cpu" not in caps["backends"]:
        caps["backends"].append("cpu")

    # Check PyTorch version
    try:
        import torch
        caps["torch_version"] = torch.__version__
    except ImportError:
        caps["torch_version"] = "not installed"

    return caps


def generate_node_id(caps: dict) -> str:
    """Generate a stable node ID from hostname + platform."""
    hostname = caps["hostname"].lower().replace(".", "-").replace(" ", "-")
    # Shorten common patterns
    if "mac" in hostname or "macbook" in hostname:
        hw = "mac"
    elif "gpu" in hostname or "vast" in hostname:
        hw = "gpu"
    else:
        hw = caps["arch"][:6]

    # Add primary backend
    primary = "cuda" if "cuda" in caps["backends"] else (
        "mps" if "mps" in caps["backends"] else "cpu")

    return f"{hostname}-{primary}"


def register_node(node_id: str, caps: dict, working_dir: str = None):
    """Register this node to Firebase so other nodes and CLI can discover it."""
    manifest = {
        "node_id": node_id,
        "backends": caps["backends"],
        "vram_mb": caps["vram_mb"],
        "cpu_cores": caps["cpu_cores"],
        "max_workers": caps["max_workers"],
        "hostname": caps["hostname"],
        "platform": caps["platform"],
        "arch": caps["arch"],
        "python": caps["python"],
        "torch_version": caps.get("torch_version", "unknown"),
        "gpu_name": caps.get("gpu_name", ""),
        "working_dir": working_dir or str(Path.cwd()),
        "status": "online",
        "registered_at": time.time(),
        "last_heartbeat": time.time(),
        "pid": os.getpid(),
    }

    # Try to detect SSH connection info for remote access
    ssh_info = _detect_ssh_info()
    if ssh_info:
        manifest["ssh"] = ssh_info

    ok = _firebase_put(f"mamba3/nodes/{node_id}", manifest)
    if ok:
        print(f"  Registered node '{node_id}' to Firebase", flush=True)
    return manifest


def _detect_ssh_info() -> dict | None:
    """Try to detect how this node can be reached via SSH."""
    # Check if running on vast.ai
    if os.path.exists("/etc/instance.crt"):
        # Vast.ai instance — SSH info from environment
        return {
            "host": "ssh2.vast.ai",
            "port": int(os.environ.get("SSH_PORT", 32783)),
            "user": "root",
        }

    # Local machine — try to get LAN IP
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return {
            "host": ip,
            "port": 22,
            "user": os.environ.get("USER", "root"),
        }
    except Exception:
        return None


def heartbeat_loop(node_id: str, caps: dict):
    """Send heartbeats to Firebase. Run in foreground."""
    print(f"  Heartbeat loop started (every {HEARTBEAT_INTERVAL}s)", flush=True)
    while True:
        try:
            _firebase_put(f"mamba3/nodes/{node_id}/last_heartbeat", time.time())
            _firebase_put(f"mamba3/nodes/{node_id}/status", "online")
        except Exception as e:
            print(f"  Heartbeat error: {e}", flush=True)
        time.sleep(HEARTBEAT_INTERVAL)


def deregister_node(node_id: str):
    """Mark node as offline in Firebase."""
    _firebase_put(f"mamba3/nodes/{node_id}/status", "offline")
    print(f"  Node '{node_id}' marked offline", flush=True)


def list_nodes() -> dict:
    """Fetch all registered nodes from Firebase."""
    return _firebase_get("mamba3/nodes") or {}


def get_online_nodes() -> list[dict]:
    """Return only nodes with recent heartbeats (< 2 min)."""
    nodes = list_nodes()
    now = time.time()
    online = []
    for nid, info in nodes.items():
        if not isinstance(info, dict):
            continue
        last_hb = info.get("last_heartbeat", 0)
        if now - last_hb < 120:  # 2 minute timeout
            info["node_id"] = nid
            online.append(info)
    return online


# ── CLI entry point ───────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import signal as sig

    parser = argparse.ArgumentParser(description="Mamba training node agent")
    parser.add_argument("--probe", action="store_true", help="Just print capabilities and exit")
    parser.add_argument("--register", action="store_true", help="Register to Firebase + heartbeat")
    parser.add_argument("--start", action="store_true", help="Register + start three_populations.py")
    parser.add_argument("--node-id", type=str, default=None, help="Override node ID")
    parser.add_argument("--working-dir", type=str, default=None, help="Working directory for training")
    args = parser.parse_args()

    caps = probe_capabilities()

    if args.probe:
        print(json.dumps(caps, indent=2))
        sys.exit(0)

    node_id = args.node_id or generate_node_id(caps)
    wd = args.working_dir or str(Path(__file__).parent.parent)

    print(f"Node: {node_id}", flush=True)
    print(f"  Backends: {caps['backends']}", flush=True)
    print(f"  VRAM: {caps['vram_mb']} MB", flush=True)
    print(f"  Workers: {caps['max_workers']}", flush=True)

    manifest = register_node(node_id, caps, wd)

    # Handle graceful shutdown
    def _shutdown(signum, frame):
        print(f"\n  Shutting down node '{node_id}'...", flush=True)
        deregister_node(node_id)
        sys.exit(0)

    sig.signal(sig.SIGTERM, _shutdown)
    sig.signal(sig.SIGINT, _shutdown)

    if args.start:
        # Start three_populations.py in background
        print(f"  Starting three_populations.py in {wd}...", flush=True)
        proc = subprocess.Popen(
            [sys.executable, "-u", "three_populations.py"],
            cwd=wd,
            stdout=open(Path(wd) / "three_pop.log", "a"),
            stderr=subprocess.STDOUT,
        )
        _firebase_put(f"mamba3/nodes/{node_id}/training_pid", proc.pid)
        print(f"  Training PID: {proc.pid}", flush=True)

    if args.register or args.start:
        heartbeat_loop(node_id, caps)
