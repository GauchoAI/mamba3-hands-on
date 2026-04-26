"""Model Registry — unified view of all teachers across all nodes.

A node doesn't care WHERE a teacher lives. It asks "give me the parity teacher"
and gets it — transparently fetched from whatever node has it.

The registry reads from Firebase (the index) and auto-syncs checkpoints
from remote nodes when needed. To the caller, it looks local.

Usage:
    from registry.model_registry import ModelRegistry
    reg = ModelRegistry()

    # List all teachers (from any node)
    teachers = reg.list_teachers()

    # Get a teacher model — fetched transparently
    model = reg.get_teacher("parity", device="cpu")
    logits = model(input_tokens)

    # Check what's available
    reg.available()  # → ["parity", "logic_gate", ...]
"""

import json
import os
import subprocess
import time
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None


# Firebase for the index
FIREBASE_URL = "https://signaling-dcfad-default-rtdb.europe-west1.firebasedatabase.app"


def _firebase_get(path):
    import urllib.request
    try:
        resp = urllib.request.urlopen(f"{FIREBASE_URL}/{path}.json", timeout=5)
        return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None


class ModelRegistry:
    """Unified model registry — transparently fetches teachers from any node."""

    def __init__(self, local_checkpoint_dir="checkpoints/specialists"):
        self.local_dir = Path(local_checkpoint_dir)
        self.local_dir.mkdir(parents=True, exist_ok=True)
        self._teacher_index = None
        self._index_ttl = 0

    def _refresh_index(self):
        """Fetch teacher index from Firebase (cached for 30s)."""
        now = time.time()
        if self._teacher_index and now < self._index_ttl:
            return self._teacher_index

        data = _firebase_get("mamba3/three_pop/teachers") or {}
        self._teacher_index = data
        self._index_ttl = now + 30
        return data

    def list_models(self) -> dict:
        """List ALL models (checkpoints) across all nodes, not just teachers.
        Returns {task: {path, size_kb, node, available}}."""
        return _firebase_get("mamba3/models") or {}

    def list_teachers(self) -> dict:
        """List all available teachers across all nodes.
        Returns {task: {accuracy, node, config, ...}}."""
        return self._refresh_index()

    def available(self) -> list[str]:
        """List task names that have graduated teachers."""
        return sorted(self._refresh_index().keys())

    def has_teacher(self, task: str) -> bool:
        """Check if a teacher exists for this task (on any node).

        Order of checks:
          1. Local .pt with non-zero accuracy — most authoritative on the
             Mac branch where Firebase may be offline.
          2. Firebase teacher index — for the cluster setup where peer
             Macs publish their mastered checkpoints.
        """
        local_path = self.local_dir / f"{task}.pt"
        if local_path.exists() and torch is not None:
            try:
                ck = torch.load(str(local_path), map_location="cpu", weights_only=False)
                if float(ck.get("accuracy", 0.0)) > 0.0:
                    return True
            except Exception:
                pass
        return task in self._refresh_index()

    def is_local(self, task: str) -> bool:
        """Check if the teacher checkpoint is already local."""
        return (self.local_dir / f"{task}.pt").exists()

    def ensure_local(self, task: str) -> Path | None:
        """Ensure the teacher checkpoint is available locally.
        Fetches from the remote node if needed. Returns local path.

        Mac branch: when no peer is reachable, the local .pt IS the
        teacher — same-task distillation uses the previously-mastered
        checkpoint as a teacher for the new student, which is the
        whole point of the resume-from-master pattern.
        """
        local_path = self.local_dir / f"{task}.pt"
        if local_path.exists():
            return local_path

        # Find which node has it
        index = self._refresh_index()
        teacher_info = index.get(task)
        if not teacher_info:
            return None

        # Find the node's SSH info
        nodes = _firebase_get("mamba3/nodes") or {}
        # Try to find the node that trained this teacher
        source_node = None
        for nid, node_info in nodes.items():
            if not isinstance(node_info, dict):
                continue
            ssh = node_info.get("ssh")
            if ssh:
                source_node = node_info
                source_node["node_id"] = nid
                break  # use first available node with SSH

        if not source_node or not source_node.get("ssh"):
            return None

        # Fetch via rsync
        ssh = source_node["ssh"]
        host = ssh["host"]
        port = ssh.get("port", 22)
        user = ssh.get("user", "root")
        remote_dir = source_node.get("working_dir", "/root/mamba3-hands-on")
        remote_path = f"{remote_dir}/checkpoints/specialists/{task}.pt"

        print(f"  Fetching teacher {task} from {source_node['node_id']}...", flush=True)
        cmd = [
            "rsync", "-az",
            "-e", f"ssh -p {port} -o StrictHostKeyChecking=no",
            f"{user}@{host}:{remote_path}",
            str(local_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode == 0 and local_path.exists():
            print(f"  Fetched {task} teacher ({local_path.stat().st_size // 1024}KB)", flush=True)
            return local_path
        else:
            return None

    def get_teacher(self, task: str, device: str = "cpu"):
        """Load a teacher model, fetching from remote if needed.
        Returns (model, config, accuracy) or None."""
        if torch is None:
            return None

        local_path = self.ensure_local(task)
        if not local_path:
            return None

        try:
            from progressive_model import ProgressiveModel
            ckpt = torch.load(local_path, map_location=device, weights_only=False)
            config = ckpt.get("config", {})
            accuracy = ckpt.get("accuracy", 0)

            model = ProgressiveModel(
                d_model=config.get("d_model", 64),
                d_state=config.get("d_state", 16),
                expand=2,
                headdim=config.get("headdim", 16),
            ).to(device)
            for _ in range(config.get("n_kernel_layers", 1)):
                model.add_kernel_layer()
            model.load_state_dict(ckpt["model"])
            model.eval()

            return model, config, accuracy
        except Exception as e:
            print(f"  Failed to load teacher {task}: {e}", flush=True)
            return None

    def sync_all(self) -> int:
        """Sync all teachers from remote nodes. Returns count synced."""
        index = self._refresh_index()
        synced = 0
        for task in index:
            if not self.is_local(task):
                if self.ensure_local(task):
                    synced += 1
        return synced
