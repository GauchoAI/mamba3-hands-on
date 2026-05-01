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
          1. Local .pt with non-zero accuracy AND non-NaN weights — the
             accuracy field can lie if the file was saved during a NaN
             training run, so we *also* scan state_dict.
          2. Firebase teacher index — for the cluster setup where peer
             Macs publish their mastered checkpoints.
        """
        local_path = self.local_dir / f"{task}.pt"
        if local_path.exists() and torch is not None:
            try:
                ck = torch.load(str(local_path), map_location="cpu", weights_only=False)
                if float(ck.get("accuracy", 0.0)) > 0.0:
                    sd = ck.get("model", {})
                    has_nan = any(
                        torch.isnan(v).any().item() if v.is_floating_point() else False
                        for v in sd.values()
                    )
                    if not has_nan:
                        return True
            except Exception:
                pass
        return task in self._refresh_index()

    def is_local(self, task: str) -> bool:
        """Check if the teacher checkpoint is already local."""
        return (self.local_dir / f"{task}.pt").exists()

    def ensure_local(self, task: str) -> Path | None:
        """Ensure the teacher checkpoint is available locally.

        Order of fallback:
          1. Local .pt — already on disk, nothing to do.
          2. Firebase teacher blob — base64 .pt stored at
             mamba3/teacher_blobs/<task>. Plain HTTPS GET, no SSH server
             needed on any node.
          3. Peer-to-peer SSH/rsync (legacy) — requires SSH server on
             the source node; kept for nodes that pre-date the blob path.
        """
        local_path = self.local_dir / f"{task}.pt"
        if local_path.exists():
            return local_path

        # Try Firebase blob first — works through any NAT, no peer SSH.
        try:
            from mamba_platform.firebase_push import download_teacher_blob
            if download_teacher_blob(task, local_path):
                print(f"  Fetched {task} teacher from Firebase blob "
                      f"({local_path.stat().st_size // 1024}KB)", flush=True)
                return local_path
        except Exception as e:
            print(f"  Firebase blob fetch failed for {task}: {e}", flush=True)

        # Find which node has it
        index = self._refresh_index()
        teacher_info = index.get(task)
        if not teacher_info:
            return None

        # Find the node's SSH info — prefer the FRESHEST heartbeat so we
        # don't try to rsync from a node that's offline. The teacher index
        # records `node` (hostname) but we look up SSH coords by scanning
        # `mamba3/nodes/*`; if multiple nodes are registered with SSH info
        # we pick the one with the most recent heartbeat (within the last
        # 5 minutes), falling back to any reachable node if none are fresh.
        import time as _time
        import socket as _socket
        nodes = _firebase_get("mamba3/nodes") or {}
        # Identify self so we don't try to rsync a file from ourselves
        # (which would silently fail when the file isn't local). We match
        # on hostname AND on lower-cased hostname-derived node_id, since
        # node_agent munges hostname to a slug.
        self_hostname = _socket.gethostname()
        self_hostname_l = self_hostname.lower()
        candidates = []
        now = _time.time()
        for nid, node_info in nodes.items():
            if not isinstance(node_info, dict):
                continue
            ssh = node_info.get("ssh")
            if not ssh:
                continue
            # Skip self
            if (node_info.get("hostname", "").lower() == self_hostname_l
                or self_hostname_l.startswith(nid.split("-")[0].lower())):
                continue
            hb_age = now - float(node_info.get("last_heartbeat", 0))
            candidates.append((hb_age, nid, node_info))
        # Sort by heartbeat age ascending (freshest first)
        candidates.sort(key=lambda c: c[0])
        # Skip nodes that haven't heartbeat-ed in the last 5 minutes —
        # they're considered offline and rsync will likely time out.
        FRESH_S = 300
        fresh = [c for c in candidates if c[0] <= FRESH_S]
        chosen = fresh[0] if fresh else (candidates[0] if candidates else None)
        if not chosen:
            return None
        _, nid, source_node = chosen
        source_node = dict(source_node)
        source_node["node_id"] = nid

        if not source_node.get("ssh"):
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

            # NaN-quarantine: if the saved teacher contains NaN weights,
            # refuse to use it. Loading a NaN teacher would poison the
            # student's distillation loss with NaN immediately.
            sd = ckpt.get("model", {})
            has_nan = any(
                torch.isnan(v).any().item() if v.is_floating_point() else False
                for v in sd.values()
            )
            if has_nan:
                print(f"  TEACHER QUARANTINE: {local_path} contains NaN "
                      f"weights (reported acc={accuracy:.0%} is stale). "
                      f"Falling back to no-distillation training.", flush=True)
                return None

            model = ProgressiveModel(
                d_model=config.get("d_model", 64),
                d_state=config.get("d_state", 16),
                expand=2,
                headdim=config.get("headdim", 16),
            ).to(device)
            for _ in range(config.get("n_kernel_layers", 1)):
                model.add_kernel_layer()
            model.load_state_dict(sd)
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
