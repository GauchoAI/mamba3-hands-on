"""Async, layered checkpoint policy.

W3.0 of the JEPA-Cortex plan. Three independent cadences (light / heavy /
best), each writing through a single background thread so training never
blocks on disk I/O. Atomic writes via .tmp + os.replace mean a crash
mid-write never corrupts a checkpoint.

Key design properties:
  - light is for *forking and evaluating*; weights only.
  - heavy is for *resuming the exact same run*; +optimizer +RNG +cursor.
  - best is for *picking a winner*; tracked against any scalar metric.

Every payload includes config, git sha, step, and a metrics snapshot —
that's what makes a saved file a true forking launchpad rather than a
recovery artifact.
"""
from __future__ import annotations
import json
import os
import queue
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch


# ---------------------------------------------------------------------------
# Storage policy
# ---------------------------------------------------------------------------
def should_keep(step: int) -> bool:
    """Exponential-density retention.

    Keep all light ckpts in the first 2k steps (training is volatile early
    and we want fine-grained bisecting). After that, keep every Nth where
    N grows. For a 30k-step run this caps light retention at ~150.
    """
    if step < 2000:
        return True
    if step < 10000:
        return step % 200 == 0
    return step % 800 == 0


# ---------------------------------------------------------------------------
# Async writer
# ---------------------------------------------------------------------------
@dataclass
class CheckpointPayload:
    name: str
    state: dict[str, Any]


class AsyncCheckpointer:
    """Single background thread, atomic writes, manifest log.

    Use:
        ck = AsyncCheckpointer("checkpoints/jepa_cortex/gpu0")
        ck.submit_light(step, model, metrics, config, run_name)
        ck.submit_heavy(step, model, opt, rng, ...)
        ck.maybe_submit_best(step, model, metric_value, ...)
        ck.flush()  # at end-of-run

    The writer never blocks the trainer. If the queue fills (rare; I/O is
    much faster than 50-step cadence allows) we drop the oldest pending
    write rather than stall the GPU.
    """

    def __init__(self, root: str | Path, queue_size: int = 8):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.q: queue.Queue[CheckpointPayload | None] = queue.Queue(maxsize=queue_size)
        self.manifest = self.root / "MANIFEST.jsonl"
        self.best_metric: float | None = None
        self.best_lock = threading.Lock()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    # -- submission API -----------------------------------------------------
    def submit_light(self, step: int, model: torch.nn.Module,
                     metrics: dict, config: dict, run_name: str = "") -> None:
        if not should_keep(step):
            return
        state = {
            "step": step,
            "kind": "light",
            "run_name": run_name,
            "model": _cpu_state_dict(model),
            "config": config,
            "metrics": metrics,
            "git_sha": _git_sha(),
            "ts": time.time(),
        }
        name = f"light_step_{step:07d}.pt"
        self._enqueue(CheckpointPayload(name, state))

    def submit_heavy(self, step: int, model: torch.nn.Module,
                     optimizer: torch.optim.Optimizer,
                     rng: dict, dataloader_cursor: dict | None,
                     metrics: dict, config: dict, run_name: str = "") -> None:
        state = {
            "step": step,
            "kind": "heavy",
            "run_name": run_name,
            "model": _cpu_state_dict(model),
            "optimizer": optimizer.state_dict(),
            "rng": rng,
            "dataloader_cursor": dataloader_cursor,
            "config": config,
            "metrics": metrics,
            "git_sha": _git_sha(),
            "ts": time.time(),
        }
        name = f"heavy_step_{step:07d}.pt"
        self._enqueue(CheckpointPayload(name, state))
        # Always keep a "last_heavy.pt" pointer so resume is one-shot.
        self._enqueue(CheckpointPayload("last_heavy.pt", state))

    def maybe_submit_best(self, step: int, model: torch.nn.Module,
                          metric_value: float, metric_name: str,
                          metrics: dict, config: dict, run_name: str = "",
                          lower_is_better: bool = True) -> bool:
        """Only writes if metric_value improves over the running best.
        Returns True if a write was scheduled."""
        with self.best_lock:
            if self.best_metric is None:
                better = True
            elif lower_is_better:
                better = metric_value < self.best_metric
            else:
                better = metric_value > self.best_metric
            if not better:
                return False
            self.best_metric = metric_value
        state = {
            "step": step,
            "kind": "best",
            "run_name": run_name,
            "model": _cpu_state_dict(model),
            "config": config,
            "metrics": metrics,
            "metric_name": metric_name,
            "metric_value": metric_value,
            "git_sha": _git_sha(),
            "ts": time.time(),
        }
        self._enqueue(CheckpointPayload("best.pt", state))
        return True

    def flush(self, timeout: float = 30.0) -> None:
        """Wait for the queue to drain. Call before exit."""
        deadline = time.time() + timeout
        while not self.q.empty() and time.time() < deadline:
            time.sleep(0.05)

    # -- internal -----------------------------------------------------------
    def _enqueue(self, payload: CheckpointPayload) -> None:
        try:
            self.q.put_nowait(payload)
        except queue.Full:
            # Drop oldest to make room — we'd rather skip a checkpoint
            # than block training. Light/heavy both repeat at known cadence
            # so a missed write is recoverable from the next one.
            try:
                _ = self.q.get_nowait()
            except queue.Empty:
                pass
            try:
                self.q.put_nowait(payload)
            except queue.Full:
                pass  # give up on this one

    def _worker(self) -> None:
        while True:
            payload = self.q.get()
            if payload is None:
                return
            try:
                self._write_atomic(payload)
                self._append_manifest(payload)
            except Exception as e:                          # noqa: BLE001
                # Surface in stderr but don't kill the thread; one bad
                # write should not stop subsequent successful ones.
                print(f"[ckpt] write failed for {payload.name}: {e}",
                      flush=True)

    def _write_atomic(self, payload: CheckpointPayload) -> None:
        final = self.root / payload.name
        tmp = final.with_suffix(final.suffix + ".tmp")
        torch.save(payload.state, tmp)
        os.replace(tmp, final)

    def _append_manifest(self, payload: CheckpointPayload) -> None:
        # Manifest holds one JSON line per write — cheap to tail, cheap to
        # parse, append-only so eval_daemon can use stat().st_size as a
        # change detector without race conditions.
        entry = {
            "name": payload.name,
            "kind": payload.state.get("kind"),
            "step": payload.state.get("step"),
            "ts": payload.state.get("ts"),
            "metrics": payload.state.get("metrics", {}),
            "metric_name": payload.state.get("metric_name"),
            "metric_value": payload.state.get("metric_value"),
        }
        with self.manifest.open("a") as f:
            f.write(json.dumps(entry) + "\n")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _cpu_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Move state dict to CPU on the caller's thread.

    Done synchronously here (not in the worker) so the worker can write
    without GPU contention. Caller pays a small CPU copy cost; the GPU
    is freed immediately.
    """
    return {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:                                       # noqa: BLE001
        return "unknown"


def capture_rng() -> dict:
    """Snapshot all RNG states needed for an exact resume."""
    state: dict[str, Any] = {
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    try:
        import numpy as np
        state["numpy"] = np.random.get_state()
    except ImportError:
        pass
    import random
    state["python"] = random.getstate()
    return state


def restore_rng(state: dict) -> None:
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    if "numpy" in state:
        import numpy as np
        np.random.set_state(state["numpy"])
    if "python" in state:
        import random
        random.setstate(state["python"])
