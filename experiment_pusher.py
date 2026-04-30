"""experiment_pusher.py — cross-experiment Firebase Realtime DB adapter.

Implements the schema documented in `docs/EXPERIMENT_FIREBASE_SCHEMA.md`.
Designed to be imported by any trainer in the repo (jepa/, rlf_cortex/,
three_populations.py, future experiments) so a single dashboard can
surface them all.

Design decisions baked in here:
  - Uses urllib only (no firebase-admin / no SDK dependency)
  - Throttled writes by channel: every 2 min for metrics, 1 h for samples,
    immediate for events + status. Caller can pump on every step; we
    only POST when the budget allows. See FREE TIER ACCOUNTING in the
    schema doc for the math.
  - Falls back to a local JSONL outbox on push failure so we never lose
    data on a flaky connection. A sibling `firebase_replay.py` (TODO)
    drains the outbox when the network returns.
  - Each push is fire-and-forget — caller doesn't block on the request.
    Failures are logged but never raised.

Existing /mamba3/* GA paths (firebase_push.py) are left alone. This module
writes under /experiments/<experiment_id>/runs/<run_id>/* — additive.

Usage:
    from experiment_pusher import ExperimentPusher

    p = ExperimentPusher(
        experiment_id="rlf-cortex-2026-04-30",
        run_id="gpu3-recurse-n3",
        kind="rlf_cortex",
        config=asdict(cfg),
        outbox_dir="runs/rlf_cortex/gpu3-recurse-n3",
    )
    p.declare_experiment(name="RLF-Cortex round 1",
                         hypothesis="Layer recursion + lifeline alone, no LoopRoPE / HaltingHead.")
    p.declare_run(purpose="n_loops=3, batch=32 (compute-doubled vs jepa baseline)",
                  gpu=3)

    # In the train loop, on every step:
    p.metrics(step=step, byte_ce=..., jepa_loss=..., intent_var=...)
    p.heartbeat(step=step, sps=sps, gpu_mem_mb=..., gpu_util_pct=...)
    if step % 100 == 0:
        p.canary_sample(step=step, prompt=..., completion=...)

    # On milestone:
    p.event(type="milestone", details="JEPA term firing", step=step)

    # At end:
    p.complete()
"""
from __future__ import annotations

import json
import os
import queue
import threading
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Firebase RTDB endpoint — same DB the existing GA dashboard reads from
# ---------------------------------------------------------------------------
DEFAULT_FIREBASE_URL = (
    "https://signaling-dcfad-default-rtdb.europe-west1.firebasedatabase.app"
)

# Throttle interval per channel, in seconds. Tuned to fit free-tier budget;
# see schema doc §1 for the math (~940 writes/day per run budget).
THROTTLE_METRICS_S  = 120.0   # write every 2 min even if caller pushes per-step
THROTTLE_SAMPLES_S  = 3600.0  # write every 1 h
THROTTLE_STATUS_S   = 600.0   # write every 10 min
# Events and milestones are immediate (no throttle).

# Rolling-window retention per channel — keep only the last N entries.
# Combined with rotation in the writer, the per-run footprint stays bounded.
KEEP_METRICS  = 720
KEEP_SAMPLES  = 24
KEEP_EVENTS   = 100


# ---------------------------------------------------------------------------
# HTTP plumbing
# ---------------------------------------------------------------------------
def _put(url: str, data: dict, timeout: float = 5.0) -> bool:
    body = json.dumps(data, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url, data=body, method="PUT",
        headers={
            "Content-Type": "application/json",
            "User-Agent": "experiment_pusher/1.0",  # avoid Cloudflare 1010
        },
    )
    try:
        urllib.request.urlopen(req, timeout=timeout)
        return True
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
        return False


def _post(url: str, data: dict, timeout: float = 5.0) -> str | None:
    """POST returns Firebase's auto-generated push key, or None on failure."""
    body = json.dumps(data, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url, data=body, method="POST",
        headers={
            "Content-Type": "application/json",
            "User-Agent": "experiment_pusher/1.0",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            resp = json.loads(r.read().decode("utf-8"))
            return resp.get("name")
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
        return None


def _patch(url: str, data: dict, timeout: float = 5.0) -> bool:
    """Real HTTP PATCH — Firebase RTDB merges fields rather than replacing
    the document. Used by complete() so the run-level meta keeps its
    started_at / config / git_sha when we add ended_at + final_state."""
    body = json.dumps(data, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url, data=body, method="PATCH",
        headers={
            "Content-Type": "application/json",
            "User-Agent": "experiment_pusher/1.0",
        },
    )
    try:
        urllib.request.urlopen(req, timeout=timeout)
        return True
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
        return False


def _delete(url: str, timeout: float = 5.0) -> bool:
    req = urllib.request.Request(url, method="DELETE",
                                 headers={"User-Agent": "experiment_pusher/1.0"})
    try:
        urllib.request.urlopen(req, timeout=timeout)
        return True
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
        return False


def _get_keys(url: str, timeout: float = 5.0) -> list[str]:
    """GET returns the dict of children under the path; we want its keys."""
    req = urllib.request.Request(url, method="GET",
                                 headers={"User-Agent": "experiment_pusher/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read().decode("utf-8")) or {}
            if isinstance(data, dict):
                return sorted(data.keys())
            return []
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
        return []


# ---------------------------------------------------------------------------
# Pusher
# ---------------------------------------------------------------------------
@dataclass
class _ChannelState:
    """Throttle bookkeeping per channel."""
    last_push_ts: float = 0.0
    n_writes: int = 0


class ExperimentPusher:
    """Single-run pusher. One instance per training process.

    Thread-safety: a single background thread drains a queue. Caller methods
    are non-blocking — they enqueue and return.
    """

    def __init__(self, experiment_id: str, run_id: str, kind: str,
                 config: dict[str, Any], outbox_dir: str | Path,
                 firebase_url: str = DEFAULT_FIREBASE_URL,
                 enabled: bool = True):
        self.experiment_id = experiment_id
        self.run_id = run_id
        self.kind = kind
        self.config = config
        self.firebase_url = firebase_url.rstrip("/")
        self.enabled = enabled

        self.outbox_path = Path(outbox_dir) / "firebase_outbox.jsonl"
        self.outbox_path.parent.mkdir(parents=True, exist_ok=True)

        self._channel_state: dict[str, _ChannelState] = {
            "metrics": _ChannelState(),
            "samples": _ChannelState(),
            "status":  _ChannelState(),
        }
        # Cap-and-rotate counters so we issue cheap DELETEs only every K writes.
        self._writes_since_rotate: dict[str, int] = {
            "metrics": 0, "samples": 0, "events": 0,
        }

        # Single background queue + thread. Caller methods enqueue and return.
        self._q: queue.Queue = queue.Queue(maxsize=64)
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    # -- path helpers -------------------------------------------------------
    def _exp_path(self, *parts: str) -> str:
        return "/".join(["experiments", self.experiment_id, *parts])

    def _run_path(self, *parts: str) -> str:
        return self._exp_path("runs", self.run_id, *parts)

    def _url(self, path: str) -> str:
        return f"{self.firebase_url}/{path}.json"

    # -- declaration (one-time) --------------------------------------------
    def declare_experiment(self, name: str, hypothesis: str = "",
                           extra: dict | None = None) -> None:
        """Create the experiment-level meta document. Idempotent."""
        meta = {
            "name": name,
            "experiment_id": self.experiment_id,
            "kind": self.kind,
            "started_at": time.time(),
            "ended_at": None,
            "hypothesis": hypothesis,
            "git_sha": _git_sha(),
            **(extra or {}),
        }
        self._enqueue(("PUT", self._exp_path("meta"), meta))

    def declare_run(self, purpose: str = "", gpu: int | None = None,
                    extra: dict | None = None) -> None:
        """Create the run-level meta document. Idempotent."""
        meta = {
            "run_id": self.run_id,
            "started_at": time.time(),
            "ended_at": None,
            "purpose": purpose,
            "gpu": gpu,
            "config": _redact(self.config),
            "git_sha": _git_sha(),
            **(extra or {}),
        }
        self._enqueue(("PUT", self._run_path("meta"), meta))

    # -- per-step API (cheap; throttled internally) -------------------------
    def metrics(self, step: int, **values: float) -> None:
        """Append a metrics row. Throttled to ~every 2 min."""
        st = self._channel_state["metrics"]
        now = time.time()
        if now - st.last_push_ts < THROTTLE_METRICS_S:
            return
        st.last_push_ts = now
        row = {"step": step, "ts": now,
               **{k: _safe_float(v) for k, v in values.items()}}
        self._enqueue(("POST", self._run_path("metrics"), row, "metrics"))

    def heartbeat(self, step: int, **values) -> None:
        """Overwrite the run's status document. Throttled to ~every 10 min."""
        st = self._channel_state["status"]
        now = time.time()
        if now - st.last_push_ts < THROTTLE_STATUS_S:
            return
        st.last_push_ts = now
        row = {"last_step": step, "last_heartbeat": now,
               "state": "running",
               **{k: _safe_float(v) for k, v in values.items()}}
        self._enqueue(("PUT", self._run_path("status"), row))

    def canary_sample(self, step: int, prompt: str, completion: str,
                      max_chars: int = 600) -> None:
        """Append a canary sample. Throttled to ~every 1 h. Truncates strings."""
        st = self._channel_state["samples"]
        now = time.time()
        if now - st.last_push_ts < THROTTLE_SAMPLES_S:
            return
        st.last_push_ts = now
        row = {"step": step, "ts": now,
               "prompt": prompt[:max_chars],
               "completion": completion[:max_chars]}
        self._enqueue(("POST", self._run_path("samples"), row, "samples"))

    def event(self, type: str, step: int, details: str = "",
              reasoning: str = "", **extra) -> None:
        """Append a milestone / decision event. NOT throttled."""
        row = {"type": type, "step": step, "ts": time.time(),
               "details": details, "reasoning": reasoning, **extra}
        self._enqueue(("POST", self._run_path("events"), row, "events"))

    # -- run lifecycle ------------------------------------------------------
    def complete(self, final_state: str = "completed",
                 final_metrics: dict | None = None) -> None:
        """Mark the run as ended; flush queue. Call before exit."""
        end_ts = time.time()
        self._enqueue(("PATCH", self._run_path("meta"),
                       {"ended_at": end_ts, "final_state": final_state}))
        if final_metrics:
            self._enqueue(("PUT", self._run_path("final_metrics"),
                           {**final_metrics, "ts": end_ts}))
        self._enqueue(("PUT", self._run_path("status"),
                       {"state": final_state, "last_heartbeat": end_ts}))
        # Drain. Wait for queue empty AND for any in-flight network call
        # to finish — the worker grabs an item *then* makes a HTTPS call,
        # so an empty queue alone does not mean all writes have landed.
        # task_done()/join() handles this exactly.
        deadline = time.time() + 30.0
        # Poll-style timeout — Queue.join() has no timeout, so spin.
        while True:
            with self._q.all_tasks_done:
                if self._q.unfinished_tasks == 0:
                    break
            if time.time() >= deadline:
                break
            time.sleep(0.1)

    # -- internal -----------------------------------------------------------
    def _enqueue(self, item: tuple) -> None:
        if not self.enabled:
            return
        try:
            self._q.put_nowait(item)
        except queue.Full:
            # Drop oldest pending item, accept new one. Better to lose one
            # write than block training.
            try: self._q.get_nowait()
            except queue.Empty: pass
            try: self._q.put_nowait(item)
            except queue.Full: pass

    def _worker(self) -> None:
        while True:
            item = self._q.get()
            try:
                if item is None:
                    return
                method, path, payload = item[0], item[1], item[2]
                channel = item[3] if len(item) > 3 else None
                ok = self._dispatch(method, path, payload)
                if not ok:
                    self._spool(method, path, payload)
                elif channel is not None:
                    self._maybe_rotate(channel, path)
            finally:
                # Mark done so complete()'s drain can verify the worker
                # actually finished this item, not just dequeued it.
                self._q.task_done()

    def _dispatch(self, method: str, path: str, payload: dict) -> bool:
        url = self._url(path)
        if method == "POST":
            return _post(url, payload) is not None
        if method == "PUT":
            return _put(url, payload)
        if method == "PATCH":
            # Real HTTP PATCH — Firebase RTDB merges fields rather than
            # replacing the document. complete() uses this to add
            # ended_at + final_state without clobbering started_at /
            # config / git_sha that declare_run() wrote.
            return _patch(url, payload)
        return False

    def _spool(self, method: str, path: str, payload: dict) -> None:
        """On push failure, append to the local outbox so a replayer can
        drain it later. The trainer keeps running."""
        try:
            with self.outbox_path.open("a") as f:
                f.write(json.dumps({"method": method, "path": path,
                                    "payload": payload, "ts": time.time()}) + "\n")
        except OSError:
            pass

    def _maybe_rotate(self, channel: str, parent_path: str) -> None:
        """Every ~16 writes per channel, fetch the children and DELETE
        anything past the keep-last-N window. Cheap GET + cheap DELETEs;
        the GET is ~one per 16 writes per channel so total bandwidth is fine.
        """
        n = self._writes_since_rotate.get(channel, 0) + 1
        self._writes_since_rotate[channel] = n
        if n % 16 != 0:
            return
        keep = {"metrics": KEEP_METRICS, "samples": KEEP_SAMPLES,
                "events": KEEP_EVENTS}.get(channel, 100)
        keys = _get_keys(self._url(parent_path))
        if len(keys) <= keep:
            return
        for old_key in keys[: len(keys) - keep]:
            _delete(self._url(f"{parent_path}/{old_key}"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _git_sha() -> str:
    try:
        import subprocess
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:                                       # noqa: BLE001
        return "unknown"


def _safe_float(v: Any) -> Any:
    """Coerce numpy / torch scalars to Python floats; pass through other types."""
    try:
        import math
        f = float(v)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    except (TypeError, ValueError):
        return v


def _redact(d: dict) -> dict:
    """Strip non-JSON-serializable values from the config dict before push."""
    out = {}
    for k, v in d.items():
        try:
            json.dumps({k: v})
            out[k] = v
        except (TypeError, ValueError):
            out[k] = repr(v)[:200]
    return out


# ---------------------------------------------------------------------------
# CLI smoke
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    """Smoke test: declare a tiny test experiment and push a few rows."""
    import os, time
    p = ExperimentPusher(
        experiment_id="smoke-" + str(int(time.time())),
        run_id="local-smoke",
        kind="smoke",
        config={"hello": "world"},
        outbox_dir="/tmp",
    )
    p.declare_experiment(name="Smoke Test", hypothesis="Pusher works.")
    p.declare_run(purpose="local connectivity check")
    # Force throttle expiry so the test actually writes.
    for ch in p._channel_state.values():
        ch.last_push_ts = 0.0
    p.metrics(step=0, byte_ce=2.5, jepa_loss=1.6, intent_var=0.1)
    p.heartbeat(step=0, sps=0.1, gpu_mem_mb=6700, gpu_util_pct=100)
    p.canary_sample(step=0, prompt="hello?", completion="world.")
    p.event(type="milestone", step=0, details="smoke ran",
            reasoning="just a connectivity test")
    p.complete()
    print(f"smoke complete; check {p.firebase_url}/experiments/{p.experiment_id}.json")
