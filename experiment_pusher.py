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

from kappa_schemas import SCHEMAS, schema_path

# ---------------------------------------------------------------------------
# Firebase RTDB endpoint — same DB the existing GA dashboard reads from
# ---------------------------------------------------------------------------
DEFAULT_FIREBASE_URL = (
    "https://signaling-dcfad-default-rtdb.europe-west1.firebasedatabase.app"
)

# Protocol versions. Bump the relevant one on any breaking change to that
# wire format. Readers should branch on these and warn (not fail) on a
# version newer than what they know about. History: v1 — initial.
KAPPA_MANIFEST_VERSION = 1
STREAM_META_VERSION = 1

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
        self.outbox_dir = Path(outbox_dir)

        self.outbox_path = self.outbox_dir / "firebase_outbox.jsonl"
        self.outbox_dir.mkdir(parents=True, exist_ok=True)

        # Kappa streams: per-stream local JSONL shards (canonical),
        # mirrored to RTDB for live observability + drained to HF Parquet
        # on pack via kappa_packer.py.
        self.streams_dir = self.outbox_dir / "streams"
        self.streams_dir.mkdir(parents=True, exist_ok=True)
        self._stream_meta: dict[str, dict[str, Any]] = {}

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
        self._q: queue.Queue = queue.Queue(maxsize=256)
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

        # Auto-pack: a single background thread, debounced via a non-blocking
        # lock so a burst of stream() writes doesn't fan out N packers.
        self._pack_lock = threading.Lock()

        # Drop a manifest so kappa_packer (run as a separate process,
        # possibly later) can find experiment_id / run_id / firebase URL
        # from any local run directory it gets pointed at.
        self._write_manifest()

        # Publish protocol schemas to RTDB so any consumer can resolve a
        # version int → field definitions without the producer's source.
        # Idempotent PUT; if the doc already matches, RTDB just rewrites.
        self._publish_schemas()

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

    # -- Kappa stream API --------------------------------------------------
    # Streams are append-only logs of small records. Each push:
    #   1. Appends a JSON line to <run_dir>/streams/<name>-<UTC-date>.jsonl
    #      (canonical, what the packer reads)
    #   2. POSTs to /streams/<exp>/<run>/<name>/<UTC-date>/<auto-id> in RTDB
    #      (live mirror, dashboardable, deleted on seal)
    #   3. Bumps in-memory counters; flushes meta to RTDB every 50 pushes.
    # When a shard hits a pack threshold (size / count / age), kappa_packer
    # converts the JSONL → Parquet on HF Bucket and calls seal_stream() to
    # drop the matching RTDB records and bump last_pack_*.

    def _stream_meta_path(self, name: str) -> str:
        return f"streams_meta/{self.experiment_id}/{self.run_id}/{name}"

    def _stream_records_path(self, name: str, *parts: str) -> str:
        return "/".join(["streams", self.experiment_id, self.run_id, name, *parts])

    def _publish_schemas(self, names: list[str] | None = None) -> None:
        """Idempotent PUT of versioned JSON Schema docs to RTDB at
        `/_schemas/<name>/v<version>`. Default: every schema in the
        registry. Cheap (a handful of small docs); RTDB rewrites are
        no-ops when the content matches.
        """
        for name in (names or list(SCHEMAS.keys())):
            entry = SCHEMAS.get(name)
            if not entry:
                continue
            self._enqueue(("PUT", schema_path(name, entry["v"]), entry["doc"]))

    def _write_manifest(self) -> None:
        """Write a small _kappa_manifest.json so kappa_packer can find
        experiment_id / run_id / firebase URL from any run directory it's
        pointed at, without needing the original ExperimentPusher object.
        """
        try:
            manifest = {
                "version": KAPPA_MANIFEST_VERSION,
                "experiment_id": self.experiment_id,
                "run_id": self.run_id,
                "kind": self.kind,
                "firebase_url": self.firebase_url,
                "hf_user": os.environ.get("HF_ARCHIVE_USER",
                                          "miguelemosreverte"),
                "hf_bucket": os.environ.get("HF_ARCHIVE_BUCKET", "GauchoAI"),
            }
            (self.outbox_dir / "_kappa_manifest.json").write_text(
                json.dumps(manifest, indent=2)
            )
        except OSError:
            pass

    def declare_stream(
        self,
        name: str,
        *,
        pack_threshold_bytes: int = 10 * 1024 * 1024,
        pack_threshold_records: int = 50_000,
        pack_threshold_hours: float = 24.0,
        hf_user: str | None = None,
        hf_bucket: str | None = None,
        hf_prefix: str | None = None,
    ) -> None:
        """Register a stream. Writes the meta node once. Idempotent.

        URL templates are stored in the meta so dashboards can build
        clickable links by substituting the filename only — neither
        records nor the live RTDB nodes carry full URLs.
        """
        user = hf_user or os.environ.get(
            "HF_ARCHIVE_USER", "miguelemosreverte"
        )
        bucket = hf_bucket or os.environ.get(
            "HF_ARCHIVE_BUCKET", "GauchoAI"
        )
        prefix = hf_prefix or f"{self.kind}/{self.run_id}/streams"
        now = time.time()
        meta = {
            "schema_version": STREAM_META_VERSION,
            "stream": name,
            "experiment_id": self.experiment_id,
            "run_id": self.run_id,
            "kind": self.kind,
            "hf_user": user,
            "hf_bucket": bucket,
            "prefix": prefix,
            "url_browse_template":
                f"https://huggingface.co/buckets/{user}/{bucket}/{prefix}/{{filename}}",
            "url_hfsync_template":
                f"hf://buckets/{user}/{bucket}/{prefix}/{{filename}}",
            "pack_threshold_bytes": pack_threshold_bytes,
            "pack_threshold_records": pack_threshold_records,
            "pack_threshold_hours": pack_threshold_hours,
            "current_size_bytes": 0,
            "current_record_count": 0,
            "pack_progress_pct": 0.0,
            "last_pack_at": None,
            "last_pack_filename": None,
            "shard_started_at": now,
        }
        self._stream_meta[name] = meta
        self._enqueue(("PUT", self._stream_meta_path(name), meta))

    def stream(self, name: str, record: dict[str, Any]) -> None:
        """Append a record to a stream. Non-blocking.

        Local JSONL is canonical; RTDB push() is the live mirror.
        Records >256 KB are rejected (silent loss is the worst possible
        bug). Counters drive the pack-progress meta the dashboard sees.
        """
        if not self.enabled:
            return
        if name not in self._stream_meta:
            self.declare_stream(name)

        # Date routing: honor caller-supplied `ts` so backfill (e.g. session
        # archive replaying historical records) lands in the original day's
        # shard, not today's. Live trainers don't pass ts; they get now().
        coerced = {k: _safe_float(v) for k, v in record.items()}
        record_ts = coerced.get("ts")
        if not isinstance(record_ts, (int, float)):
            record_ts = time.time()
        record_with_ts = {**coerced, "ts": record_ts}
        line = json.dumps(record_with_ts, default=str) + "\n"
        line_bytes = line.encode("utf-8")
        if len(line_bytes) > 256_000:
            raise ValueError(
                f"stream record too large: {len(line_bytes)} bytes "
                f"(>256 KB cap). Truncate fields or split records."
            )

        # 1. Canonical local JSONL append (UTC date sharded — based on the
        #    record's own timestamp).
        today = time.strftime("%Y-%m-%d", time.gmtime(record_ts))
        shard_path = self.streams_dir / f"{name}-{today}.jsonl"
        try:
            with shard_path.open("a") as f:
                f.write(line)
        except OSError as e:
            # Local write fails — log + continue; RTDB still gets the record.
            print(f"[pusher] stream {name} local append failed: {e}",
                  flush=True)

        # 2. Live mirror to RTDB — only if this record's date is the current
        #    UTC date. Historical writes (backfill: session archive replay,
        #    log import, etc.) skip RTDB entirely. Auto-pack will move them
        #    to Parquet on HF; RTDB never has to see them. By construction,
        #    RTDB only ever contains today's records.
        current_utc = time.strftime("%Y-%m-%d", time.gmtime())
        if today == current_utc:
            self._enqueue((
                "POST",
                self._stream_records_path(name, today),
                record_with_ts,
                None,  # don't trigger _maybe_rotate; we manage retention via seal
            ))

        # 3. Counter bookkeeping. Flush meta on every push so the
        # dashboard's pack_progress_pct is genuinely live. At the
        # worst-case threshold (50k records / pack), this is ~50k
        # ~200-byte PATCHes per cycle = ~10 MB transfer, well within
        # free-tier budget. Typical training runs (~1000 records)
        # generate negligible meta traffic. Earlier every-50 throttle
        # made dashboards lag by 50 records, which for short bursts
        # meant they showed 0 when records existed.
        m = self._stream_meta[name]
        m["current_record_count"] += 1
        m["current_size_bytes"] += len(line_bytes)
        # Only flush meta to RTDB for current-day writes (live trainers).
        # Historical writes get summarized in one PATCH per shard at
        # seal-time, saving N round-trips for a backfill of N records.
        if today == current_utc:
            self._flush_stream_meta(name)

        # 4. Auto-pack: any shard whose date is strictly older than the
        # current write's date is non-active and safe to pack. The next
        # stream() call after a UTC-day rollover (or after backfill steps
        # past a day boundary) is what triggers ripening. One background
        # thread per pusher; the lock prevents fan-out under bursts.
        self._maybe_kick_pack(today)

    def _maybe_kick_pack(self, current_date: str) -> None:
        """Spawn a background pack thread if any non-active jsonl shards
        exist (date < current_date AND no matching .parquet)."""
        if not self._pack_lock.acquire(blocking=False):
            return  # one already running
        try:
            candidates: list[Path] = []
            for p in self.streams_dir.glob("*.jsonl"):
                stem = p.stem
                # `<stream>-YYYY-MM-DD` — date is the last 10 chars after a dash.
                if len(stem) < 11 or stem[-11] != "-":
                    continue
                shard_date = stem[-10:]
                if shard_date >= current_date:
                    continue
                # Don't skip when .parquet exists — pack_one is merge-aware
                # and reads the carry-over rows before appending the new
                # ones. Skipping here would lose records on watcher re-runs.
                candidates.append(p)
            if not candidates:
                self._pack_lock.release()
                return
            t = threading.Thread(
                target=self._pack_worker, args=(candidates,), daemon=True,
            )
            t.start()
        except Exception:
            self._pack_lock.release()
            raise

    def _pack_worker(self, candidates: list[Path]) -> None:
        """Pack each candidate JSONL → Parquet, then seal RTDB.

        Drains the pusher queue first so any in-flight RTDB POST for these
        dates lands before we DELETE the date subtree. Failures are logged
        but never raised — the JSONL stays on disk and gets retried on the
        next ripening tick.
        """
        try:
            try:
                from kappa_packer import pack_one, seal_via_rtdb, find_manifest
            except ImportError as e:
                print(f"[pusher] auto-pack disabled: {e}", flush=True)
                return
            # Drain pending RTDB writes for these dates. Bounded wait so a
            # partitioned network doesn't stall the pack thread forever.
            deadline = time.time() + 60.0
            while self._q.unfinished_tasks > 0 and time.time() < deadline:
                time.sleep(0.2)
            manifest = find_manifest(self.outbox_dir)
            for jsonl in candidates:
                try:
                    report = pack_one(jsonl, delete_source=True)
                    if report.get("skipped"):
                        continue
                    parquet = jsonl.with_suffix(".parquet")
                    if manifest is not None:
                        seal_via_rtdb(manifest, jsonl, parquet)
                    print(f"[pusher] auto-packed {jsonl.name}: "
                          f"{report.get('n_records', 0):,} rows "
                          f"({report.get('bytes_before', 0):,} → "
                          f"{report.get('bytes_after', 0):,} bytes)",
                          flush=True)
                except Exception as e:                          # noqa: BLE001
                    print(f"[pusher] pack failed for {jsonl}: {e}",
                          flush=True)
        finally:
            self._pack_lock.release()

    def _flush_stream_meta(self, name: str) -> None:
        m = self._stream_meta.get(name)
        if not m:
            return
        age_h = (time.time() - m["shard_started_at"]) / 3600.0
        pct = max(
            m["current_size_bytes"] / max(1, m["pack_threshold_bytes"]),
            m["current_record_count"] / max(1, m["pack_threshold_records"]),
            age_h / max(1e-9, m["pack_threshold_hours"]),
        )
        m["pack_progress_pct"] = round(100.0 * pct, 2)
        update = {
            "current_size_bytes": m["current_size_bytes"],
            "current_record_count": m["current_record_count"],
            "pack_progress_pct": m["pack_progress_pct"],
        }
        self._enqueue(("PATCH", self._stream_meta_path(name), update))

    def seal_stream(self, name: str, shard_filename: str,
                    parquet_filename: str | None = None) -> None:
        """Called by kappa_packer after a successful pack + upload.

        Drops the sealed records (RTDB DELETE on the date subtree) and
        updates last_pack_* in the meta. Resets live counters.
        """
        # Filename like "metrics-2026-04-30.jsonl"; date is the suffix.
        base = shard_filename.removesuffix(".jsonl").removesuffix(".parquet")
        date = base.rsplit("-", 1)[-1] if "-" in base else "unknown"
        if parquet_filename is None:
            parquet_filename = base + ".parquet"

        # 1. Drop the sealed records from RTDB.
        self._enqueue(("DELETE", self._stream_records_path(name, date), None))

        # 2. Update meta — last_pack_* and reset counters.
        if name not in self._stream_meta:
            self.declare_stream(name)
        m = self._stream_meta[name]
        m["last_pack_at"] = time.time()
        m["last_pack_filename"] = parquet_filename
        m["current_size_bytes"] = 0
        m["current_record_count"] = 0
        m["pack_progress_pct"] = 0.0
        m["shard_started_at"] = time.time()
        self._enqueue(("PUT", self._stream_meta_path(name), m))

    # -- per-step API (cheap; throttled internally) -------------------------
    def metrics(self, step: int, **values: float) -> None:
        """Append a metrics row. Routes through stream("metrics", ...)
        so the local JSONL + meta tracking happen automatically.
        Existing callers don't need code changes."""
        self.stream("metrics", {"step": step,
                                **{k: _safe_float(v) for k, v in values.items()}})

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
        """Append a canary sample. Routes through stream("samples", ...)."""
        self.stream("samples", {"step": step,
                                "prompt": prompt[:max_chars],
                                "completion": completion[:max_chars]})

    def event(self, type: str, step: int, details: str = "",
              reasoning: str = "", **extra) -> None:
        """Append a milestone / decision event. Routes through
        stream("events", ...)."""
        self.stream("events", {"type": type, "step": step,
                               "details": details, "reasoning": reasoning,
                               **extra})

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
        # 5 min deadline is generous: covers bursty end-of-run flushes
        # (e.g. 1000 unposted stream records each needing a HTTPS round
        # trip) without hanging forever on a partitioned network.
        deadline = time.time() + 300.0
        while True:
            with self._q.all_tasks_done:
                if self._q.unfinished_tasks == 0:
                    break
            if time.time() >= deadline:
                # Give up; surface the count so it's visible in logs.
                print(f"[pusher] complete() drain timeout — "
                      f"{self._q.unfinished_tasks} task(s) still in flight",
                      flush=True)
                break
            time.sleep(0.1)

        # Final pack: everything left in streams_dir (including today's
        # active shard, which is no longer being written to). Caller
        # contract is "no stream() calls after complete()", so safe.
        # Use a sentinel date in the future so all real dates qualify.
        self._maybe_kick_pack("9999-99-99")
        # Wait for the pack thread to finish (at most ~60s drain + pack).
        pack_deadline = time.time() + 180.0
        while self._pack_lock.locked() and time.time() < pack_deadline:
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
        if method == "DELETE":
            # Used by seal_stream() to drop a sealed date's records once
            # they've been packed into Parquet on HF.
            return _delete(url)
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
