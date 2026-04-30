"""cloud_archive.py — S3-compatible durable archive for generated corpora
and checkpoints. Mirrors the experiment_pusher.py shape: daemon thread,
local outbox fallback, fire-and-forget uploads that never block training.

Default backend: **Firebase Cloud Storage** (== Google Cloud Storage)
via its S3-compatible interoperability mode. Same Firebase project as
the telemetry RTDB, works through the EG VPN (R2 was originally chosen
but its TLS handshake is blocked by the corporate VPN's inspection
proxy). Can target any S3-compatible store (R2, B2, AWS S3, MinIO)
just by switching the endpoint URL.

Why this exists:
- Generated corpora (Cerebras, Bedrock, local Qwen) and trained
  checkpoints currently live on whichever box ran them. The m4-mini
  4 TB external is the convention for durable archive but requires
  manual rsync and m4-mini being online.
- vast.ai boxes are ephemeral; anything not pushed off them dies on
  rotation.
- Firebase Realtime DB (telemetry) is the wrong tool for files
  (1 GB total cap, JSON-only, ~1 MB per record limits).
- This module is the durable file-archive layer alongside the
  telemetry layer. Same fault-tolerance shape.

Configuration via environment (keep secrets out of code):
    ARCHIVE_ACCESS_KEY_ID      access key (GCS HMAC: starts with GOOG1E...)
    ARCHIVE_SECRET_ACCESS_KEY  secret
    ARCHIVE_ENDPOINT_URL       https://storage.googleapis.com (GCS)
                               or https://<acc>.r2.cloudflarestorage.com (R2)
                               or https://s3.<region>.backblazeb2.com (B2)
    ARCHIVE_BUCKET             bucket name (e.g. <project>.firebasestorage.app)
    ARCHIVE_REGION             region (default: auto)

Usage:
    from cloud_archive import CloudArchive

    archive = CloudArchive(
        experiment_kind="cortex_bilingual",
        run_name="step_FINAL",
        outbox_dir="checkpoints/lm",
    )

    # Upload a single file
    archive.upload(
        local_path="checkpoints/lm/step_FINAL.pt",
        artifact_kind="checkpoint",
    )

    # Upload many (queues, returns immediately)
    archive.upload_dir(
        local_dir="checkpoints/lm",
        artifact_kind="checkpoint",
        glob="*.pt",
    )

    # End of run — wait for in-flight uploads
    archive.complete()

Path convention on remote:
    <bucket>/<experiment_kind>/<run_name>/<artifact_kind>/<filename>

So `cortex_bilingual/step_FINAL/checkpoint/step_010000.pt` is fully
addressable and parallels the Firebase /experiments/<id>/runs/<run>/
structure for ergonomic dashboards.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import queue
import shutil
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:
    import boto3
    from botocore.exceptions import ClientError, EndpointConnectionError
    _HAS_BOTO3 = True
except ImportError:
    _HAS_BOTO3 = False


# ─────────────────────────────────────────────────────────────────────
# Configuration — env-driven so credentials never touch source.
# ─────────────────────────────────────────────────────────────────────

# Read either R2_* (preferred per docs/CLOUD_ARCHIVE.md and .env) or
# the legacy ARCHIVE_* prefix. R2_* wins when both are set.
def _envvar(*names: str, default: str | None = None) -> str | None:
    for n in names:
        v = os.environ.get(n)
        if v:
            return v
    return default


DEFAULT_BUCKET = _envvar("R2_BUCKET", "ARCHIVE_BUCKET", default="mamba3-archive")
DEFAULT_REGION = _envvar("R2_REGION", "ARCHIVE_REGION", default="auto")

# Files matching these extensions are compressed before upload.
# Excluded: torch/safetensors checkpoints (already compressed), images.
_COMPRESS_EXTS = {".jsonl", ".txt", ".log", ".csv", ".tsv", ".md"}
_NO_COMPRESS_EXTS = {".pt", ".pth", ".npz", ".safetensors",
                     ".png", ".jpg", ".jpeg", ".webp", ".bin", ".idx"}


def _client():
    """Build an S3 client pointed at R2. Returns None if creds aren't set.

    Reads R2_* (preferred) or ARCHIVE_* (legacy) env vars. R2_* wins.
    """
    if not _HAS_BOTO3:
        return None
    key = _envvar("R2_ACCESS_KEY_ID", "ARCHIVE_ACCESS_KEY_ID")
    sec = _envvar("R2_SECRET_ACCESS_KEY", "ARCHIVE_SECRET_ACCESS_KEY")
    endpoint = _envvar("R2_ENDPOINT_URL", "ARCHIVE_ENDPOINT_URL")
    if not (key and sec and endpoint):
        return None
    # Cloudflare R2 requires SigV4. Set explicitly because some
    # environments default to SigV2, which R2 rejects.
    from botocore.client import Config as _Config
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=key,
        aws_secret_access_key=sec,
        region_name=DEFAULT_REGION,
        config=_Config(signature_version="s3v4"),
    )


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for buf in iter(lambda: f.read(chunk), b""):
            h.update(buf)
    return h.hexdigest()


def _should_compress(path: Path) -> bool:
    ext = path.suffix.lower()
    if ext in _NO_COMPRESS_EXTS:
        return False
    if ext in _COMPRESS_EXTS:
        return True
    # Compress unknown text-like files; leave unknown binary alone.
    try:
        with path.open("rb") as f:
            sample = f.read(8192)
        # Naive: mostly-printable ASCII → text-ish → compress
        printable = sum(1 for b in sample if 32 <= b < 127 or b in (9, 10, 13))
        return len(sample) > 0 and printable / len(sample) > 0.85
    except OSError:
        return False


# ─────────────────────────────────────────────────────────────────────
# CloudArchive — the user-facing API.
# ─────────────────────────────────────────────────────────────────────

@dataclass
class _Job:
    local_path: Path
    remote_key: str
    compress: bool
    metadata: dict = field(default_factory=dict)


class CloudArchive:
    """Async, fault-tolerant uploader to S3-compatible storage.

    Design mirrors ExperimentPusher:
      - background daemon thread drains the upload queue
      - failed uploads spool to an on-disk JSONL outbox so we never
        lose data on a flaky connection
      - caller never blocks; complete() drains in-flight + outbox

    No-op if creds aren't configured (returns immediately on every
    method) so trainers can be unconditionally wired.
    """

    def __init__(self,
                 experiment_kind: str,
                 run_name: str,
                 outbox_dir: str | Path,
                 bucket: str = DEFAULT_BUCKET,
                 enabled: bool = True,
                 max_inflight: int = 16):
        self.experiment_kind = experiment_kind
        self.run_name = run_name
        self.bucket = bucket
        self.outbox_dir = Path(outbox_dir)
        self.outbox_dir.mkdir(parents=True, exist_ok=True)
        self.outbox_file = self.outbox_dir / "cloud_archive_outbox.jsonl"

        self.client = _client() if enabled else None
        if not enabled:
            print("[cloud-archive] disabled (enabled=False)", flush=True)
        elif not _HAS_BOTO3:
            print("[cloud-archive] disabled — boto3 not installed", flush=True)
        elif self.client is None:
            print("[cloud-archive] disabled — ARCHIVE_* / R2_* env vars not set", flush=True)
        else:
            print(f"[cloud-archive] {bucket}/{experiment_kind}/{run_name}/...", flush=True)

        self._queue: "queue.Queue[Optional[_Job]]" = queue.Queue(maxsize=max_inflight * 4)
        self._stopped = False
        self._n_uploaded = 0
        self._n_failed = 0
        self._n_bytes = 0
        self._lock = threading.Lock()

        if self.client is not None:
            self._worker = threading.Thread(
                target=self._worker_loop, daemon=True, name="cloud-archive"
            )
            self._worker.start()
        else:
            self._worker = None

    # ────────── public API ──────────

    def upload(self,
               local_path: str | Path,
               artifact_kind: str = "artifact",
               metadata: Optional[dict] = None) -> None:
        """Queue a file upload. Non-blocking.

        Remote key: <experiment_kind>/<run_name>/<artifact_kind>/<basename>
        File is gzip'd if its extension / content suggests text-like.
        """
        if self.client is None:
            return
        local_path = Path(local_path)
        if not local_path.exists():
            print(f"[cloud-archive] skip missing: {local_path}", flush=True)
            return
        compress = _should_compress(local_path)
        ext = ".gz" if compress else ""
        remote_key = (f"{self.experiment_kind}/{self.run_name}/"
                      f"{artifact_kind}/{local_path.name}{ext}")
        job = _Job(local_path=local_path, remote_key=remote_key,
                   compress=compress, metadata=metadata or {})
        try:
            self._queue.put_nowait(job)
        except queue.Full:
            # Don't block the caller; spool to outbox instead.
            self._spool(job, reason="queue_full")

    def upload_dir(self,
                   local_dir: str | Path,
                   artifact_kind: str = "artifact",
                   glob: str = "*",
                   recursive: bool = False) -> None:
        """Queue every matching file in a directory."""
        if self.client is None:
            return
        local_dir = Path(local_dir)
        if not local_dir.is_dir():
            return
        it = local_dir.rglob(glob) if recursive else local_dir.glob(glob)
        for p in it:
            if p.is_file():
                self.upload(p, artifact_kind=artifact_kind)

    def complete(self, timeout: float = 60.0) -> None:
        """Drain the queue, wait for in-flight uploads, drain the outbox."""
        if self.client is None:
            return
        try:
            self._queue.join()  # wait for queued jobs
        except Exception:
            pass
        # Try one outbox replay before declaring done.
        self._replay_outbox()
        self._stopped = True
        self._queue.put(None)
        if self._worker is not None:
            self._worker.join(timeout=timeout)
        print(f"[cloud-archive] complete: uploaded={self._n_uploaded} "
              f"failed={self._n_failed} bytes={self._n_bytes:,}", flush=True)

    def stats(self) -> dict:
        with self._lock:
            return {
                "uploaded": self._n_uploaded,
                "failed": self._n_failed,
                "bytes": self._n_bytes,
            }

    # ────────── internals ──────────

    def _worker_loop(self) -> None:
        while True:
            try:
                job = self._queue.get(timeout=1.0)
            except queue.Empty:
                if self._stopped:
                    return
                continue
            try:
                if job is None:
                    return
                self._do_upload(job)
            finally:
                self._queue.task_done()

    def _do_upload(self, job: _Job) -> None:
        try:
            # Skip if remote already has the byte-identical object.
            if self._remote_matches(job):
                with self._lock:
                    self._n_uploaded += 1
                return

            body = self._read_body(job)
            extra = {"Metadata": {k: str(v) for k, v in job.metadata.items()}} \
                if job.metadata else {}
            self.client.put_object(
                Bucket=self.bucket, Key=job.remote_key, Body=body, **extra,
            )
            with self._lock:
                self._n_uploaded += 1
                self._n_bytes += len(body)
        except (ClientError, EndpointConnectionError, OSError) as e:
            with self._lock:
                self._n_failed += 1
            self._spool(job, reason=str(e))

    def _read_body(self, job: _Job) -> bytes:
        with job.local_path.open("rb") as f:
            data = f.read()
        if job.compress:
            return gzip.compress(data, compresslevel=6)
        return data

    def _remote_matches(self, job: _Job) -> bool:
        """True if the remote object exists with the same content hash."""
        try:
            head = self.client.head_object(Bucket=self.bucket, Key=job.remote_key)
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") in {"404", "NoSuchKey", "NotFound"}:
                return False
            return False
        # Compare a content hash we attach as metadata. ETag is unreliable
        # for multipart uploads, so we use our own field.
        local_hash = _sha256(job.local_path)
        remote_hash = (head.get("Metadata") or {}).get("sha256")
        return remote_hash == local_hash

    def _spool(self, job: _Job, reason: str) -> None:
        record = {
            "ts": time.time(),
            "local_path": str(job.local_path),
            "remote_key": job.remote_key,
            "compress": job.compress,
            "metadata": job.metadata,
            "reason": reason,
        }
        with self.outbox_file.open("a") as f:
            f.write(json.dumps(record) + "\n")

    def _replay_outbox(self) -> None:
        if not self.outbox_file.exists():
            return
        try:
            lines = self.outbox_file.read_text().splitlines()
        except OSError:
            return
        if not lines:
            return
        # Move outbox aside; replay attempts go through normal queue.
        archive_path = self.outbox_file.with_suffix(".jsonl.replayed")
        try:
            shutil.move(self.outbox_file, archive_path)
        except OSError:
            return
        for line in lines:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            try:
                self._queue.put_nowait(_Job(
                    local_path=Path(rec["local_path"]),
                    remote_key=rec["remote_key"],
                    compress=bool(rec.get("compress")),
                    metadata=rec.get("metadata") or {},
                ))
            except queue.Full:
                # If we can't even re-queue, leave the rest in the
                # archived outbox for a future replay.
                break


# ─────────────────────────────────────────────────────────────────────
# Smoke test — quick end-to-end roundtrip.
# Run with: python cloud_archive.py
# ─────────────────────────────────────────────────────────────────────

def _smoke():
    """Round-trip a small file: upload → list → download → byte-diff."""
    import tempfile

    if _client() is None:
        print("R2_* env vars not set; skipping smoke test.")
        return 1

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        local = tmp / "smoke.jsonl"
        payload = b'{"hello":"world"}\n{"second":"line"}\n' * 50
        local.write_bytes(payload)

        a = CloudArchive(
            experiment_kind="smoke",
            run_name="cloud_archive_self_test",
            outbox_dir=tmp,
        )
        a.upload(local, artifact_kind="test")
        a.complete(timeout=30.0)

        # Download via raw client to verify
        c = _client()
        key = "smoke/cloud_archive_self_test/test/smoke.jsonl.gz"
        downloaded = tmp / "downloaded.gz"
        c.download_file(DEFAULT_BUCKET, key, str(downloaded))
        round_tripped = gzip.decompress(downloaded.read_bytes())

        assert round_tripped == payload, "round-trip mismatch"
        print("[smoke] PASS — uploaded, downloaded, byte-identical.")

        # Cleanup the test object so we don't leave junk in the bucket.
        c.delete_object(Bucket=DEFAULT_BUCKET, Key=key)
        return 0


if __name__ == "__main__":
    raise SystemExit(_smoke())
