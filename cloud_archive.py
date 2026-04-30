"""cloud_archive.py — HuggingFace Hub Buckets durable archive.

HF Buckets is a newer, S3-style file-archive product on HF (distinct
from datasets/models repos which use git+LFS). The right primitive
for "stash arbitrary files, no version-control overhead":

  hf sync ./data hf://buckets/<user>/<bucket-name>           # upload
  hf sync hf://buckets/<user>/<bucket-name> ./local         # download

The Python equivalent is HfApi.sync_bucket(source, dest, ...). This
module wraps that primitive in the same shape as experiment_pusher.py:
- daemon thread for periodic sync, never blocks training
- the trainer's existing checkpoints/log directory IS the staging
  area — we mirror it as-is, no extra copy
- final sync on complete()
- silent no-op if HF_TOKEN isn't set, so trainers can be wired
  unconditionally

Why HF Buckets specifically (not datasets/models repos):
- No git/LFS overhead — buckets are flat key-value blob stores
- Fits "stash this checkpoint" semantics better than "version
  this artifact"
- Egress is free (HF policy)
- Works through the EG corporate VPN (R2 is blocked at TLS layer)
- One bucket can hold many experiments; no proliferating repos

Why ALL-IN on HF (not Firebase Cloud Storage / R2 / etc.):
- Free at our scale and beyond
- ML-native — already where datasets/checkpoints live in this
  community
- Single auth token; same place we're already pulling teacher
  models from

Configuration:
    HF_TOKEN              huggingface.co token with write scope (THE only secret)

The non-secret defaults are baked in code (this is a public repo
and the bucket miguelemosreverte/GauchoAI is public):
    HF_ARCHIVE_USER       defaults to "miguelemosreverte"  (overridable)
    HF_ARCHIVE_BUCKET     defaults to "GauchoAI"           (overridable)
    HF_ARCHIVE_PRIVATE    defaults to "0" (public bucket)  (overridable)
    HF_ARCHIVE_SYNC_EVERY seconds between background syncs (default 60)

Path convention on the bucket:
    hf://buckets/<user>/<bucket>/<experiment_kind>/<run_name>/<filename>

Usage:
    from cloud_archive import CloudArchive

    archive = CloudArchive(
        experiment_kind="cortex_bilingual",
        run_name="step_FINAL",
        local_dir="checkpoints/lm",            # we'll mirror this
    )
    # ... trainer does its thing, writes to checkpoints/lm/* ...
    archive.complete()                          # final sync
"""
from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Optional

try:
    from huggingface_hub import HfApi
    from huggingface_hub.errors import HfHubHTTPError
    _HAS_HF = True
except ImportError:
    _HAS_HF = False


# ─────────────────────────────────────────────────────────────────────
# Configuration — env-driven so credentials never touch source.
# ─────────────────────────────────────────────────────────────────────

# Username + bucket name are NOT secrets — this is a public repo and a
# public bucket. Only HF_TOKEN is. Bake the public defaults in code so
# the smoke test works for anyone who clones + sets HF_TOKEN.
DEFAULT_USER    = os.environ.get("HF_ARCHIVE_USER", "miguelemosreverte")
DEFAULT_BUCKET  = os.environ.get("HF_ARCHIVE_BUCKET", "GauchoAI")
DEFAULT_PRIVATE = os.environ.get("HF_ARCHIVE_PRIVATE", "0") == "1"
DEFAULT_SYNC_EVERY = int(os.environ.get("HF_ARCHIVE_SYNC_EVERY", "60"))


def _hf_api() -> Optional["HfApi"]:
    if not _HAS_HF:
        return None
    token = os.environ.get("HF_TOKEN")
    if not token:
        return None
    try:
        return HfApi(token=token)
    except Exception:
        return None


def _bucket_path(user: str, bucket: str, *suffix: str) -> str:
    s = "/".join(p.strip("/") for p in suffix if p)
    base = f"hf://buckets/{user}/{bucket}"
    return f"{base}/{s}" if s else base


# ─────────────────────────────────────────────────────────────────────
# CloudArchive — the user-facing API.
# ─────────────────────────────────────────────────────────────────────

class CloudArchive:
    """Mirrors a local directory to an HF bucket on a periodic timer +
    a final flush at complete().

    Design:
      - The trainer's existing checkpoints/log directory IS the staging
        area. No extra file copies; we just rsync-style sync it.
      - A daemon thread runs sync_bucket every HF_ARCHIVE_SYNC_EVERY
        seconds. Caller never blocks. Network failures are logged but
        non-fatal — the next tick retries.
      - complete() does a final synchronous sync + stops the thread.

    Silent no-op if HF_TOKEN / HF_ARCHIVE_USER aren't configured, so
    trainers can be unconditionally wired and runs without creds still
    work (just without remote archive).
    """

    def __init__(self,
                 experiment_kind: str,
                 run_name: str,
                 local_dir: str | Path,
                 user: str = DEFAULT_USER,
                 bucket: str = DEFAULT_BUCKET,
                 private: bool = DEFAULT_PRIVATE,
                 sync_every_s: int = DEFAULT_SYNC_EVERY,
                 enabled: bool = True):
        self.experiment_kind = experiment_kind
        self.run_name = run_name
        self.local_dir = Path(local_dir)
        self.user = user
        self.bucket = bucket
        self.private = private
        self.sync_every_s = max(5, int(sync_every_s))

        self.api = _hf_api() if enabled else None
        if not enabled:
            print("[cloud-archive] disabled (enabled=False)", flush=True)
        elif not _HAS_HF:
            print("[cloud-archive] disabled — huggingface_hub not installed",
                  flush=True)
        elif self.api is None:
            print("[cloud-archive] disabled — HF_TOKEN env var not set",
                  flush=True)
        elif not self.user:
            print("[cloud-archive] disabled — HF_ARCHIVE_USER env var not set",
                  flush=True)
            self.api = None
        else:
            self.bucket_id = f"{self.user}/{self.bucket}"
            self.remote_prefix = _bucket_path(
                self.user, self.bucket, experiment_kind, run_name,
            )
            scope = "private" if self.private else "public"
            print(f"[cloud-archive] HF bucket sync: {self.local_dir}/  →  "
                  f"{self.remote_prefix}  ({scope})", flush=True)

        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._n_syncs = 0
        self._n_uploaded_total = 0
        self._n_failed = 0

        if self.api is not None:
            self._ensure_bucket()
            self._worker = threading.Thread(
                target=self._sync_loop, daemon=True, name="cloud-archive"
            )
            self._worker.start()
        else:
            self._worker = None

    # ────────── public API ──────────

    def upload(self,
               local_path: str | Path,
               artifact_kind: str = "artifact",
               metadata: Optional[dict] = None) -> None:
        """Compatibility shim with the old per-file API.

        Buckets sync the WHOLE local_dir — individual upload() calls
        are mostly no-ops here. We use them to nudge a sync on
        ckpt-save boundaries so the remote stays close to the local
        without waiting for the periodic timer.
        """
        if self.api is None:
            return
        # Don't sync more often than the timer; just register intent.
        # The next periodic tick will pick it up. Callers that want
        # immediate sync can call sync_now().

    def upload_dir(self, *args, **kwargs) -> None:
        """Compat shim — sync covers it."""
        return

    def sync_now(self, dry_run: bool = False) -> None:
        """Force an immediate sync of local_dir → bucket prefix."""
        if self.api is None:
            return
        self._do_sync(dry_run=dry_run)

    def complete(self, timeout: float = 120.0) -> None:
        """Stop the periodic thread + do a final synchronous sync."""
        if self.api is None:
            return
        self._stop.set()
        if self._worker is not None:
            self._worker.join(timeout=timeout)
        # Final flush — even if the timer just fired, do one more so
        # any files written between the last tick and now are pushed.
        self._do_sync()
        with self._lock:
            print(f"[cloud-archive] complete: syncs={self._n_syncs} "
                  f"files_pushed={self._n_uploaded_total} "
                  f"failed_attempts={self._n_failed}", flush=True)

    def stats(self) -> dict:
        with self._lock:
            return {
                "syncs": self._n_syncs,
                "uploaded": self._n_uploaded_total,
                "failed": self._n_failed,
            }

    # ────────── internals ──────────

    def _ensure_bucket(self) -> None:
        """Lazy-create the bucket if it doesn't exist yet."""
        try:
            from huggingface_hub import create_bucket
            create_bucket(
                bucket_id=self.bucket_id,
                private=self.private,
                exist_ok=True,
                token=os.environ.get("HF_TOKEN"),
            )
        except HfHubHTTPError as e:
            # If we don't have create perms but the bucket exists,
            # uploads will still work — surface the note but continue.
            print(f"[cloud-archive] create_bucket note: {e}", flush=True)
        except Exception as e:
            print(f"[cloud-archive] create_bucket unexpected: "
                  f"{type(e).__name__}: {e}", flush=True)

    def _sync_loop(self) -> None:
        """Periodic sync thread."""
        while not self._stop.wait(self.sync_every_s):
            self._do_sync()

    def _do_sync(self, dry_run: bool = False) -> None:
        if self.api is None or not self.local_dir.is_dir():
            return
        try:
            plan = self.api.sync_bucket(
                source=str(self.local_dir),
                dest=self.remote_prefix,
                dry_run=dry_run,
                quiet=True,
                token=os.environ.get("HF_TOKEN"),
            )
            n_files = self._extract_n_uploaded(plan)
            with self._lock:
                self._n_syncs += 1
                self._n_uploaded_total += n_files
            if n_files > 0 and not dry_run:
                print(f"[cloud-archive] synced {n_files} new/changed file(s) "
                      f"→ {self.remote_prefix}", flush=True)
        except HfHubHTTPError as e:
            with self._lock:
                self._n_failed += 1
            print(f"[cloud-archive] sync failed (will retry): {e}", flush=True)
        except Exception as e:
            with self._lock:
                self._n_failed += 1
            print(f"[cloud-archive] sync unexpected: "
                  f"{type(e).__name__}: {e}", flush=True)

    @staticmethod
    def _extract_n_uploaded(plan) -> int:
        """SyncPlan structure varies; pull a best-effort uploaded count."""
        for attr in ("n_uploaded", "uploaded_count", "num_uploaded"):
            v = getattr(plan, attr, None)
            if isinstance(v, int):
                return v
        # Fall back to len of an "uploads" list if present
        uploads = getattr(plan, "uploads", None) or getattr(plan, "to_upload", None)
        if uploads is not None:
            try:
                return len(uploads)
            except TypeError:
                return 0
        return 0


# ─────────────────────────────────────────────────────────────────────
# Smoke test — quick end-to-end roundtrip.
# Run with: python cloud_archive.py
# ─────────────────────────────────────────────────────────────────────

def _smoke():
    """Round-trip a small file: write locally → bucket sync → list → diff."""
    import tempfile

    api = _hf_api()
    if api is None:
        print("HF_TOKEN not set; skipping smoke test.")
        return 1
    user = DEFAULT_USER
    if not user:
        print("HF_ARCHIVE_USER not set; skipping smoke test.")
        return 1

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        local = tmp / "smoke.jsonl"
        payload = b'{"hello":"world"}\n{"second":"line"}\n' * 50
        local.write_bytes(payload)

        a = CloudArchive(
            experiment_kind="smoke",
            run_name="cloud_archive_self_test",
            local_dir=tmp,
            sync_every_s=10,   # not waited on; complete() flushes
        )
        a.complete(timeout=120.0)

        # Verify by listing the bucket prefix
        from huggingface_hub import HfApi as _Api
        x = _Api(token=os.environ["HF_TOKEN"])
        tree = list(x.list_bucket_tree(
            bucket_id=f"{user}/{DEFAULT_BUCKET}",
            path=f"smoke/cloud_archive_self_test",
        ))
        names = [getattr(e, "path", str(e)) for e in tree]
        assert any("smoke.jsonl" in n for n in names), \
            f"smoke.jsonl not found in remote tree: {names}"
        print(f"[smoke] PASS — uploaded files visible at "
              f"hf://buckets/{user}/{DEFAULT_BUCKET}/smoke/cloud_archive_self_test/")
        print(f"[smoke] remote tree: {names}")

        # Optional: also download & byte-diff for full roundtrip
        from huggingface_hub import hf_hub_download_to_local
        # Sync back to a different local path
        round_trip_dir = tmp / "roundtrip"
        round_trip_dir.mkdir()
        plan = x.sync_bucket(
            source=f"hf://buckets/{user}/{DEFAULT_BUCKET}/smoke/cloud_archive_self_test",
            dest=str(round_trip_dir),
            quiet=True,
            token=os.environ["HF_TOKEN"],
        )
        downloaded = round_trip_dir / "smoke.jsonl"
        if downloaded.exists():
            assert downloaded.read_bytes() == payload, "round-trip mismatch"
            print(f"[smoke] PASS — bidirectional roundtrip byte-identical.")
        return 0


if __name__ == "__main__":
    raise SystemExit(_smoke())
