"""stream_reader.py — transparent reader over Kappa-architecture streams.

Producers write records to two stores:
  - **Hot tier** Firebase RTDB at /streams/<exp>/<run>/<name>/<UTC-date>/
    (live mirror, dashboardable, deleted on seal)
  - **Cold tier** HF Bucket at <prefix>/<name>-<UTC-date>.parquet
    (sealed, immutable, zstd-compressed)

This module reads from both and merges them, so a consumer never has
to care which store a record currently lives in. Output is identical
in shape to what the producer pushed.

Usage:
    from stream_reader import read_stream, read_meta, list_streams

    # discovery — what streams exist project-wide?
    for exp_id, run_id, name in list_streams():
        print(exp_id, run_id, name)

    # a single stream's metadata
    meta = read_meta("cortex_bilingual-2026-04-30",
                     "mlx-bilingual-widerN-2026-04-30",
                     "metrics")
    print(meta["pack_progress_pct"], meta["last_pack_filename"])

    # iterate records from any time range, transparent over both stores
    for rec in read_stream(
        experiment_id="cortex_bilingual-2026-04-30",
        run_id="mlx-bilingual-widerN-2026-04-30",
        stream="metrics",
        since="2026-04-25", until="2026-05-05",
    ):
        print(rec["step"], rec["loss"])

`since` and `until` are inclusive UTC date strings (`YYYY-MM-DD`) or
`None` for unbounded.

Auth:
- For RTDB reads on a public DB: no auth needed.
- For HF Bucket reads on a public bucket: no auth needed.
- For private resources: HF_TOKEN env var (read-scope sufficient).

Caching:
Sealed Parquet shards are cached at ~/.cache/mamba3-archive/<bucket>/
(or $MAMBA3_ARCHIVE_CACHE if set). Idempotent — repeat reads of the
same shard hit the cache, not the network.
"""
from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Iterator

# RTDB endpoint — keep aligned with experiment_pusher.py default.
DEFAULT_FIREBASE_URL = (
    "https://signaling-dcfad-default-rtdb.europe-west1.firebasedatabase.app"
)

# Local Parquet cache. Each shard is downloaded once, then read locally.
DEFAULT_CACHE = Path(
    os.environ.get("MAMBA3_ARCHIVE_CACHE",
                   str(Path.home() / ".cache" / "mamba3-archive"))
)


# ─────────────────────────────────────────────────────────────────────
# RTDB helpers (urllib only; no Firebase SDK dependency)
# ─────────────────────────────────────────────────────────────────────

def _rtdb_get(path: str, *, fb_url: str = DEFAULT_FIREBASE_URL,
              shallow: bool = False, timeout: float = 10.0) -> Any:
    url = f"{fb_url.rstrip('/')}/{path}.json"
    if shallow:
        url += "?shallow=true"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise


def list_streams(fb_url: str = DEFAULT_FIREBASE_URL
                 ) -> Iterator[tuple[str, str, str]]:
    """Walk /streams_meta/ and yield (experiment_id, run_id, stream)
    tuples for every stream that has metadata.
    """
    exps = _rtdb_get("streams_meta", fb_url=fb_url, shallow=True) or {}
    for exp_id in sorted(exps):
        runs = _rtdb_get(f"streams_meta/{exp_id}", fb_url=fb_url,
                         shallow=True) or {}
        for run_id in sorted(runs):
            streams = _rtdb_get(f"streams_meta/{exp_id}/{run_id}",
                                fb_url=fb_url, shallow=True) or {}
            for s in sorted(streams):
                yield exp_id, run_id, s


def read_meta(experiment_id: str, run_id: str, stream: str,
              fb_url: str = DEFAULT_FIREBASE_URL) -> dict[str, Any] | None:
    """Get the meta node for one stream, or None if it doesn't exist."""
    return _rtdb_get(
        f"streams_meta/{experiment_id}/{run_id}/{stream}", fb_url=fb_url,
    )


def _live_dates(experiment_id: str, run_id: str, stream: str,
                fb_url: str = DEFAULT_FIREBASE_URL) -> list[str]:
    """List UTC-date keys currently present under
    /streams/<exp>/<run>/<stream>/. Empty list if no live records."""
    base = f"streams/{experiment_id}/{run_id}/{stream}"
    keys = _rtdb_get(base, fb_url=fb_url, shallow=True)
    if not keys:
        return []
    return sorted(keys)


def _live_records(experiment_id: str, run_id: str, stream: str,
                  date: str,
                  fb_url: str = DEFAULT_FIREBASE_URL) -> list[dict]:
    """Fetch records for a single open (un-packed) date from RTDB."""
    path = f"streams/{experiment_id}/{run_id}/{stream}/{date}"
    body = _rtdb_get(path, fb_url=fb_url)
    if not body:
        return []
    # body is {auto_id: record_dict, ...}; drop auto_ids, keep records.
    return list(body.values())


# ─────────────────────────────────────────────────────────────────────
# HF Bucket Parquet shards
# ─────────────────────────────────────────────────────────────────────

def _list_sealed_shards(meta: dict[str, Any]
                        ) -> list[tuple[str, str, str]]:
    """Return a list of (filename, parquet_path_in_bucket, cache_key)
    for every sealed Parquet shard for this stream.

    The cache_key encodes the remote object's size + last-modified so the
    reader can detect when a previously-cached parquet has been re-written
    on HF (which happens whenever pack_one runs in merge-aware mode and
    appends new rows). Form: `<size>:<last_modified_iso>` — a content
    fingerprint cheap enough to compute on every list.
    """
    try:
        from huggingface_hub import HfApi
    except ImportError:
        return []

    user = meta.get("hf_user")
    bucket = meta.get("hf_bucket")
    prefix = meta.get("prefix")
    stream = meta.get("stream")
    if not all([user, bucket, prefix, stream]):
        return []

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    bucket_id = f"{user}/{bucket}"
    try:
        tree = api.list_bucket_tree(bucket_id=bucket_id, prefix=prefix,
                                    recursive=True)
    except Exception:
        return []

    out = []
    suffix = ".parquet"
    name_prefix = f"{stream}-"
    for entry in tree:
        path = getattr(entry, "path", None) or str(entry)
        filename = Path(path).name
        if not (filename.startswith(name_prefix)
                and filename.endswith(suffix)):
            continue
        size = getattr(entry, "size", None)
        last_modified = getattr(entry, "last_modified", None)
        cache_key = f"{size}:{last_modified}"
        out.append((filename, path, cache_key))
    return sorted(out)


def _download_shard(meta: dict[str, Any], remote_path: str,
                    cache_key: str = "",
                    cache_root: Path = DEFAULT_CACHE) -> Path:
    """Download a Parquet shard to the local cache; return the local
    path. If a cached file exists AND its sidecar `.cachekey` matches
    the remote `cache_key` (size + last-modified), reuse it. Otherwise
    re-download. Empty `cache_key` = always re-download (defensive).
    """
    user = meta["hf_user"]
    bucket = meta["hf_bucket"]
    bucket_id = f"{user}/{bucket}"
    cache_dir = cache_root / user / bucket / Path(remote_path).parent
    cache_dir.mkdir(parents=True, exist_ok=True)
    local_path = cache_dir / Path(remote_path).name
    sidecar = local_path.with_suffix(local_path.suffix + ".cachekey")

    if local_path.exists() and cache_key:
        try:
            cached_key = sidecar.read_text().strip()
        except OSError:
            cached_key = ""
        if cached_key == cache_key:
            return local_path

    from huggingface_hub import HfApi
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    api.sync_bucket(
        source=f"hf://buckets/{bucket_id}/{remote_path}",
        dest=str(cache_dir),
        quiet=True,
        token=os.environ.get("HF_TOKEN"),
    )
    if not local_path.exists():
        # Fallback: the file may have landed under a nested path.
        candidates = list(cache_dir.rglob(Path(remote_path).name))
        if candidates:
            local_path = candidates[0]
        else:
            raise FileNotFoundError(
                f"download produced no file for {remote_path}"
            )

    # Persist the new cache key so the next read can short-circuit.
    if cache_key:
        try:
            sidecar.write_text(cache_key)
        except OSError:
            pass
    return local_path


def _read_parquet(path: Path) -> list[dict]:
    """Read a Parquet shard as list of dicts. pyarrow lazy import."""
    import pyarrow.parquet as pq
    return pq.read_table(path).to_pylist()


# ─────────────────────────────────────────────────────────────────────
# Date filtering
# ─────────────────────────────────────────────────────────────────────

def _date_in_range(date: str, since: str | None, until: str | None) -> bool:
    if since is not None and date < since:
        return False
    if until is not None and date > until:
        return False
    return True


def _date_from_shard(filename: str, stream: str) -> str:
    """Extract `2026-04-29` from `metrics-2026-04-29.parquet`."""
    base = filename
    if base.endswith(".parquet"):
        base = base[: -len(".parquet")]
    if base.endswith(".jsonl"):
        base = base[: -len(".jsonl")]
    prefix = f"{stream}-"
    if base.startswith(prefix):
        return base[len(prefix):]
    return base


# ─────────────────────────────────────────────────────────────────────
# The transparent reader
# ─────────────────────────────────────────────────────────────────────

def read_stream(experiment_id: str, run_id: str, stream: str,
                *,
                since: str | None = None, until: str | None = None,
                fb_url: str = DEFAULT_FIREBASE_URL,
                cache_root: Path = DEFAULT_CACHE,
                ) -> Iterator[dict[str, Any]]:
    """Yield records from a stream, merging sealed Parquet (HF) +
    open RTDB records. Records sorted by `ts` within each shard;
    shards iterated in chronological order. Caller doesn't know
    which store a record came from.

    Args:
        experiment_id: e.g. 'cortex_bilingual-2026-04-30'
        run_id:        e.g. 'mlx-bilingual-widerN-2026-04-30'
        stream:        e.g. 'metrics' / 'samples' / 'events'
        since:         inclusive lower-bound 'YYYY-MM-DD' (or None)
        until:         inclusive upper-bound 'YYYY-MM-DD' (or None)
    """
    meta = read_meta(experiment_id, run_id, stream, fb_url=fb_url)
    if meta is None:
        return  # stream doesn't exist; yield nothing

    sealed = _list_sealed_shards(meta)
    sealed_dates: set[str] = set()
    sealed_path_by_date: dict[str, str] = {}
    sealed_key_by_date: dict[str, str] = {}
    for filename, path, cache_key in sealed:
        d = _date_from_shard(filename, stream)
        sealed_dates.add(d)
        sealed_path_by_date[d] = path
        sealed_key_by_date[d] = cache_key

    live_dates = _live_dates(experiment_id, run_id, stream, fb_url=fb_url)

    # Union, ordered chronologically. A date can appear in BOTH stores
    # if a producer is mid-day on a date that ALSO has an intra-day
    # Parquet (a seal happened between two record bursts on the same
    # day). The two sets are disjoint by construction: seal_via_rtdb
    # issues DELETE on the date subtree before pack returns, so any
    # records present in RTDB after the seal are strictly new pushes.
    # We can therefore concatenate sealed + live for that date and
    # sort by ts without risk of double-counting.
    all_dates = sorted(sealed_dates | set(live_dates))
    for date in all_dates:
        if not _date_in_range(date, since, until):
            continue
        records: list[dict] = []
        if date in sealed_dates:
            shard_path = _download_shard(
                meta, sealed_path_by_date[date],
                cache_key=sealed_key_by_date[date],
                cache_root=cache_root,
            )
            records.extend(_read_parquet(shard_path))
        if date in live_dates:
            records.extend(_live_records(experiment_id, run_id, stream,
                                         date, fb_url=fb_url))
        # Sort each date's records by ts ascending. Records without ts
        # go last in their shard (rare; producer always sets ts).
        for r in sorted(records, key=lambda r: r.get("ts", float("inf"))):
            yield r


# ─────────────────────────────────────────────────────────────────────
# CLI — quick inspection
# ─────────────────────────────────────────────────────────────────────

def _cli() -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("ls", help="list every stream project-wide")

    p = sub.add_parser("meta", help="print one stream's meta as JSON")
    p.add_argument("experiment_id")
    p.add_argument("run_id")
    p.add_argument("stream")

    r = sub.add_parser("read", help="read records (merged across stores)")
    r.add_argument("experiment_id")
    r.add_argument("run_id")
    r.add_argument("stream")
    r.add_argument("--since", default=None)
    r.add_argument("--until", default=None)
    r.add_argument("--limit", type=int, default=10,
                   help="cap output rows (0 = unlimited)")
    r.add_argument("--fields", default=None,
                   help="comma-sep list of fields to print (default: all)")

    args = ap.parse_args()

    if args.cmd == "ls":
        for exp, run, s in list_streams():
            print(f"{exp}/{run}/{s}")
        return 0

    if args.cmd == "meta":
        m = read_meta(args.experiment_id, args.run_id, args.stream)
        print(json.dumps(m, indent=2, default=str))
        return 0

    if args.cmd == "read":
        fields = args.fields.split(",") if args.fields else None
        n = 0
        for r in read_stream(args.experiment_id, args.run_id, args.stream,
                             since=args.since, until=args.until):
            if fields:
                r = {k: r.get(k) for k in fields}
            print(json.dumps(r, default=str))
            n += 1
            if args.limit and n >= args.limit:
                break
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(_cli())
