"""kappa_packer.py — pack older append-only JSONL shards into Parquet.

Generalized from `experiments/10_jepa_structured/storage_packer.py`.
Same pattern: a directory contains time-sharded `*.jsonl` files (one
record per line, append-only). Files older than `--age-hours` get
read in, written as zstd-compressed Parquet, and the source JSONL
deleted. Idempotent — re-running with the same args is a no-op.

This is the canonical Kappa-architecture move:
    JSONL = the append-only log (cheap to write, schema-flexible)
    Parquet = the materialized columnar view (cheap to read, query)

Reader contract (transparent fallback): consumers should look for
`<base>.parquet` first, then `<base>.jsonl`. Either is the source of
truth depending on whether the shard has been packed yet.

Time-sharded naming convention:
    <stream>-<YYYY-MM-DD>.jsonl       active or recent shard
    <stream>-<YYYY-MM-DD>.parquet     packed shard (zstd)
where <stream> is "metrics" / "samples" / "events" / etc.

Usage:
    # Show what would be packed
    python kappa_packer.py --dir checkpoints/lm/streams --dry-run

    # Pack everything older than 24 hours
    python kappa_packer.py --dir checkpoints/lm/streams --age-hours 24

    # Pack everything regardless of age (e.g. final flush at end of run)
    python kappa_packer.py --dir checkpoints/lm/streams --age-hours 0

    # Recursive — walk subdirectories
    python kappa_packer.py --dir checkpoints/ --recursive

After packing, CloudArchive's next sync will push the Parquet shards
to the HF bucket + local mirror. The deleted JSONL won't reappear on
the remote because sync_bucket compares names: parquet-only on remote
is the desired end state.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path


def _pa():
    """Lazy pyarrow import so the rest of the module works when not packing."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    return pa, pq


# ─────────────────────────────────────────────────────────────────────
# Manifest discovery + RTDB seal — direct REST so the packer doesn't
# need an in-process ExperimentPusher (it's a separate command, may
# run between training sessions).
# ─────────────────────────────────────────────────────────────────────

def find_manifest(start: Path) -> Path | None:
    """Walk upward from `start` looking for `_kappa_manifest.json`.

    The pusher writes this on init at the run's outbox dir
    (typically the trainer's ckpt_dir), so packing any file inside
    that tree finds it. Returns None if not found.
    """
    p = start.resolve()
    if p.is_file():
        p = p.parent
    while True:
        candidate = p / "_kappa_manifest.json"
        if candidate.exists():
            return candidate
        if p.parent == p:
            return None
        p = p.parent


_SHARD_RE = re.compile(r"^(?P<stream>.+)-(?P<date>\d{4}-\d{2}-\d{2})$")


def _stream_name_from_filename(path: Path) -> tuple[str, str]:
    """Extract `(stream, date)` from a `<stream>-YYYY-MM-DD.jsonl` shard.
    The date is YYYY-MM-DD which has its own dashes — use a regex
    rather than naive rpartition.
    """
    base = path.stem
    m = _SHARD_RE.match(base)
    if not m:
        return base, "unknown"
    return m.group("stream"), m.group("date")


def seal_via_rtdb(manifest_path: Path, jsonl_path: Path,
                  parquet_path: Path) -> dict:
    """Tell RTDB the JSONL shard is now packed:
      - DELETE /streams/<exp>/<run>/<stream>/<date>/   (drop sealed records)
      - PUT    /streams_meta/<exp>/<run>/<stream>     (last_pack_*, reset counters)

    Direct REST via urllib so we don't need to import the pusher class.
    Returns a small status dict; safe to ignore on failure (kappa_packer
    will still report the local pack success).
    """
    import json as _json
    import time as _time
    import urllib.request as _req
    import urllib.error as _err

    try:
        manifest = _json.loads(manifest_path.read_text())
    except (OSError, _json.JSONDecodeError) as e:
        return {"ok": False, "reason": f"manifest read: {e}"}

    fb = manifest.get("firebase_url", "").rstrip("/")
    exp = manifest.get("experiment_id")
    run = manifest.get("run_id")
    if not (fb and exp and run):
        return {"ok": False, "reason": "manifest incomplete"}

    stream, date = _stream_name_from_filename(jsonl_path)
    parquet_filename = parquet_path.name

    # 1. DELETE the date subtree of records.
    rec_url = f"{fb}/streams/{exp}/{run}/{stream}/{date}.json"
    delete_ok = False
    try:
        req = _req.Request(rec_url, method="DELETE")
        with _req.urlopen(req, timeout=10.0) as resp:
            delete_ok = 200 <= resp.status < 300
    except _err.URLError as e:
        return {"ok": False, "reason": f"DELETE: {e}"}

    # 2. PATCH the meta with last_pack_* + counter reset.
    meta_url = f"{fb}/streams_meta/{exp}/{run}/{stream}.json"
    update = {
        "last_pack_at": _time.time(),
        "last_pack_filename": parquet_filename,
        "current_size_bytes": 0,
        "current_record_count": 0,
        "pack_progress_pct": 0.0,
        "shard_started_at": _time.time(),
    }
    patch_ok = False
    try:
        req = _req.Request(
            meta_url, data=_json.dumps(update).encode("utf-8"),
            method="PATCH",
            headers={"Content-Type": "application/json"},
        )
        with _req.urlopen(req, timeout=10.0) as resp:
            patch_ok = 200 <= resp.status < 300
    except _err.URLError as e:
        return {"ok": False, "reason": f"PATCH: {e}"}

    return {"ok": delete_ok and patch_ok,
            "delete_ok": delete_ok, "patch_ok": patch_ok,
            "stream": stream, "date": date,
            "parquet_filename": parquet_filename}


def read_records(path: Path) -> list[dict]:
    """Read a stream record-by-record from .parquet (preferred) or .jsonl."""
    parquet = path.with_suffix(".parquet")
    if parquet.exists():
        _, pq = _pa()
        return pq.read_table(parquet).to_pylist()
    if path.exists() and path.suffix == ".jsonl":
        out = []
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return out
    return []


def pack_one(jsonl_path: Path, *, delete_source: bool = True) -> dict:
    """JSONL → zstd Parquet. Returns a small report dict.

    Merge-aware: if a `.parquet` already exists for this shard (e.g. an
    earlier pass packed pass-1 records, then more records landed in a
    fresh JSONL on the same date), we read the existing rows first and
    accumulate. Without this, the watcher pattern (poll → append → pack)
    would lose every-other-pass's records.
    """
    if not jsonl_path.exists() or jsonl_path.suffix != ".jsonl":
        return {"skipped": "not a jsonl file", "path": str(jsonl_path)}
    parquet_path = jsonl_path.with_suffix(".parquet")

    pa, pq = _pa()
    records = []
    n_carry = 0
    if parquet_path.exists():
        try:
            records.extend(pq.read_table(parquet_path).to_pylist())
            n_carry = len(records)
        except Exception:                                       # noqa: BLE001
            # Unreadable existing parquet — overwrite, surface in report.
            n_carry = -1

    with jsonl_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    n_new = len(records) - max(0, n_carry)
    if n_new == 0:
        return {"skipped": "empty (no new records)",
                "path": str(jsonl_path), "n_carry": n_carry}

    # Coerce nested dict / list fields to JSON strings so the schema is
    # uniform across rows (the storage_packer pattern).
    for r in records:
        for k, v in list(r.items()):
            if isinstance(v, (dict, list)) and k != "_meta":
                # leave _meta as-is (already string-coerced by old code if
                # generated by storage_packer); for arbitrary nested fields
                # stringify only if heterogeneous types appear in column.
                pass
        if "_meta" in r and not isinstance(r["_meta"], str):
            r["_meta"] = json.dumps(r["_meta"])

    table = pa.Table.from_pylist(records)
    tmp = parquet_path.with_suffix(".parquet.tmp")
    pq.write_table(table, tmp, compression="zstd")
    tmp.replace(parquet_path)

    bytes_before = jsonl_path.stat().st_size
    bytes_after = parquet_path.stat().st_size
    if delete_source:
        jsonl_path.unlink()

    return {
        "path": str(jsonl_path),
        "n_records": len(records),
        "n_carry": max(0, n_carry),
        "n_new": n_new,
        "bytes_before": bytes_before,
        "bytes_after": bytes_after,
        "ratio": round(bytes_after / max(1, bytes_before), 3),
    }


def find_jsonl_shards(root: Path, recursive: bool = False) -> list[Path]:
    if recursive:
        return sorted(p for p in root.rglob("*.jsonl") if p.is_file())
    return sorted(p for p in root.glob("*.jsonl") if p.is_file())


def is_old_enough(path: Path, age_hours: float) -> bool:
    if age_hours <= 0:
        return True
    age_s = time.time() - path.stat().st_mtime
    return age_s >= age_hours * 3600.0


def cmd_pack(args: argparse.Namespace) -> None:
    root = Path(args.dir)
    if not root.is_dir():
        print(f"not a directory: {root}", file=sys.stderr)
        sys.exit(2)

    shards = find_jsonl_shards(root, recursive=args.recursive)
    eligible = [p for p in shards if is_old_enough(p, args.age_hours)]

    if args.dry_run:
        print(f"dry-run: {len(eligible)}/{len(shards)} JSONL shards would be packed")
        for p in eligible:
            mtime_age_h = (time.time() - p.stat().st_mtime) / 3600
            print(f"  {p}  ({p.stat().st_size:,} bytes, age={mtime_age_h:.1f}h)")
        return

    if not eligible:
        print(f"nothing to pack ({len(shards)} JSONL files in {root}, "
              f"none older than {args.age_hours}h)")
        return

    total_before = total_after = 0
    print(f"packing {len(eligible)} JSONL shard(s) in {root}:")
    for p in eligible:
        r = pack_one(p, delete_source=not args.keep_jsonl)
        if r.get("skipped"):
            print(f"  - {p.name}: skipped ({r['skipped']})")
            continue
        print(f"  + {p.name}: {r['n_records']:,} rows, "
              f"{r['bytes_before']:,} → {r['bytes_after']:,} bytes "
              f"(ratio {r['ratio']:.3f})")
        total_before += r["bytes_before"]
        total_after += r["bytes_after"]

        # Auto-seal RTDB if a manifest is reachable and --no-seal isn't set.
        if not args.no_seal:
            manifest = find_manifest(p)
            if manifest is not None:
                parquet_path = p.with_suffix(".parquet")
                seal = seal_via_rtdb(manifest, p, parquet_path)
                if seal["ok"]:
                    print(f"      sealed RTDB: stream={seal['stream']} "
                          f"date={seal['date']}")
                else:
                    print(f"      seal skipped: {seal.get('reason')}")
            elif args.require_manifest:
                print(f"      ! no manifest found near {p}; skipping seal "
                      "(--require-manifest set)")

    if total_before:
        print(f"\ntotal: {total_before:,} → {total_after:,} bytes "
              f"({100 * total_after / total_before:.1f}%)")


def cmd_status(args: argparse.Namespace) -> None:
    root = Path(args.dir)
    if not root.is_dir():
        print(f"not a directory: {root}", file=sys.stderr)
        sys.exit(2)
    n_jsonl = sum(1 for p in find_jsonl_shards(root, args.recursive))
    n_parquet = len(list(
        root.rglob("*.parquet") if args.recursive else root.glob("*.parquet")
    ))
    j_bytes = sum(
        p.stat().st_size for p in find_jsonl_shards(root, args.recursive)
    )
    p_bytes = sum(
        p.stat().st_size
        for p in (root.rglob("*.parquet") if args.recursive else root.glob("*.parquet"))
    )
    print(f"directory:    {root}")
    print(f"  JSONL:      {n_jsonl} files, {j_bytes:,} bytes "
          f"({j_bytes / 1024 / 1024:.2f} MB)")
    print(f"  Parquet:    {n_parquet} files, {p_bytes:,} bytes "
          f"({p_bytes / 1024 / 1024:.2f} MB)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dir", default="data/streams",
                    help="directory containing JSONL shards")
    ap.add_argument("--age-hours", type=float, default=24.0,
                    help="only pack shards whose mtime is older than this "
                         "(default 24h; 0 = pack all)")
    ap.add_argument("--recursive", action="store_true",
                    help="walk subdirectories")
    ap.add_argument("--keep-jsonl", action="store_true",
                    help="don't delete the source JSONL after packing")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--status", action="store_true",
                    help="show JSONL/Parquet counts and exit")
    ap.add_argument("--no-seal", action="store_true",
                    help="skip RTDB seal even if a _kappa_manifest.json is found")
    ap.add_argument("--require-manifest", action="store_true",
                    help="warn loudly when no manifest is reachable")
    args = ap.parse_args()

    if args.status:
        cmd_status(args)
    else:
        cmd_pack(args)


if __name__ == "__main__":
    main()
