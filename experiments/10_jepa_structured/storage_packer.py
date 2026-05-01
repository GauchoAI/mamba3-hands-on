"""Storage packer: pack older JSONL tiles into Parquet when disk pressure rises.

Trigger condition: total bytes used by data/ exceeds `threshold_bytes`. When
hit, the packer selects the oldest fully-completed tiles (by their
`last_generated_at` timestamp in the registry), reads each tile's JSONL,
writes it as a parquet file alongside, and removes the source JSONL.

Trainer reads transparently: load_tile_examples() tries .jsonl first, then
.parquet, returning a list of dicts.

Usage:
    python storage_packer.py status                     # show disk pressure
    python storage_packer.py pack --threshold-tb 3      # pack until under
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from curriculum import Curriculum, TileRegistry

EXPERIMENT_DIR = Path(__file__).resolve().parent


# pyarrow is optional — import lazily so the rest of the module works without it
def _pa():
    import pyarrow as pa
    import pyarrow.parquet as pq
    return pa, pq


TB = 1024 ** 4
GB = 1024 ** 3


def disk_usage_bytes(path: Path) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def load_tile_examples(out_dir: Path, tile_relpath: str) -> list[dict]:
    """Read a tile's examples, transparently from JSONL or Parquet."""
    jsonl = out_dir / tile_relpath
    parquet = jsonl.with_suffix(".parquet")
    if jsonl.exists():
        examples = []
        with open(jsonl) as f:
            for line in f:
                try:
                    examples.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return examples
    if parquet.exists():
        _, pq = _pa()
        table = pq.read_table(parquet)
        return table.to_pylist()
    return []


def pack_tile(out_dir: Path, tile_relpath: str) -> dict:
    """Pack one tile's JSONL → parquet. Returns size before/after, or skip reason."""
    jsonl = out_dir / tile_relpath
    if not jsonl.exists():
        return {"skipped": "no_jsonl"}
    parquet = jsonl.with_suffix(".parquet")
    pa, pq = _pa()
    examples = []
    with open(jsonl) as f:
        for line in f:
            try:
                examples.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if not examples:
        return {"skipped": "empty"}
    # Coerce _meta to a string column so schema is uniform across tiles.
    for ex in examples:
        if "_meta" in ex and not isinstance(ex["_meta"], str):
            ex["_meta"] = json.dumps(ex["_meta"])
    table = pa.Table.from_pylist(examples)
    pq.write_table(table, parquet, compression="zstd")
    bytes_before = jsonl.stat().st_size
    bytes_after = parquet.stat().st_size
    jsonl.unlink()
    return {
        "tile": tile_relpath,
        "n": len(examples),
        "bytes_before": bytes_before,
        "bytes_after": bytes_after,
        "ratio": bytes_after / max(1, bytes_before),
    }


def pack_until_under_threshold(
    curriculum_path: Path,
    out_dir: Path,
    state_dir: Path,
    threshold_bytes: int,
) -> dict:
    """Pack oldest fully-completed tiles until disk usage is below threshold."""
    curr = Curriculum.from_yaml(curriculum_path)
    registry = TileRegistry(state_dir / "registry.json")

    results = []
    while True:
        used = disk_usage_bytes(out_dir)
        if used < threshold_bytes:
            break
        # Pick oldest fully-completed tile that's still in JSONL form.
        candidates = []
        for t in curr.tiles:
            if not registry.is_generated(t):
                continue
            jsonl = out_dir / t.disk_path
            if not jsonl.exists():
                continue
            ts = registry.get(t.id).last_generated_at or 0.0
            candidates.append((ts, t))
        if not candidates:
            break
        candidates.sort(key=lambda c: c[0])
        _, victim = candidates[0]
        r = pack_tile(out_dir, victim.disk_path)
        r["tile_id"] = victim.id
        results.append(r)

    return {
        "final_bytes": disk_usage_bytes(out_dir),
        "threshold_bytes": threshold_bytes,
        "n_packed": len(results),
        "results": results[:10],  # cap output size
    }


def cmd_status(args: argparse.Namespace) -> None:
    out_dir = Path(args.out)
    used = disk_usage_bytes(out_dir)
    print(f"data dir:        {out_dir}")
    print(f"used:            {used:,} bytes  ({used / GB:.3f} GB, {used / TB:.4f} TB)")
    print(f"threshold:       {args.threshold_tb:.2f} TB  ({int(args.threshold_tb * TB):,} bytes)")
    pct = 100 * used / (args.threshold_tb * TB)
    print(f"% of threshold:  {pct:.2f}%")
    n_jsonl = sum(1 for _ in Path(out_dir).rglob("*.jsonl"))
    n_parquet = sum(1 for _ in Path(out_dir).rglob("*.parquet"))
    print(f"tile files:      {n_jsonl} JSONL, {n_parquet} parquet")


def cmd_pack(args: argparse.Namespace) -> None:
    threshold = int(args.threshold_tb * TB)
    out_dir = Path(args.out)
    used = disk_usage_bytes(out_dir)
    if used < threshold:
        print(f"under threshold ({used:,} < {threshold:,}); nothing to pack")
        return
    result = pack_until_under_threshold(
        Path(args.curriculum), out_dir, Path(args.state_dir), threshold
    )
    print(json.dumps(result, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--curriculum",
                    default=str(EXPERIMENT_DIR / "curriculum.yaml"))
    ap.add_argument("--out",
                    default=str(EXPERIMENT_DIR / "data"))
    ap.add_argument("--state-dir",
                    default=str(EXPERIMENT_DIR / "state"))
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("status")
    s.add_argument("--threshold-tb", type=float, default=3.0)
    s.set_defaults(func=cmd_status)
    p = sub.add_parser("pack")
    p.add_argument("--threshold-tb", type=float, default=3.0)
    p.set_defaults(func=cmd_pack)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
