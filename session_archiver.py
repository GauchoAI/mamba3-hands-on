"""session_archiver.py — back up Claude Code session jsonls to Kappa.

Reads every `.jsonl` under `~/.claude/projects/<encoded-folder>/` and pushes
each record through `ExperimentPusher.stream("events", record)`. The pusher
handles tiering automatically: live records sit on Firebase, sealed shards
land on HF as zstd Parquet, dates older than the latest write auto-pack.
The reader (`stream_reader.py`) merges sealed + live transparently.

We push **every** record verbatim — including `queue-operation`,
`file-history-snapshot`, `last-prompt`, `permission-mode`, `ai-title`. The
archive's purpose is "reopen this session correctly on another machine,"
which means losing nothing. Filtering would save ~7% of bytes and risk
breaking session-reopen semantics. Not worth it.

Dedup: records carry stable IDs (`uuid` for conversation events,
`leafUuid` / `messageId` / `sessionId` for housekeeping types).  We
maintain a per-session set in `~/.cache/mamba3-archive/session_archiver/`
so re-runs are idempotent.

Date routing: each record's `timestamp` (ISO-8601) is converted to a
float epoch and passed to `pusher.stream()` as `ts`, so the pusher's
date-aware shard routing places it in the original day's shard. Records
without a timestamp (~7% of them — the housekeeping types) fall back to
the parent record's mtime.

There's no separate backfill vs watch mode — the first poll iteration
naturally handles records not yet in the dedup state (= "backfill");
subsequent iterations handle new appends (= "realtime"). Same loop,
same code path. SIGINT to stop.

CLI:
    python session_archiver.py                # poll forever, 30 s interval
    python session_archiver.py --session UUID # restrict to one session
    python session_archiver.py --interval 5   # tighter polling
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Iterator

from experiment_pusher import ExperimentPusher
try:
    from cloud_archive import CloudArchive
    _HAS_ARCHIVE = True
except ImportError:
    _HAS_ARCHIVE = False

DEFAULT_PROJECTS_DIR = Path.home() / ".claude" / "projects"
DEFAULT_STATE_DIR = Path.home() / ".cache" / "mamba3-archive" / "session_archiver"

EXPERIMENT_ID = "claude-sessions"


def _encoded_folder_for(repo: Path) -> str:
    """Claude Code encodes the absolute path by replacing `/` and `_`
    with `-`. The leading slash becomes a leading dash."""
    return str(repo.resolve()).replace("/", "-").replace("_", "-")


def _iter_jsonl(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _record_id(rec: dict, *, line_no: int, session_uuid: str) -> str:
    """Stable identifier for dedup. Most records carry uuid / leafUuid /
    messageId. The few that don't (`permission-mode`, `ai-title`) get a
    hash of (session_uuid, line_no, content) as a synthetic key.
    """
    for key in ("uuid", "leafUuid", "messageId"):
        v = rec.get(key)
        if isinstance(v, str) and v:
            return v
    blob = json.dumps(rec, sort_keys=True, default=str).encode("utf-8")
    return f"{session_uuid}:{line_no}:" + hashlib.sha1(blob).hexdigest()[:16]


def _record_ts(rec: dict, fallback: float) -> float:
    """Convert ISO-8601 `timestamp` to epoch seconds; `fallback` if absent."""
    iso = rec.get("timestamp")
    if isinstance(iso, str) and iso:
        try:
            # `2026-04-29T15:02:07.445Z` → fromisoformat with Z mapped to +00:00
            return datetime.fromisoformat(
                iso.replace("Z", "+00:00")
            ).timestamp()
        except ValueError:
            pass
    return fallback


def _load_seen(state_path: Path) -> set[str]:
    if not state_path.exists():
        return set()
    return {line.strip() for line in state_path.read_text().splitlines() if line.strip()}


def _append_seen(state_path: Path, ids: list[str]) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    with state_path.open("a") as f:
        for rid in ids:
            f.write(rid + "\n")


def archive_one(jsonl_path: Path, state_dir: Path,
                outbox_dir: Path) -> dict:
    """Archive a single session jsonl. Returns a small report."""
    session_uuid = jsonl_path.stem
    state_path = state_dir / f"{session_uuid}.uuids"
    seen = _load_seen(state_path)
    fallback_ts = jsonl_path.stat().st_mtime

    session_outbox = outbox_dir / session_uuid
    pusher = ExperimentPusher(
        experiment_id=EXPERIMENT_ID,
        run_id=session_uuid,
        kind="claude-sessions",
        config={"source_path": str(jsonl_path)},
        outbox_dir=session_outbox,
    )
    pusher.declare_run(
        purpose=f"Claude Code session archive: {jsonl_path.name}"
    )
    pusher.declare_stream("events")

    # CloudArchive uploads anything under session_outbox/ to
    # `<user>/<bucket>/claude-sessions/<session_uuid>/...` — matches the
    # URL template that pusher.declare_stream wrote into the meta node.
    archive = None
    if _HAS_ARCHIVE:
        archive = CloudArchive(
            experiment_kind="claude-sessions",
            run_name=session_uuid,
            local_dir=str(session_outbox),
            sync_every_s=30,
        )

    new_ids: list[str] = []
    n_total = n_new = n_skipped = 0
    for line_no, rec in enumerate(_iter_jsonl(jsonl_path)):
        n_total += 1
        rid = _record_id(rec, line_no=line_no, session_uuid=session_uuid)
        if rid in seen:
            n_skipped += 1
            continue
        # Inject ts (epoch) for date-aware shard routing inside the pusher.
        # We mutate a copy so the original record is preserved verbatim
        # in `_payload`.
        # JSON-encode the original record. Session records have wildly
        # heterogeneous shapes (some fields list, some dict, some null)
        # which breaks pyarrow column inference at pack time. A single
        # string column is uniform, packs well under zstd, and round-trips
        # losslessly via json.loads.
        rec_with_ts = {
            "_id": rid,
            "ts": _record_ts(rec, fallback_ts),
            "type": rec.get("type", "?"),
            "session_id": rec.get("sessionId", session_uuid),
            "_payload": json.dumps(rec, default=str, ensure_ascii=False),
        }
        try:
            pusher.stream("events", rec_with_ts)
        except ValueError as e:
            # Records >256 KB get rejected by the pusher's safety cap.
            # Surface and skip — almost always a giant tool result.
            print(f"  ! line {line_no}: {e}", flush=True)
            continue
        new_ids.append(rid)
        n_new += 1

    pusher.complete()
    if archive is not None:
        archive.complete()
    _append_seen(state_path, new_ids)

    return {
        "session": session_uuid,
        "total": n_total,
        "new": n_new,
        "skipped_seen": n_skipped,
    }


def list_sessions(repo: Path, projects_dir: Path) -> list[Path]:
    folder = projects_dir / _encoded_folder_for(repo)
    if not folder.is_dir():
        return []
    return sorted(folder.glob("*.jsonl"))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--repo", default=str(Path.cwd()),
                    help="repo path whose sessions to archive (default: cwd)")
    ap.add_argument("--projects-dir", default=str(DEFAULT_PROJECTS_DIR),
                    help="Claude Code projects dir")
    ap.add_argument("--state-dir", default=str(DEFAULT_STATE_DIR),
                    help="dedup state dir")
    ap.add_argument("--outbox-dir", default="runs/claude-sessions",
                    help="local kappa run dir for the streams")
    ap.add_argument("--session",
                    help="archive only this session uuid")
    ap.add_argument("--interval", type=int, default=30,
                    help="poll interval in seconds (default 30)")
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    projects_dir = Path(args.projects_dir)
    state_dir = Path(args.state_dir)
    outbox_dir = Path(args.outbox_dir)

    while True:
        sessions = list_sessions(repo, projects_dir)
        if args.session:
            sessions = [p for p in sessions if p.stem == args.session]
        if not sessions:
            print(f"no session jsonls found under "
                  f"{projects_dir / _encoded_folder_for(repo)}")
        for jsonl in sessions:
            print(f"archiving {jsonl.name} ({jsonl.stat().st_size:,} bytes)…",
                  flush=True)
            t0 = time.time()
            r = archive_one(jsonl, state_dir, outbox_dir)
            dt = time.time() - t0
            print(f"  total={r['total']:,}  new={r['new']:,}  "
                  f"skipped_seen={r['skipped_seen']:,}  ({dt:.1f}s)",
                  flush=True)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
