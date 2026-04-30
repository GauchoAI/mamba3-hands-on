# Kappa architecture for the Mamba-3 hands-on repo

> **TL;DR.** Append-only logs are the source of truth. Materialized
> views (Parquet, model checkpoints) are derived from the logs and
> can be rebuilt at any time. Storage has two always-on sinks
> (HuggingFace bucket + LAN-attached 4 TB disk). Code propagates
> across machines automatically. Firebase signals; it does not store.

This document codifies the data-flow conventions every producer in
this repo should follow. New experiments inherit them automatically
by using the existing primitives (`cloud_archive.py`,
`experiment_pusher.py`, `kappa_packer.py`).

## The five invariants

### 1. Files are immutable once written

Once a file is on disk, it is **never modified in place**. New data
goes into a new file with a unique name. The disk is an append-only
log of files.

In practice:

| Stream | Naming | Mutation rule |
|---|---|---|
| Checkpoints | `step_NNNNNN.pt`, `step_FINAL.pt` | new file per save; never overwritten |
| Training metrics (JSONL) | `metrics-YYYY-MM-DD.jsonl` | append rows only; one file per UTC day |
| Generated samples | `samples-<run>-YYYY-MM-DD.jsonl` | append rows only |
| Telemetry events | pushed to Firebase RTDB (write-once at unique paths) | RTDB is the log |
| Corpora | `<source>-<YYYY-MM-DD>.jsonl` (raw) → `.parquet` (packed) | written once, then promoted to Parquet |

Why: immutability makes archiving idempotent. `sync_bucket` and
`rsync` only need to compare names + sizes; they can never
silently rewrite an existing remote artifact.

### 2. JSONL is the log; Parquet is the materialized view

Producers write JSONL — one record per line, schema-flexible,
crash-safe (a torn write loses at most the in-flight line). After
a shard is "complete" (no more writes coming, e.g. it's older than
24 hours or the run has ended), `kappa_packer.py` reads it and
writes a zstd-compressed Parquet file alongside, then deletes the
JSONL.

Reader contract — every consumer should look for `<base>.parquet`
first, then `<base>.jsonl`:

```python
from kappa_packer import read_records
rows = read_records(Path("data/streams/metrics-2026-04-29.jsonl"))
# Reads .parquet if it exists, otherwise .jsonl. Same dict-list
# either way.
```

Why: JSONL is fast to write incrementally and trivial to debug
(grep, head, tail). Parquet is fast to read (column-oriented,
queryable) and 30-70% smaller (zstd). Move data from one to the
other once the shard is done; never both, never the wrong format.

### 3. Two always-on storage sinks

`cloud_archive.py` mirrors the local working directory to:

- **Primary:** HuggingFace bucket
  `hf://buckets/miguelemosreverte/GauchoAI/<experiment_kind>/<run_name>/`
  — durable, off-machine, always reachable from VPN-bound or off-VPN
  hosts, free egress
- **Secondary (opportunistic):** LAN rsync to
  `miguel_lemos@192.168.0.170:/Volumes/<external>/mamba3-archive/<kind>/<run>/`
  — fast LAN reads, redundancy if HF is down, configured via
  `LOCAL_MIRROR_DEST` env var; skipped silently when m4-mini is
  asleep, retried on next sync tick

Both sinks are append-only (consequence of invariant #1). Sync is
rsync-style: only changed files transfer. The trainer's
`local_dir` is the staging area — no extra copies are made.

### 4. Code propagates automatically across machines

`cluster_sync.py --watch` polls the source tree every 15 s; when
something is newer than the last successful sync, it rsyncs to the
configured remote nodes. Editing on m4-pro auto-propagates to
m4-mini without manual intervention. SSH is over LAN (en0); no VPN
involvement.

This means a multi-machine workflow looks like:

```
[m4-pro]  edit code → cluster_sync watch picks it up → rsync → m4-mini
[m4-pro]  run training → CloudArchive picks up writes → HF + LAN mirror
[m4-mini] runs eval/dispatched job, reads same code, archives same way
```

### 5. Firebase signals; it does not store

Firebase Realtime DB is the live-coordination channel:
- `experiment_pusher.py` writes telemetry (per-step metrics, samples,
  events) on a per-run path
- Cross-machine signaling (CLI dispatching jobs to other nodes)
  rides the same RTDB

Firebase free tier: 1 GB total + 10 GB / month transfer. **Files
must not live there.** RTDB is for tiny, frequent, observable
records. Anything that's a "file" lives in HF + the LAN mirror.

## Pipeline summary

```
producer ── writes immutable file ─→ local working dir
                                       │
                                       ├──→ Firebase (telemetry / signaling)
                                       │
                                       ├──→ HF bucket (durable archive)
                                       │
                                       └──→ LAN mirror (4 TB redundant)

kappa_packer.py ── reads old JSONL ─→ writes Parquet ─→ deletes JSONL
                                       │
                                       └──→ next sync uploads Parquet, removes JSONL from sinks

cluster_sync.py --watch ── reads source tree ─→ rsyncs changes to other nodes
```

## Daemon / process-shape

We have two long-lived background components:

- **`experiment_pusher.py` daemon thread** — embedded in every
  trainer, pushes telemetry to Firebase
- **`cloud_archive.py` daemon thread** — embedded in every
  trainer, mirrors the working directory to HF + LAN every
  `HF_ARCHIVE_SYNC_EVERY` s (default 60)

Both are intra-process daemons — they live and die with the trainer
that owns them. **There is no separate watcher process** outside
the producer. While a producer is running, persistence is real-time;
when no producer is running, there's nothing to persist.

`cluster_sync.py --watch` is the one user-launched daemon (run it
in a terminal or as a launchd agent), and it only handles code
propagation, not data.

## Failure modes and resilience

| Failure | Behavior |
|---|---|
| HF rate limit / 429 | sync logs, retries on next tick; trainer continues |
| HF outage | same as above, possibly for many ticks; data accumulates locally; uploads when service returns |
| m4-mini offline | rsync fast-fails (4 s SSH connect timeout); next tick retries |
| Producer crashes | files on disk are still complete (immutability); next CloudArchive instance with the same `local_dir` will pick them up |
| Network partition between machines | each machine archives to HF independently; LAN mirror catches up when reconnected |
| Disk full on a producer | `kappa_packer.py` shrinks JSONL → Parquet (~30-70% reduction); `cluster_sync.py` does not delete on remote unless `--delete` is set |
| Firebase quota blown | telemetry stops, archive continues (different sinks) |

## Adding a new experiment

To follow the convention:

1. Create a folder under `experiments/` (or at repo root if it's a
   first-class line of work). Per-experiment isolation per the
   subfolder convention.
2. Producer scripts open output files with date-shard names:
   ```python
   from datetime import datetime, timezone
   today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
   path = run_dir / f"metrics-{today}.jsonl"
   ```
3. Instantiate `CloudArchive` once per run; let its background
   thread handle persistence:
   ```python
   from cloud_archive import CloudArchive
   archive = CloudArchive(
       experiment_kind="<your-kind>",
       run_name=run_name,
       local_dir=str(run_dir),
   )
   ...
   archive.complete()  # final flush
   ```
4. Schedule the packer to run periodically (cron, launchd, or
   manually before deciding "this run is done"):
   ```bash
   python kappa_packer.py --dir <run_dir> --age-hours 24
   ```
5. Make readers transparent: use `kappa_packer.read_records(path)`
   instead of `open(path).read()`.

That's the whole convention. New experiments inherit Firebase
telemetry, durable HF archive, LAN redundancy, code propagation,
and Parquet compaction by following the four steps above.

## Why Kappa, not Lambda

Lambda architecture splits batch and streaming processing into two
parallel codepaths. Kappa unifies them: there's one path (the log),
and views are computed by reading the log either incrementally
(streaming) or in bulk (batch).

Our entire stack is already log-shaped — append-only files,
content-addressable artifacts, derived Parquet views. The Lambda
duplication doesn't apply here because we have one source of truth
(the JSONL/blob log), not a separate batch warehouse.
