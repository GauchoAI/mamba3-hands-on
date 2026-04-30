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

## Two-tier streams: hot Firebase log + cold HF Parquet

Streams are how producers write growing series of small records
(metrics, samples, events). Each push lands in **two places**:

1. **Local JSONL shard** at `<run_dir>/streams/<stream>-<UTC-date>.jsonl`
   — canonical, append-only, one record per line. The packer reads
   from here.
2. **Firebase RTDB** at `/streams/<exp>/<run>/<stream>/<UTC-date>/<auto-id>`
   — live mirror, dashboardable, deleted on seal.

A **meta node** tracks the live state of each stream:

```yaml
# /streams_meta/<exp>/<run>/<stream>
stream:                  "metrics"
hf_user:                 "miguelemosreverte"
hf_bucket:               "GauchoAI"
prefix:                  "cortex_bilingual/lm-bf16/streams"
url_browse_template:     "https://huggingface.co/buckets/{hf_user}/{hf_bucket}/{prefix}/{filename}"
url_hfsync_template:     "hf://buckets/{hf_user}/{hf_bucket}/{prefix}/{filename}"
pack_threshold_bytes:    10485760     # 10 MB
pack_threshold_records:  50000
pack_threshold_hours:    24.0
current_size_bytes:      12345
current_record_count:    87
pack_progress_pct:       0.18
last_pack_at:            null
last_pack_filename:      null
shard_started_at:        1714521600
```

A live record is then **minimal** — no URL strings, no run/exp
fields (those are the path):

```yaml
# /streams/<exp>/<run>/<stream>/<date>/<auto-id>
ts:    1714521602.5
step:  100
loss:  1.23
lr:    0.001
```

The dashboard joins `meta.url_browse_template` + the shard
filename to render a clickable link to the Parquet on HF. **No URL
is ever stored per-record.** This is the prefix/suffix
compression: the convention is the function, the records are the
arguments.

## Producer API

`experiment_pusher.py` exposes three methods:

```python
p = ExperimentPusher(experiment_id="cortex_bilingual-2026-04-30",
                     run_id="lm-bf16",
                     kind="cortex_bilingual",
                     config=cfg, outbox_dir=run_dir)
p.declare_run(...)

# Optional — a stream is auto-declared on first push() with defaults.
p.declare_stream("metrics",
                 pack_threshold_bytes=10*1024*1024,
                 pack_threshold_records=50_000,
                 pack_threshold_hours=24.0)

# Push records (cheap, non-blocking, dual-write).
p.stream("metrics", {"step": 100, "loss": 1.23, "lr": 1e-3})

# After kappa_packer rolls a shard to Parquet:
p.seal_stream("metrics", "metrics-2026-04-30.jsonl")
# (kappa_packer.py does this for you via REST when given a manifest.)
```

The existing `pusher.metrics(step, **values)` /
`pusher.canary_sample(step, prompt, completion)` / `pusher.event(...)`
are now **backward-compat shims**: they internally call
`pusher.stream("metrics" | "samples" | "events", ...)`. So every
trainer in the repo gets local JSONL + meta tracking + RTDB live
mirror automatically with zero migration churn.

## Per-record size cap

**Records >256 KB are rejected with a hard error.** Silent
truncation is the worst possible bug. Truncate fields at the call
site or split the record into two pushes.

Most records are <1 KB (metrics, events). Sample records can hit
~10 KB if the prompt + completion are long. Both fit comfortably.

## Pack triggers

`kappa_packer.py` packs a shard when ANY of these is reached:

| Trigger | Default | Why |
|---|---|---|
| Size | 10 MB JSONL (uncompressed) | ~1-2 MB Parquet after zstd; fast read, big enough to amortize column overhead |
| Count | 50,000 records | bounds RTDB record count per shard |
| Age | 24 hours | guarantees a daily rollover even if traffic is sparse |

First trigger wins. Override per-stream via `declare_stream()`.

## Run manifest

The pusher writes `<run_dir>/_kappa_manifest.json` on init:

```json
{
  "experiment_id": "cortex_bilingual-2026-04-30",
  "run_id":        "lm-bf16",
  "kind":          "cortex_bilingual",
  "firebase_url":  "https://signaling-...",
  "hf_user":       "miguelemosreverte",
  "hf_bucket":     "GauchoAI"
}
```

`kappa_packer.py` walks upward from any JSONL shard to find this
manifest, then talks to RTDB directly (no in-process pusher needed)
to issue the DELETE + meta PATCH. So packing can happen between
training runs without re-instantiating anything.

## Reader

`stream_reader.py` is the merged-source consumer API. Three calls:

```python
from stream_reader import list_streams, read_meta, read_stream

# discovery — what streams exist project-wide?
for exp_id, run_id, name in list_streams():
    print(exp_id, run_id, name)

# single stream's meta
meta = read_meta("cortex_bilingual-2026-04-30",
                 "mlx-bilingual-widerN-2026-04-30",
                 "metrics")

# transparent record iterator — sealed Parquet (HF) + live RTDB,
# merged by `ts` within each UTC-date shard, dates in chronological
# order
for rec in read_stream(experiment_id, run_id, "metrics",
                       since="2026-04-25", until="2026-05-05"):
    print(rec["step"], rec["byte_ce"])
```

The caller sees one ordered iterator and never has to know which
store any given record came from. The reader does what the
dashboard would otherwise need to do itself: list sealed Parquet
shards on HF, list any open RTDB dates, merge.

Sealed shards are cached at `~/.cache/mamba3-archive/<bucket>/...`
(override with `MAMBA3_ARCHIVE_CACHE` env var). Re-reads are local.

A date can appear in **both** stores: when a producer is mid-day on
a date that ALSO has an intra-day Parquet (because a seal happened
between two record bursts on the same UTC day). The two record sets
are **disjoint by construction** — `seal_via_rtdb` issues `DELETE`
on the date subtree before pack returns, so any records present in
RTDB after the seal are strictly new pushes. The reader concatenates
both and sorts by `ts` without double-counting.

Quick CLI:

```bash
python stream_reader.py ls                                   # list everything
python stream_reader.py meta <exp> <run> <stream>            # one stream's meta
python stream_reader.py read <exp> <run> <stream> --limit 10 # read records
```

## Why Kappa, not Lambda

Lambda architecture splits batch and streaming processing into two
parallel codepaths. Kappa unifies them: there's one path (the log),
and views are computed by reading the log either incrementally
(streaming) or in bulk (batch).

Our entire stack is already log-shaped — append-only files,
content-addressable artifacts, derived Parquet views. The Lambda
duplication doesn't apply here because we have one source of truth
(the JSONL/blob log), not a separate batch warehouse.
