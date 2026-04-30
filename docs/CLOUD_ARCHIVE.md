# Cloud Archive — durable file storage for corpora and checkpoints

`cloud_archive.py` is the file-level analog of `experiment_pusher.py`:
every experiment can call it, it never blocks training, and any
network failure spools to a local outbox for later replay.

This is the answer to "the m4-mini 4 TB external drive is fragile —
requires the mini to be online and the share to be mounted." Files
stop relying on a single always-on host and live in a content-
addressable durable store with no egress fees.

## Backend

Default: **Cloudflare R2** (S3-compatible, 10 GB free, **unlimited
egress** — the standout deal for our use case where the same corpus
gets pulled to vast.ai / m4-pro / m4-mini repeatedly).

R2 speaks the S3 API, so the same code works for **Backblaze B2,
AWS S3, MinIO**, or any S3-compatible store — only the endpoint URL
changes.

## Configuration

Four environment variables. Put them in your shell rc or a `.env`
that's gitignored:

```bash
export R2_ACCESS_KEY_ID="<your access key id>"
export R2_SECRET_ACCESS_KEY="<your secret>"
export R2_ENDPOINT_URL="https://<account_id>.r2.cloudflarestorage.com"
export R2_BUCKET="mamba3-archive"     # default
export R2_REGION="auto"               # default; Cloudflare's edge
```

If any of `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`, or
`R2_ENDPOINT_URL` is missing, `cloud_archive.py` becomes a quiet
no-op — every method returns immediately. **Trainers can be
unconditionally wired to it** and runs without credentials still
work, just without remote archive.

## Setup recipe (5 minutes, browser)

1. Sign in at https://dash.cloudflare.com
2. R2 → "Get started" (asks for a payment method even on free tier;
   no charge under 10 GB / unlimited egress)
3. Create bucket `mamba3-archive` (region: `auto`)
4. R2 → Manage R2 API Tokens → Create token
   - Permissions: **Object Read & Write**
   - Scoped to: `mamba3-archive` bucket only
5. Copy the access key, secret, and endpoint URL into your shell rc.

## Smoke test

```bash
python cloud_archive.py
```

Roundtrips a 1.5 KB JSONL file: upload → list via raw `head_object` →
download → byte-diff → delete. Prints `[smoke] PASS` on success.

## Path convention

Remote keys mirror the Firebase telemetry path so the dashboard and
the archive can show the same run:

```
<bucket>/<experiment_kind>/<run_name>/<artifact_kind>/<filename>
```

Examples:
```
mamba3-archive/cortex_bilingual/step_FINAL/checkpoint/step_010000.pt
mamba3-archive/cerebras-bilingual/2026-04-30/corpus/cerebras_bilingual.jsonl.gz
mamba3-archive/jepa/gpu3-recurse-n3/teacher_thoughts/teacher_thoughts.bin
```

## Compression

Files compress to `.gz` automatically based on extension or content
heuristic:

- **Compressed:** `.jsonl`, `.txt`, `.log`, `.csv`, `.tsv`, `.md`,
  and any unknown file that scans as mostly-printable ASCII
- **Not compressed:** `.pt`, `.pth`, `.npz`, `.safetensors`,
  `.png`, `.jpg`, `.jpeg`, `.webp`, `.bin`, `.idx` (already
  compressed or binary)

Typical bilingual corpus saves ~5-10× space when gzipped.

## Content-addressed dedupe

Before each upload, the client checks if the remote object already
has a matching `sha256` metadata field. If yes, the upload is
skipped and counted toward the success total. This means:

- Re-running a deterministic gen script (`make_bilingual_corpus.py`,
  fixed seed) is **free** on the second run — no transfer.
- Resuming an interrupted training run only re-uploads the changed
  checkpoints.

## Usage from a trainer

The pattern matches `experiment_pusher.py`:

```python
from cloud_archive import CloudArchive

archive = CloudArchive(
    experiment_kind="cortex_bilingual",
    run_name=cfg.run_name or ckpt_dir.name,
    outbox_dir=str(ckpt_dir),
)

# After saving a checkpoint locally:
archive.upload(ckpt_path, artifact_kind="checkpoint",
               metadata={"step": step})

# At end of run:
archive.complete()
```

Calls are non-blocking. The daemon thread drains a queue; failures
spool to `<outbox_dir>/cloud_archive_outbox.jsonl` for replay on the
next run.

## Failure modes

- **No creds:** silent no-op. Trainers run normally; nothing uploads.
- **Network drop mid-upload:** the failed job goes to the outbox
  JSONL. On the next run that constructs `CloudArchive` with the
  same `outbox_dir`, `complete()` replays the outbox.
- **R2 quota exceeded:** uploads start failing with `ClientError`,
  which spools to outbox. Storage and egress quotas show in the
  Cloudflare dashboard; we'll see them before they bite.
- **Bucket missing / wrong endpoint:** `head_object` and `put_object`
  raise; everything spools. `python cloud_archive.py` (smoke) tells
  you immediately.

## Retention (out of scope for v1)

A separate `cloud_archive_prune.py` will eventually keep:
- Latest checkpoint
- Every Nth step (configurable, default 10)
- Final checkpoint (always)
- Latest sample log

Deleting older intermediate checkpoints. Free-tier R2 (10 GB) can
hold ~1800 of our 5.5 MB checkpoints, so retention only matters once
several long runs have accumulated.
