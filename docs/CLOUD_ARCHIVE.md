# Cloud Archive — durable file storage for corpora and checkpoints

`cloud_archive.py` is the file-level analog of `experiment_pusher.py`:
every experiment can call it, it never blocks training, and any
network failure spools to a local outbox for later replay.

This is the answer to "the m4-mini 4 TB external drive is fragile —
requires the mini to be online and the share to be mounted." Files
stop relying on a single always-on host and live in a durable
S3-compatible store.

## Backend

Default: **Firebase Cloud Storage** (== Google Cloud Storage backed;
same Firebase project as the telemetry RTDB). 5 GB free, $0.026/GB-
month after, 1 GB/day download free / $0.12/GB egress after.

**Why not Cloudflare R2?** R2 is the better deal in isolation
(10 GB free, unlimited egress) but the EG corporate VPN's TLS
inspection proxy actively blocks `*.r2.cloudflarestorage.com`. We
verified by direct curl: connection succeeds at the IP level but the
TLS handshake fails. Major cloud-storage providers (GCS, AWS S3,
B2) are allowlisted; Cloudflare is not. Firebase Cloud Storage works
through the VPN out of the box because it's the same provider as the
RTDB telemetry, which already works.

The R2 keys are kept commented out in `.env` for the m4-mini path
(off-VPN host, could still upload to R2). If we ever want both
backends active simultaneously, that's a small extension.

The code speaks the plain S3 API, so any S3-compat backend works —
**Backblaze B2, AWS S3, MinIO, R2** — only the endpoint URL changes.

## Configuration

Backend-neutral env vars (preferred):

```bash
export ARCHIVE_ACCESS_KEY_ID="GOOG1E..."           # GCS HMAC key
export ARCHIVE_SECRET_ACCESS_KEY="..."             # 40-char secret
export ARCHIVE_ENDPOINT_URL="https://storage.googleapis.com"
export ARCHIVE_BUCKET="<project>.firebasestorage.app"
export ARCHIVE_REGION="auto"
```

Legacy `R2_*` env vars are still recognized — useful when an
off-VPN box (m4-mini, vast.ai) wants to use the R2 keys directly
without renaming. Code reads `R2_*` first, then falls back to
`ARCHIVE_*`.

If credentials aren't set, `cloud_archive.py` is a quiet no-op:
every method returns immediately. **Trainers can be unconditionally
wired** and runs without credentials still work — just without
remote archive.

## Setup recipe — Firebase Cloud Storage (5-10 min, browser)

Firebase Cloud Storage uses Google Cloud Storage under the hood.
For S3-compatible auth we need GCS's HMAC keys feature.

**1. Enable Storage on your Firebase project**

- https://console.firebase.google.com → your project (the one
  the telemetry RTDB uses) → **Build → Storage** → Get started
- Pick "Production mode" rules
- Region: pick closest, e.g. `europe-west1` to match the RTDB

This creates the default bucket. Copy its name — looks like
`<project-id>.firebasestorage.app` or `<project-id>.appspot.com`.

**2. Create HMAC keys (S3-compatible auth)**

- https://console.cloud.google.com → same project (Firebase ⊆ GCP)
- **Cloud Storage → Settings → Interoperability** tab
- "Create a key for a service account"
  - Either pick existing or create new with **Storage Object Admin**
    role on the bucket
- Result: **Access key** (`GOOG1E...`) and **Secret** (one-time
  display — copy now or you'll need to regenerate)

**3. Set env vars**

```bash
export ARCHIVE_ACCESS_KEY_ID="GOOG1E..."
export ARCHIVE_SECRET_ACCESS_KEY="<40-char secret>"
export ARCHIVE_ENDPOINT_URL="https://storage.googleapis.com"
export ARCHIVE_BUCKET="<your-bucket-name>"
```

Endpoint URL is fixed for everyone using GCS — `https://storage.googleapis.com`.

## Alternative recipe — Cloudflare R2 (off-VPN hosts only)

If running on a host that isn't behind the EG VPN (m4-mini, vast.ai,
personal laptop on home wifi), R2 is still a better deal:

1. Sign in at https://dash.cloudflare.com
2. R2 → "Get started" (asks for a payment method on Spark; no charge
   under 10 GB / unlimited egress)
3. Create bucket `mamba3-archive` (region: `auto`)
4. R2 → Manage R2 API Tokens → Create token
   - Permissions: **Object Read & Write**
   - Scoped to: `mamba3-archive` bucket only

Use the `R2_*` env vars — `R2_ENDPOINT_URL` looks like
`https://<account_id>.r2.cloudflarestorage.com`.

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
