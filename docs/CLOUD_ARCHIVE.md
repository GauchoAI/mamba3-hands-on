# Cloud Archive — HuggingFace Buckets durable archive

`cloud_archive.py` is the file-level analog of `experiment_pusher.py`:
every experiment can call it, it never blocks training, and any
network failure is logged and retried on the next sync tick.

## Backend: HuggingFace Buckets

**HF Buckets** is HF's newer S3-style file-archive product (distinct
from datasets/models repos which are git+LFS-backed). One bucket
holds many experiments via path prefixes — no proliferating repos
per experiment.

```bash
# rsync-style upload from local dir to bucket
hf sync ./checkpoints hf://buckets/miguelemosreverte/GauchoAI

# rsync-style download from bucket to local dir
hf sync hf://buckets/miguelemosreverte/GauchoAI ./local
```

The Python equivalent (what `cloud_archive.py` calls under the hood):
`HfApi.sync_bucket(source, dest)`.

## Why HF — recap of the path that got us here

| Provider | Result |
|---|---|
| Cloudflare R2 | TLS handshake blocked by EG corporate VPN — verified by direct curl |
| Firebase Cloud Storage | Works on VPN, but $0.12/GB egress — kills our "pull corpus from each compute box" pattern |
| Backblaze B2 | Works, generous tier, but yet another cloud account to manage |
| **HuggingFace Buckets** | Works on VPN, free egress, native to ML, single auth token |

HF wins because:
- **Free egress in both directions** — every training run pulls the
  corpus, and we don't get billed
- **Free at our scale and beyond** — public buckets unlimited, private
  generous; our realistic 6-month working set is 20-100 GB
- **Single auth** — same `HF_TOKEN` we already use for downloading
  teacher models (Qwen, Llama, etc.)
- **ML-native** — the artifact discovery and sharing path is HF-shaped
- **Works through the EG VPN** — `huggingface.co` is widely allowlisted

## Configuration

This is a public repo and the bucket
[`hf://buckets/miguelemosreverte/GauchoAI`](https://huggingface.co/buckets/miguelemosreverte/GauchoAI)
is public. The non-secret defaults are baked into `cloud_archive.py`
so anyone who clones + sets a write token can use it. **Only `HF_TOKEN`
is a secret.**

In `.env` (gitignored):

```bash
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

That's the whole thing. Optional overrides if you want to point at a
different bucket:

```bash
HF_ARCHIVE_USER=<your-user-or-org>
HF_ARCHIVE_BUCKET=<your-bucket>
HF_ARCHIVE_PRIVATE=1                # default 0 (public)
HF_ARCHIVE_SYNC_EVERY=60            # seconds between background syncs
```

If `HF_TOKEN` is missing, `cloud_archive.py` is a quiet no-op — every
method returns immediately. **Trainers can be unconditionally wired**
and runs without a token still work (without remote archive).

## Setup recipe (3 minutes)

1. Sign in at https://huggingface.co (or sign up — free, no card).
2. **Settings → Access Tokens → Create new token**
   - Type: **Write** (so the token can upload to the bucket)
   - Name: `mamba3-archive` or whatever, cosmetic
   - Copy the token (starts with `hf_...`)
3. Drop it into `.env`:
   ```
   HF_TOKEN=hf_...
   ```
4. The bucket `miguelemosreverte/GauchoAI` is already created and
   public — `cloud_archive.py`'s first call will use it.

## Smoke test

```bash
set -a; source .env; set +a
python cloud_archive.py
```

Roundtrips a 1.5 KB JSONL file: write locally → bucket-sync → list
remote → bucket-sync back → byte-diff. Prints `[smoke] PASS` on
success and the URL where you can browse the file.

## Path convention

Remote keys mirror the Firebase telemetry path so the dashboard and
the archive can show the same run:

```
hf://buckets/<user>/<bucket>/<experiment_kind>/<run_name>/<filename>
```

Examples:
```
hf://buckets/miguelemosreverte/GauchoAI/cortex_bilingual/step_FINAL/step_010000.pt
hf://buckets/miguelemosreverte/GauchoAI/cerebras-bilingual/2026-04-30/cerebras_bilingual.jsonl
hf://buckets/miguelemosreverte/GauchoAI/jepa/gpu3-recurse-n3/teacher_thoughts.bin
```

## How the sync works

- The trainer's existing `checkpoints/<run>/` or `data/` directory IS
  the staging area — `CloudArchive` mirrors it without copying.
- A daemon thread runs `sync_bucket(local, remote)` every
  `HF_ARCHIVE_SYNC_EVERY` seconds (default 60). Sync is rsync-style:
  only new or changed files transfer.
- `complete()` triggers a final synchronous flush + stops the thread.
- Network failures are logged and retried on the next tick — the
  trainer never blocks on network.

## Usage from a trainer (the wiring pattern)

```python
from cloud_archive import CloudArchive

archive = CloudArchive(
    experiment_kind="cortex_bilingual",
    run_name=cfg.run_name or ckpt_dir.name,
    local_dir=str(ckpt_dir),
)

# ... trainer runs, writes checkpoints to ckpt_dir, samples to its log ...
# Periodic syncs happen automatically in the background.

archive.complete()  # final sync at end of run
```

That's it. No per-file `archive.upload(path)` calls needed — the
sync covers the whole directory. (The `upload()` method exists as
a compat shim for the old per-file API; it's a no-op now.)

## Failure modes

- **No creds:** silent no-op. Trainers run normally; nothing uploads.
- **Network drop mid-sync:** sync raises, gets logged, next tick retries.
  No data lost — files are still on local disk.
- **HF rate limiting:** rare at our scale. Sync would 429; retries
  on next tick.
- **Bucket access lost:** `create_bucket` and `sync_bucket` would 401/403;
  this surfaces in the trainer's stdout.
- **Wrong username/bucket:** sync fails immediately with 404; check
  `HF_ARCHIVE_USER` and `HF_ARCHIVE_BUCKET`.

## Public vs private

Default is **private** (`HF_ARCHIVE_PRIVATE=1`). Reasons to flip a
specific run public later:
- Tatoeba-derived bilingual corpora are CC-BY anyway; sharing helps
  reproducibility
- Trained checkpoints + findings could be a public release
- Public buckets give discoverability + community traction

To make a specific bucket public, do it in the HF dashboard
(Settings → Visibility on the bucket page).
