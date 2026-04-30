# Signoff — 2026-04-30: Kappa architecture + wider-N cortex experiment

End-of-session situation report. Companion to
`signoff/2026-04-29-cortex-counter-attach-on-bilingual-lm.md`
(yesterday's findings) and
`signoff/2026-04-28-mlx-port-and-experiments-queued.md` (the
MLX port).

**Branch:** `main` (tracking `origin/main`).

---

## TL;DR

- **Persistence + telemetry are now project-wide and Kappa-shaped.**
  Every producer's writes land in three places automatically: local
  JSONL log, Firebase RTDB live mirror, HF bucket durable archive.
  Eventually (when a shard ripens) the JSONL gets rolled to
  zstd-compressed Parquet on HF and the matching live records get
  dropped from RTDB.
- **Code propagates across machines automatically.** `cluster_sync.py
  --watch` polls every 15 s; editing on m4-pro auto-rsyncs to m4-mini
  over LAN (no VPN dependency — verified).
- **In-flight experiment to test yesterday's diagnosis:** the
  bilingual LM is being retrained with a wider unary mixin (N up to
  60, doubled from 30) to test whether the cortex-composition OOD
  ceiling came from the LM's "stars are short" prior rather than
  the primitive itself.

---

## What landed this session

### Kappa architecture (commits `520b188`, `506b700`, `e7afdd0`)

The five invariants documented in `docs/KAPPA_ARCHITECTURE.md`:

1. Files are immutable once written
2. JSONL is the log; Parquet is the materialized view
3. Two always-on storage sinks (HF bucket + LAN 4 TB mirror)
4. Code propagates automatically across machines
5. Firebase signals; it does not store

### Concrete pieces

| File | What it is |
|---|---|
| `experiment_pusher.py` | Adds `declare_stream/stream/seal_stream`; `metrics/canary_sample/event` are now backward-compat shims that route through `stream()`. Writes `_kappa_manifest.json` on init. |
| `kappa_packer.py` | Generic JSONL→Parquet packer. `find_manifest()` walks up from any shard, `seal_via_rtdb()` issues DELETE + meta PATCH after a successful pack. Auto-seal by default; `--no-seal` opts out. |
| `cloud_archive.py` | HF bucket sync + optional LAN rsync second sink (`LOCAL_MIRROR_DEST` env var). Background daemon thread inside every producer. |
| `cluster_sync.py` | `--watch` mode for code auto-propagation. ConnectTimeout=4s for fast-fail when m4-mini is offline. |
| `docs/KAPPA_ARCHITECTURE.md` | Full architecture + URL convention + meta schema + adding-a-new-experiment recipe |
| `docs/CLOUD_ARCHIVE.md` | HF Buckets setup, env vars, smoke recipe |

### URL convention (the elegant bit)

Records carry no URL strings. URL *templates* live in the meta node
once per stream:

```
/streams_meta/<exp>/<run>/<stream>:
  url_browse_template:  https://huggingface.co/buckets/{user}/{bucket}/{prefix}/{filename}
  url_hfsync_template:  hf://buckets/{user}/{bucket}/{prefix}/{filename}
  current_record_count, current_size_bytes, pack_progress_pct, ...
```

Records carry only the unique fields (step, loss, sample text). The
dashboard reconstructs a clickable URL by substituting `{filename}`
into the template. Convention is the function; records are the
arguments.

### End-to-end smoke (verified live)

100 records → all 100 in local JSONL + 100 in RTDB → meta shows
`count=100, pct=0.2%` → kappa_packer rolls JSONL to Parquet (34%
ratio with zstd) → auto-seal: `DELETE` records subtree + `PATCH`
last_pack_filename → after seal: `count=0, last_pack_filename set,
records-in-rtdb=0`. All gates green against the live Firebase RTDB.

---

## In-flight experiment

**Hypothesis:** Yesterday's counter-attach result on a frozen
bilingual LM was OOD-capped by the LM's implicit "stars are short →
switch out of unary mode" prior, learned because the bilingual
training corpus only contained `*N:aN` lines with N ≤ 30. The
counter primitive's signal was correct; the LM was the bottleneck.

**Test:** Retrain the same bilingual LM (4 layers d=128, MLX bf16,
10k steps) on a corpus with the unary mixin widened to N up to 60.
Then re-run the counter-attach experiment from yesterday and compare
OOD behavior. Expected paths:

- **OOD extends materially past 60** → diagnosis confirmed; the LM's
  prior shifted, and the counter primitive's contribution is
  successfully overriding the new (higher) cap. Strong-claim
  validation: composition reaches further as the host LM's
  distribution widens.
- **OOD stays roughly the same** → diagnosis was wrong, the issue is
  deeper (e.g., the SSM's hidden state at long sequence positions is
  insufficient regardless of training distribution). Triggers
  next-experiment design: hidden-state distillation (the JEPA
  direction in `jepa/`) becomes the indicated path.
- **Mixed** → partial diagnosis confirmation; the LM prior matters
  but isn't the only factor.

**Status as of signoff:**

| Process | Where | ETA | Output |
|---|---|---|---|
| MLX wider-N bilingual training | m4-pro MLX (Metal), background `be35n7zom` | ~2.5-3 h to step 10k | `checkpoints/lm_mlx_widerN/` |
| Live Kappa pipeline | telemetry to Firebase, archive to HF, all auto | continuous | `/streams_meta/.../mlx-bilingual-widerN-2026-04-30/metrics` |

Reproduction:

```bash
# Wider-N corpus (already regenerated on disk)
python make_bilingual_corpus.py

# Training (this is what's running now)
python cortex_bilingual/train_bilingual_mlx.py \
    --steps 10000 --ckpt-every 500 --log-every 100 \
    --seq-len 128 --dtype bfloat16 \
    --ckpt-dir checkpoints/lm_mlx_widerN \
    --run-name mlx-bilingual-widerN-2026-04-30
```

When training finishes, the next step is the counter-attach + demo
on the new LM:

```bash
python cortex_bilingual/train_counter_attach.py \
    --lm-ckpt checkpoints/lm_mlx_widerN/step_FINAL.npz \
    --steps 1000 --injection-scale 30.0 \
    --out checkpoints/lm_counter_widerN \
    --run-name lm-counter-widerN-2026-04-30

python cortex_bilingual/demo_cortex.py \
    --lm-ckpt checkpoints/lm_mlx_widerN/step_FINAL.npz \
    --counter-ckpt checkpoints/lm_counter_widerN/step_FINAL.pt
```

(Note: if the MLX trainer's checkpoint format isn't directly
PyTorch-loadable for `train_counter_attach.py`, we'll need to
either (a) port `train_counter_attach.py` to MLX, or (b) write a
small npz→pt converter. Cross that bridge when we get there.)

---

## What's NOT done — explicit defer list

These were considered and intentionally deferred:

1. **`stream_reader.py`** (transparent merge of sealed Parquet
   shards + open RTDB tail). Not needed yet — the in-RTDB live
   records + the HF Parquet shards are queried separately by today's
   only consumer (the dashboard). Build when the second consumer
   asks.
2. **Auto-pack inside the trainer.** kappa_packer is a separate
   command. Reasoning: packing reads + rewrites a JSONL the trainer
   may still be appending to. Better to run packer between training
   runs (or via cron) than risk file-write contention.
3. **launchd plist for `cluster_sync.py --watch`.** For now it's
   manually started in a terminal. Ready to install as a permanent
   agent when persistence-on-reboot is wanted.
4. **Migration of `experiments/jepa_structured_data/storage_packer.py`
   to call into the new generic `kappa_packer.py`.** Both work; the
   experiment-local one is scoped to its own curriculum-aware logic.
   Generalize later if it ever matters.
5. **`LOCAL_MIRROR_DEST` env var population.** m4-mini was offline
   when this session ran, so I couldn't peek at `/Volumes/` to find
   the external drive's mount name. When mini comes online, set the
   env var to `miguel_lemos@192.168.0.170:/Volumes/<mount>/mamba3-archive`
   and the LAN mirror activates automatically.

---

## Risks and unknowns for the wider-N experiment

- **Training slower than the seq_len=128 budget?** Should be roughly
  the same — the SSM scan is O(L), L unchanged. If the unary mixin's
  *effective* max length doubles (62 → 122 chars), some batches will
  truncate slightly differently. Monitor for unusual loss curves.
- **Does N=60 actually shift the prior?** It's a doubling, not a
  10× shift. If the LM still has a strong "stars are short" prior
  at N=60, we'll see only marginal OOD extension. In that case,
  next iteration: bump seq_len=256 and N=120.
- **MLX checkpoint compatibility with the (PyTorch) counter-attach
  trainer.** The MLX trainer saves `.npz`, the PyTorch trainer
  expects `.pt`. There may be a conversion step. If it's painful,
  we run the wider-N training in PyTorch instead (slower but
  drop-in compatible).

---

## Pickup checklist

1. Check the live training:
   ```bash
   tail -3 /tmp/mlx_widerN_train.log
   ```
2. Look at the live meta in Firebase RTDB:
   ```bash
   curl -s "https://signaling-dcfad-default-rtdb.europe-west1.firebasedatabase.app/streams_meta/cortex_bilingual-2026-04-30/mlx-bilingual-widerN-2026-04-30/metrics.json" | jq
   ```
3. Browse archived artifacts:
   `https://huggingface.co/buckets/miguelemosreverte/GauchoAI/cortex_bilingual/mlx-bilingual-widerN-2026-04-30/`
4. When training completes (`step_FINAL.npz` exists), run the
   counter-attach experiment as documented above.
5. If results match the diagnosis, write up `findings.md` entry
   "Counter primitive on a wider-N bilingual LM" and consolidate
   the architectural takeaway: composition is bounded by the LM's
   training distribution, not by the primitive.

---

## Commit log this session

```
e7afdd0  Drop redundant cloud_archive_daemon.py; widen Tatoeba unary mixin N≤60
506b700  Kappa stream API: dual-write log + meta tracking + auto-seal on pack
520b188  Kappa architecture: 2-sink persistence + JSONL→Parquet packer + code auto-sync
b44a4fd  Retire handoff/, consolidate session notes under signoff/
... (HF bucket + Firebase wiring earlier in the day)
```

---

## End state

Tree clean. The persistence + telemetry layer is fully migrated and
operational. A real training run is exercising it live. Pickup is
`git pull && tail /tmp/mlx_widerN_train.log && cat
signoff/2026-04-30-kappa-architecture-and-widerN-experiment.md`.
