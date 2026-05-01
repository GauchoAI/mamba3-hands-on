# Lab Run and Lab Book

`LabRun` is the unified trainer API for live telemetry and archival. It keeps
the current v1 Firebase and Hugging Face layout intact while giving new code a
single object to use.

```python
from lab_platform.lab_run import LabRun

run = LabRun(
    experiment_id="cortex_bilingual-2026-04-30",
    run_id="mlx-widerN",
    kind="cortex_bilingual",
    config=cfg,
    out_dir=ckpt_dir,
)
run.start(name="cortex_bilingual", purpose="MLX widerN run")
run.metric(step=100, byte_ce=1.2, bpc=1.7)
run.sample(step=100, prompt="The cat ", completion="sat")
run.event("milestone", step=100, details="checkpoint saved")
run.complete(final_metrics={"byte_ce": 1.2})
```

Under the hood:

- Firebase remains the live/control plane.
- Hugging Face remains the durable archive.
- Local JSONL remains the write-ahead stream log.
- Existing UI consumers can keep reading the same v1 paths.

## Lab Book

`docs/lab_book/index.html` is a static dashboard with chapter-style navigation:

- Overview
- Experiments
- Streams
- Archive
- Nodes
- Schema

Run it locally:

```bash
lab book
# or
lab-book
```

The page reads the existing v1 Firebase paths and also looks for optional future
v2 archive records at `/archive_v2/sealed_shards`. If those records are absent,
the UI falls back to the current `streams_meta` archive pointers.

## V2 Direction

The first additive v2 contract is `_schemas/sealed_shard/v2`. It does not
replace `stream_meta`; it defines the richer archive index we want after the
packer verifies a Parquet shard on Hugging Face.

Current data stays valid. Future readers should support both:

- v1: `streams_meta/<experiment>/<run>/<stream>`
- v2: `archive_v2/sealed_shards/...`
