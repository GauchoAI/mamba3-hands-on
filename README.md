# Lab

A research project building small Mamba-3 byte-level language models
that **reason** through residual-stream primitives. The chapters
below are individual experiments, ordered chronologically. Experiment
sources live under [`experiments/`](experiments/). Closed experiments stay
archived there so the timeline is reconstructable; active ones are clearly
marked.

For the current working route and exact commands, see
[`ACTIVE.md`](ACTIVE.md).

For reusable platform imports, install the checkout in editable mode:

```bash
.venv/bin/python -m pip install -e .
```

This also installs the `lab` CLI plus utility commands such as
`lab-kappa-pack`, `lab-cluster-sync`, and
`lab-make-bilingual-corpus`.

## Chapters

| # | Chapter | Status | Synopsis |
|---|---|---|---|
| 01 | [`experiments/01_ga_tournament/`](experiments/01_ga_tournament/) | archival | Multi-task GA mastered 14 of 15 tasks in ~24 h H100 |
| 02 | [`experiments/02_ptx_engine/`](experiments/02_ptx_engine/) | archival | Hand-rolled PTX Mamba-3 engine (~14× PyTorch); pod-archive branch |
| 03 | [`experiments/03_synapse_parity/`](experiments/03_synapse_parity/) | archival | Early Mamba-3 / RoPE / parity foundations |
| 04 | [`experiments/04_hanoi/`](experiments/04_hanoi/) | archival | Hanoi line: bounded-counter → EOS-bias → tool-use, 5,000× extrapolation |
| 05 | [`experiments/05_lego_library/`](experiments/05_lego_library/) | archival | Step-function library + Light-CA + Cornell renderer |
| 06 | [`experiments/06_cortex_existence/`](experiments/06_cortex_existence/) | archival | 772-param counter primitive, 16.7× OOD on a synthetic LM |
| 07 | [`experiments/07_jepa/`](experiments/07_jepa/) | archival | Original JEPA-Cortex with Qwen-teacher distillation |
| 08 | [`experiments/08_rlf_cortex/`](experiments/08_rlf_cortex/) | archival | RLF-inspired layer-recursion + lifeline |
| 09 | [`experiments/09_cortex_bilingual/`](experiments/09_cortex_bilingual/) | **closed** 2026-04-30 | Counter on bilingual LM; ~17× OOD shift; closed by decision |
| 10 | [`experiments/10_jepa_structured/`](experiments/10_jepa_structured/) | **active** | Structured-data JEPA-Cortex daemon (current daily driver) |

Each chapter has its own `README.md` with synopsis, status, and a
cross-link to the relevant `docs/findings/<topic>.md` entry.

## Infrastructure

The reusable platform modules used across chapters live under
[`src/lab_platform/`](src/lab_platform/):

- **Persistence + telemetry**:
  [`experiment_pusher.py`](src/lab_platform/experiment_pusher.py),
  [`kappa_packer.py`](src/lab_platform/kappa_packer.py),
  [`kappa_schemas.py`](src/lab_platform/kappa_schemas.py),
  [`stream_reader.py`](src/lab_platform/stream_reader.py),
  [`cloud_archive.py`](src/lab_platform/cloud_archive.py),
  [`session_archiver.py`](src/lab_platform/session_archiver.py),
  [`firebase_push.py`](src/lab_platform/firebase_push.py),
  [`firebase_sync.py`](src/lab_platform/firebase_sync.py).
- **Cluster control**:
  [`cluster_dispatch.py`](src/lab_platform/cluster_dispatch.py),
  [`cluster_sync.py`](src/lab_platform/cluster_sync.py).
- **Models + kernels**:
  [`mamba3_minimal.py`](src/lab_platform/mamba3_minimal.py),
  [`mamba3_lm.py`](src/lab_platform/mamba3_lm.py),
  [`cortex_counting.py`](src/lab_platform/cortex_counting.py) (the `Primitive` base
  class + `CortexLM`),
  `ssm_*.py` (SSM scan kernels).
- **Corpora**:
  [`make_bilingual_corpus.py`](src/lab_platform/make_bilingual_corpus.py),
  [`make_opensubtitles_corpus.py`](src/lab_platform/make_opensubtitles_corpus.py),
  [`make_teacher_corpus.py`](src/lab_platform/make_teacher_corpus.py).

## Architecture documents

- [`docs/KAPPA_ARCHITECTURE.md`](docs/KAPPA_ARCHITECTURE.md) — the
  five-invariant Kappa pipeline (immutable JSONL log, Parquet
  materialized view, two-sink storage, code auto-propagation,
  Firebase signals).
- [`docs/CLOUD_ARCHIVE.md`](docs/CLOUD_ARCHIVE.md) — HF Buckets
  setup + LAN mirror.
- [`docs/EXPERIMENT_FIREBASE_SCHEMA.md`](docs/EXPERIMENT_FIREBASE_SCHEMA.md)
  — RTDB layout + free-tier accounting.
- [`docs/UI_VISION.md`](docs/UI_VISION.md) — what the dashboard
  surfaces.
- [`VISION.md`](VISION.md) — project-level vision.
- [`docs/legacy/`](docs/legacy/) — older planning / handoff documents
  preserved for context.

## Findings

Cross-project findings live in
[`findings.md`](findings.md) (root, shorter) and
[`docs/findings/`](docs/findings/) (per-topic, deep). Per-chapter
findings live alongside the chapter (e.g.
[`experiments/09_cortex_bilingual/findings.md`](experiments/09_cortex_bilingual/findings.md)).

## Tools

[`tools/`](tools/) collects analysis / dashboard / cluster-control
scripts that aren't part of any single chapter:
[`tools/dashboard/`](tools/dashboard/),
[`tools/db/`](tools/db/),
[`tools/cluster/`](tools/cluster/),
plus the `analyze_*` / `check_*` / `diagnostician` family.
