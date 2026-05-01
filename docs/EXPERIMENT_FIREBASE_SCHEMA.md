# Cross-Experiment Firebase Schema

A standardized way for *any* experiment in this repo (`three_populations`,
`jepa/`, `hanoi-exec`, `lego/*`, future ones) to push live training data
to Firebase Realtime DB so a single dashboard can surface them.

The existing `firebase_push.py` is GA-specific (mastery / unlock / evolve /
plateau / lineage events). This doc proposes a *general* layer that
captures the common shape of any experiment's training run, treats the GA
events as one specialization, and stays within the free tier.

The goal of this doc is **schema + write-budget + an imagined UI**, not a
build. Once the shape is agreed, adapters land per-experiment, and the
dashboard reads them.

## 1. Free-tier accounting

We're on Firebase Spark (free). Hard limits we must respect:

| Resource | Limit | What burns it |
|---|---|---|
| Realtime DB storage | **1 GB** | Cumulative records. Old data must rotate. |
| Realtime DB transfer | **10 GB / month** | Upload + dashboard read. |
| Simultaneous connections | **100** | Dashboard tabs + push clients. |
| Firestore writes (if used) | 20K/day | Why we use Realtime DB, not Firestore. |

**The dominant cost is *connection-bytes-per-month*.** A naive trainer
pushing every 50 steps with a 200-byte JSON for 3 days continuous = ~600
KB. Four parallel runs pushing per-step canary samples could blow 1 GB
storage in a week. **Every adopter must respect the budget.**

### Write-budget convention (per-run, per-day, soft cap)

| Channel | Max writes/day | Max bytes/write | Daily bytes |
|---|---|---|---|
| `metrics` (training scalars) | 720 (every 2 min) | 200 B | 144 KB |
| `samples` (canary text) | 24 (every 1 h) | 1 KB | 24 KB |
| `events` (decisions, milestones) | 50 | 300 B | 15 KB |
| `status` (heartbeat / state) | 144 (every 10 min) | 100 B | 14 KB |
| **Total per run** | ~940 writes/day | — | **~200 KB/day** |

Four parallel runs ≈ 800 KB/day. Storage is the constraint that bites
first; at 800 KB/day a run can stay live ~3.5 weeks before it eats
into the 1 GB cap (assuming nothing else is using the DB). Mitigation:
**rotation** — keep only the last N entries per channel (Realtime DB's
[ON_CHILD_REMOVED](https://firebase.google.com/docs/database/admin/retrieve-data)
pattern; the writer deletes its own old entries after each push).

We do **not** push:
- Checkpoints (1 GB cap kills this immediately; checkpoints stay local
  on the box, are rsync'd or copied via filesystem).
- Per-step metrics (way over budget; downsample to every-2-min or every-N-step).
- Full samples on every push (rotate; keep last N=24 per run).
- Full thought tensors / hidden state dumps. Those go to local disk only.

## 2. Top-level schema

Realtime DB tree, written as JSON. Existing `/mamba3/*` paths
(`firebase_push.py`'s GA pushers) stay where they are; the new
cross-experiment surface lives under `/experiments/`:

```
/experiments/
  /<experiment_id>/                 ← e.g. "jepa-cortex-2026-04-29"
    meta:                            ← single document, written once at start
      name: "JEPA-Cortex"
      experiment_id: "jepa-cortex-2026-04-29"
      kind: "jepa"                   ← schema variant: "jepa" | "ga" | "hanoi" | ...
      started_at: 1777400000
      ended_at: null                 ← updated when done
      git_sha: "857e2af"
      box: "vast.ai 4×4070Ti"
      hypothesis: "JEPA latent regularizer beats pure byte CE on small corpus"
      runs_root: "/workspace/.../runs/jepa_cortex/"
      checkpoints_root: "/workspace/.../checkpoints/jepa_cortex/"
    /runs/<run_id>/                  ← one per parallel run
      meta:                           ← written once when the run starts
        run_id: "gpu0-ref"
        started_at: ...
        ended_at: null
        config:                       ← all hyperparameters
          lambda_jepa: 1.0
          lambda_sigreg: 0.1
          lambda_aux: 0.5
          batch_size: 64
          d_model: 192
          ...
        gpu: 0
        purpose: "full-strength reference"
      status:                         ← single document, overwritten on heartbeat
        last_step: 4000
        sps: 0.1
        last_heartbeat: 1777500000
        state: "running"              ← "running" | "completed" | "failed" | "killed"
        gpu_mem_mb: 6694
        gpu_util_pct: 100
      /metrics/                       ← timeseries, append-only with rotation
        <push_key>: { step: 4000, ts: 1777500000, byte_ce_biling: 1.20, ... }
        <push_key>: { step: 4050, ts: 1777500120, byte_ce_biling: 1.19, ... }
      /samples/                       ← canary samples, rolling buffer (last 24)
        <push_key>: {
          step: 4000, ts: 1777500000,
          prompt: "Hola, como estas?\n",
          completion: "She leaned out the window..."
        }
      /events/                        ← decisions + milestones, append-only
        <push_key>: { step: 2900, ts: ..., type: "variant_retired",
                      details: "gpu3-zerojepa retired due to mode collapse",
                      reasoning: "..." }
        <push_key>: { step: 2150, ts: ..., type: "milestone",
                      details: "JEPA term first nonzero contribution" }
```

The four channels — `metrics`, `samples`, `events`, `status` — are
**universal** across experiment kinds. The `meta.config` document is
where experiment-specific hyperparameters live.

## 3. Adopter API

A new module `experiment_pusher.py` (to be written) wraps Firebase
Realtime DB into a tiny client that any trainer can use:

```python
# pseudocode of the proposed API:
from experiment_pusher import ExperimentPusher

p = ExperimentPusher(
    experiment_id="jepa-cortex-2026-04-29",
    run_id="gpu0-ref",
    config=asdict(cfg),
    kind="jepa",
)

# Once at start:
p.declare_experiment(name="JEPA-Cortex", hypothesis="...")
p.declare_run(purpose="full-strength reference", gpu=0)

# In the train loop, throttled by ExperimentPusher (it knows the budget):
p.metrics(step=step, byte_ce=..., jepa_loss=..., intent_var=...)
p.heartbeat(step=step, sps=sps, gpu_mem_mb=..., gpu_util_pct=...)

# Periodically:
p.canary_sample(step=step, prompt=..., completion=...)

# When something interesting happens:
p.event(type="milestone", details="JEPA term firing", step=step)
p.event(type="variant_retired", details="...", reasoning="...")

# At end:
p.complete()  # sets status.state = "completed", ended_at = now
```

The pusher is responsible for:
- **Throttling.** `metrics()` called every step but writes only every 2
  minutes. `samples()` called every 100 steps but writes only every 1 h.
  Throttling keeps adopter code simple — every train loop just calls the
  method on every step, the pusher batches.
- **Rotation.** After each push, deletes oldest entries past the rolling
  window (last 24 samples, last 720 metrics, last 100 events).
- **Compression.** Single metric push contains *multiple steps* of data
  packed into one record (200B vs 200B × 5 = 1000B saved per cycle).
- **Failure tolerance.** Falls back to a local JSONL log on push failure
  so we don't lose data if Firebase is briefly unreachable. Re-pushes on
  next cycle.

## 4. The dashboard, imagined

A single static SPA (no Firebase Functions / no Blaze tier) reading
directly from Realtime DB via the browser SDK. Three screens:

### Screen 1 — Experiments index (`/`)

A grid of experiment cards. One card per `/experiments/<id>/`. Each card
shows:

```
┌────────────────────────────────────────────────────┐
│ JEPA-Cortex                            [running]   │
│ kind: jepa  •  started 2026-04-29 13:14            │
│ 4 runs • 1.0M params • 4×4070Ti                    │
│                                                    │
│ "JEPA latent regularizer beats byte CE on small    │
│  corpus"                                           │
│                                                    │
│ ▰▰▰▰▰▰▰▱▱▱  72% (~step 4000 / 30000)              │
│                                                    │
│ → 4 active runs · 0 completed · 1 retired         │
└────────────────────────────────────────────────────┘
```

Sorted by `started_at` desc. Filter chips: all / running / completed /
failed. Click a card → Screen 2.

### Screen 2 — Experiment detail (`/<experiment_id>`)

The four runs side-by-side. For each run, a small panel:

```
┌── gpu0-ref ─────────────────┐ ┌── gpu1-lowjepa ─────────────────┐
│ λ_jepa=1.0 λ_sig=0.1        │ │ λ_jepa=0.3 λ_sig=0.1            │
│ step 4000 · 0.1 sps         │ │ step 4000 · 0.1 sps             │
│ ╭─chart────────────────╮    │ │ ╭─chart────────────────╮         │
│ │ byte_ce_biling: 1.20 │    │ │ │ byte_ce_biling: 1.19 │         │
│ │  decreasing         ↘   │ │ │  decreasing         ↘   │         │
│ ╰─────────────────────╯    │ │ ╰─────────────────────╯         │
│ Last sample (8 min ago):    │ │ Last sample (8 min ago):        │
│   "Tom se habla cabeza..."  │ │   "10::Diez 11::Once 12::Doce" │
│ [open]                      │ │ [open]                          │
└─────────────────────────────┘ └─────────────────────────────────┘
```

Top of the screen: a synced multi-line chart showing all 4 runs'
`byte_ce_biling` over time on one axis, plus an event log column on
the side ("step 2150 — JEPA term firing", "step 2900 — gpu3-zerojepa
retired").

This is the screen that makes the parallel-experiment story legible.

### Screen 3 — Run detail (`/<experiment_id>/<run_id>`)

Single run, full depth.

```
─── status header ─────────────────────────────────────────────────
  step 4000 · running · gpu0 · 100% util · 6.7 GB / 12 GB
  config: λ_jepa=1.0, λ_sig=0.1, λ_aux=0.5, batch=64, ...
  git: 857e2af  •  box: vast.ai 4×4070Ti  •  uptime 7h 14m
─── metric chart ──────────────────────────────────────────────────
  [tabs: byte_ce | jepa_loss | sigreg | intent_var | counter_acc]
   1.5 ┤
       │      ╱╲
   1.3 ┤    ╱    ╲╱╲
       │  ╱          ╲╲___
   1.1 ┤╱                  
       └────────────────────────────────────────
        0    1000   2000   3000   4000  step
─── canary samples (last 24) ──────────────────────────────────────
  step 4000  >>> Hola, como estas?
              "She leaned out the window to catch a glimpse..."
  step 3900  >>> Tom said
              "What time are we meeting."
  ...
─── decision log ──────────────────────────────────────────────────
  step 2150  milestone        JEPA term first nonzero contribution
  step 2400  observation      gpu0 and gpu1 began diverging
  step 2900  variant_retired  gpu3-zerojepa replaced by gpu3-bigaux
  ...
```

This is the screen that lets you understand *why a run looks the way
it does* without reading the source code.

## 5. Migration of existing `firebase_push.py`

No rewrite required. The existing `/mamba3/*` paths stay. We add a
**second adapter** at `/experiments/three-populations-<date>/...` that
mirrors a curated subset of the GA event stream into the cross-experiment
schema:

| GA event | New schema mapping |
|---|---|
| `evt_mastery` | `events` with `type=milestone`, `details="task X mastered in Y steps"` |
| `evt_unlock` | `events` with `type=milestone` |
| `evt_evolve` | `events` with `type=variant_spawned` |
| `evt_pause` | `events` with `type=run_paused` |
| `evt_new_best` | `events` with `type=milestone` |
| `evt_error` | `events` with `type=error` |
| `push_snapshot` | `metrics` (downsampled to every 2 min from every gen) |
| `push_gpu_tick` | `status` (heartbeat) |

This is an *additive* mirror. Old dashboards keep reading `/mamba3/*`;
the new dashboard reads `/experiments/*`. Once the new path proves out
the old one can be retired.

## 6. What this doc deliberately does *not* cover

- The actual UI build. (Spec says "imagine the UI", not "build it".) When
  build time comes, this doc is the contract.
- Authentication / rules. Free-tier dashboard is read-only-public; tighten
  later if needed.
- Multi-experiment cross-comparison views (overlay byte_ce_biling from
  jepa-cortex onto a hanoi-exec run on the same chart, etc.). Easy
  follow-on once the per-experiment screens land.
- Exporting Firebase data back to local for offline analysis. The
  pusher's local JSONL fallback (§3) doubles as this; format below if
  needed.

## 7. Local-fallback log format

When Firebase is unreachable or for offline replay, the pusher writes
the same payloads to `runs/<run_id>/firebase_outbox.jsonl`:

```jsonl
{"path":"experiments/jepa-cortex-2026-04-29/runs/gpu0-ref/metrics","data":{"step":4000,"ts":1777500000,"byte_ce_biling":1.20,...}}
{"path":"experiments/jepa-cortex-2026-04-29/runs/gpu0-ref/heartbeat","data":{"step":4000,"sps":0.1,...}}
```

A separate `firebase_replay.py` (to be written) drains that file when the
network returns. Same format means we never lose data on a flaky
connection — exactly what the existing `firebase_push.py` does *not*
have today.
