# ptxd scheduler telemetry — wire format

The slot scheduler emits compact `tick` events at most once per second
(default; configurable via `Scheduler::tick_interval_s`). When piped
through `ptxd_tail.py`, ticks are forwarded to Firebase Realtime DB
under `mamba3/scheduler_history/{generation}` — each tick is appended
as a sortable child via POST.

## Tick wire format

One tick, after `_post` to Firebase:

```json
{
  "t":       17.32,         // seconds since the scheduler started
  "mem":     0.0,           // percent of mem_budget used (0..100)
  "sm":      57.0,          // percent of sm_budget used  (0..100)
  "running": 4,             // active slots
  "queue":   3              // pending jobs
}
```

5 fields × ~10 bytes = ~50-byte body. Firebase POST adds a small overhead
(headers, auto-key) for ~200 bytes total per tick. At 1Hz that's
~12 KB/min, ~1 MB/hr per active scheduler — fits any free tier.

## Ingest pipeline

```
   ptxd ─stdout(jsonl)─▶  ptxd_tail.py  ─Firebase POST─▶  RTDB
                              │
                              └─▶ stdout (cycle/final events pass through)
```

`ptxd_tail.py`:
- Reads JSONL from stdin (or runs `--cmd` and tails its stdout).
- Buffers `tick` events; flushes every `--flush-interval` seconds (default 5).
- Forwards `cycle` and `final` events to stdout unchanged (so consumers
  like `ptxd_specialist.py` keep working).
- Falls through silently if `firebase_push` isn't importable.

```bash
./ptxd --concurrent 4 < jobs.jsonl | python3 ptxd_tail.py --gen 42
```

## UI retrieval

Firebase RTDB REST query for the last hour of ticks for generation 42:

```
GET https://signaling-dcfad-default-rtdb.europe-west1.firebasedatabase.app/mamba3/scheduler_history/42.json
    ?orderBy="t"&startAt=1234560000
```

Or via the Firebase JS SDK:

```js
import { ref, onValue, query, orderByChild } from "firebase/database";
const q = query(ref(db, `mamba3/scheduler_history/${gen}`), orderByChild("t"));
onValue(q, (snap) => {
  const ticks = [];
  snap.forEach(c => ticks.push(c.val()));
  // ticks: [{t, mem, sm, running, queue}, ...] sorted by t
  drawSparkline(ticks);
});
```

## What to plot

A useful dashboard panel for one ptxd generation:

- **Line / area:** `sm` over `t` — GPU utilisation over time
- **Line:** `running` over `t` — slot fill-level
- **Line:** `queue` over `t` — backlog depth
- **Single number:** latest `mem` — memory headroom

For the Tetris-pack-as-it-evolves visualisation, sample the latest
tick: `(running, queue, mem, sm)` updates the bar widths, no history
needed.

## Cadence / cost summary

| | Value |
|---|---|
| Tick interval (default) | 1.0s |
| Bytes per tick (wire) | ~50 |
| HTTP requests per minute (per ptxd) | ~60 |
| Firebase RTDB tier needed | free is fine |
| Latency to UI | ~1s (RTDB onValue listener) |

## Pruning

Each generation accumulates ~3600 tick rows per hour. The orchestrator
should periodically prune old generations, e.g. delete
`mamba3/scheduler_history/{gen}` once the corresponding GA round is
finalised and the lineage is committed elsewhere. The existing `pruning
hooks` in `firebase_push.py` are the right place — pattern after
whatever's already done for `gpu_history`.
