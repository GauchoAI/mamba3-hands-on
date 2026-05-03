---
title: Parliament Protocol
summary: "Multi-model deliberation protocol for motions, checkpoint reviews, and evidence-bearing speeches."
---

# Parliament Protocol

Parliament is a thin coordination layer. It does not replace Firebase,
Kappa, Hugging Face, the node agent, or the lab book.

## Roles

- **Firebase** is the hot coordination plane: motions, claims, heartbeats,
  short speeches, and event notifications.
- **Kappa / Hugging Face** is the durable archive: long traces, artifacts,
  checkpoints, and full deliberation bundles.
- **Model backends** are replaceable speakers: Codex/Symphony, Claude, local
  command adapters, or dry-run simulators.
- **The clerk** attaches metadata. Models do not hand-write identity
  front matter every time.

## Firebase Paths

```text
/parliament/nodes/{node_id}
/parliament/motions/{motion_id}
/parliament/claims/{motion_id}/{speaker_id}
/parliament/speeches/{motion_id}/{speaker_id}/{auto_id}
/parliament/events/{auto_id}
```

## Speech

Models write only the content body:

```json
{
  "kind": "position",
  "position": "amend",
  "body": "The checkpoint is promising, but the next benchmark must target repeated checks.",
  "evidence": ["experiments/12_chess_experts/findings.md"],
  "prediction": "A stronger KPI will expose queen-check loops.",
  "falsifier": "If adversarial play shows no repeated-check failure, this critique weakens.",
  "confidence": 0.72
}
```

The appender wraps it:

```json
{
  "schema": "parliament.speech.v1",
  "motion_id": "example-chess-kpi",
  "speaker": "gpt5-ch12-chess-champion",
  "identity_ref": "parliament/identities/gpt5-ch12-chess-champion.yaml",
  "node_id": "host-mps",
  "dry_run": true,
  "created_at": "2026-05-03T12:00:00Z",
  "speech": {}
}
```

## Silence Rule

Parliament should not speak merely because a long job is still running.
A new deliberation is warranted when there is evidence to inspect:

- checkpoint produced
- KPI changed
- benchmark failed or passed
- human playtest submitted
- publication requested
- explicit human motion opened
- scheduled retrospective reached

Running/training/in-progress heartbeats without a checkpoint should produce
`position=defer` or no session.

## Dry Runs

Dry runs write a trace under `runs/parliament/dry_runs/` and print wrapped
speech records to stdout. They do not write Firebase, Hugging Face, commits,
or the durable `parliament/log.jsonl`.
