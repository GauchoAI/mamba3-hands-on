---
title: Parliament
summary: "Plan and operating guide for multi-model deliberation over lab motions, checkpoints, and reviews."
---

# Parliament

Parliament gives models a voice in the lab without asking them to pretend they
remember the lab. The repository invokes each speaker with identity, evidence,
procedure, and a motion. The model then inspects the project, reasons for a
bounded run, and emits a structured speech.

## Plan

1. **Protocol first.** Keep identity, motions, speeches, and event policy in a
   small repo-native format.
2. **Dry runs first.** A dry run writes only an inspectable trace under
   `runs/parliament/dry_runs/` and prints the speech. It does not write
   Firebase, Hugging Face, commits, or `parliament/log.jsonl`.
3. **Backend adapters.** Codex/Symphony, Claude, and local tools all satisfy
   the same contract: receive a prompt, inspect the repo, return speech JSON.
4. **Firebase for live coordination.** Motions, claims, node heartbeats, and
   short speeches live under `/parliament/*`.
5. **Kappa/Hugging Face for durable archive.** Long traces and artifacts should
   be sealed outside Firebase, matching the existing Kappa convention.
6. **Silence while work is merely running.** Training heartbeats do not trigger
   speeches. Parliament speaks when there is a checkpoint, KPI movement,
   benchmark result, human playtest, publication request, explicit motion, or
   scheduled retrospective.

## Compatibility

Parliament is compatible with Claude because Claude is not special-cased in the
protocol. It is a speaker identity plus a backend command. The same is true for
Codex, Symphony, local subprocesses, and future agents.

For example:

```bash
PARLIAMENT_BACKEND_CLAUDE_CMD='claude --json' \
.venv/bin/python tools/parliament.py speak \
  --speaker claude-opposition-architect \
  --backend claude \
  --motion parliament/motions/example_chess_kpi.md \
  --dry-run
```

The command must read the full prompt from stdin and return one JSON speech
object to stdout.

## Dry-Run Examples

One speaker:

```bash
.venv/bin/python tools/parliament.py speak \
  --speaker gpt5-ch12-chess-champion \
  --motion parliament/motions/example_chess_kpi.md \
  --dry-run
```

Two-speaker chamber:

```bash
.venv/bin/python tools/parliament.py chamber \
  --speakers gpt5-ch12-chess-champion claude-opposition-architect \
  --motion parliament/motions/example_chess_kpi.md \
  --dry-run
```

Silence policy:

```bash
.venv/bin/python tools/parliament.py event \
  --event '{"type":"heartbeat","status":"training"}'
```

Expected decision: `silent`.

## Five-Minute Iteration Rule

A Parliament run should fit the lab's normal rhythm. Even long training should
produce bounded inspectable moments:

- one-minute smoke run
- checkpoint or trace
- metric/KPI update
- Parliament review only if there is evidence
- next iteration

This keeps model speech attached to concrete artifacts instead of ambient
commentary.
