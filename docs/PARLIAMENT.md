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

Claude Code's `--output-format json` wrapper is accepted as long as its
`result` field contains a speech JSON object.

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

Real mixed chamber:

```bash
.venv/bin/python tools/parliament.py chamber \
  --speakers all \
  --backend auto \
  --motion parliament/motions/example_chess_kpi.md \
  --dry-run \
  --timeout-s 300
```

With `--backend auto`, identities whose `model_family` contains `Claude` use
Claude Code. Other identities use Codex in a read-only sandbox.

Use `--speakers all` to invite every current owner identity.

## Chapter Owners

The initial Parliament identities are domain owners, not generic personas:

| Speaker | Scope |
|---|---|
| `gpt5-ch12-chess-champion` | Chapter 12 chess experts and playable champion |
| `claude-hanoi-lego-puzzle-solver` | Chapters 04-05: Hanoi, LoopCounter, Lego specialists |
| `claude-cortex-primitive-owner` | Chapters 06, 08, 09: cortex primitives, RLF, bilingual counter attach |
| `claude-language-jepa-owner` | Chapters 07, 10, 13: JEPA/language-model line and autopilot work |
| `claude-phi-composition-owner` | Chapter 11: frozen Phi cold composition and solver registry |
| `claude-platform-kappa-clerk` | Platform, Kappa, Firebase/Hugging Face archive, Lab Book |
| `claude-opposition-architect` | General opposition and architecture critique |

Silence policy:

```bash
.venv/bin/python tools/parliament.py event \
  --event '{"type":"heartbeat","status":"training"}'
```

Expected decision: `silent`.

Register this machine as a Parliament node:

```bash
.venv/bin/python tools/parliament.py register-node
.venv/bin/python tools/parliament.py nodes
```

Keep a node online:

```bash
.venv/bin/python tools/parliament.py register-node --watch --interval-s 30
```

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
