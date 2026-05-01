# Active Route

The current daily-driver experiment is:

```text
experiments/10_jepa_structured/
```

This is the only supported active route. Top-level numbered experiment
directories are intentionally not duplicated.

## Commands

Use the project virtualenv:

```bash
.venv/bin/python experiments/10_jepa_structured/orchestrator.py status
```

Generate curriculum examples:

```bash
AWS_PROFILE=cc .venv/bin/python experiments/10_jepa_structured/orchestrator.py gen --daemon --order topo
```

Train continuously:

```bash
.venv/bin/python experiments/10_jepa_structured/orchestrator.py train --continuous --order topo
```

Run a bounded training smoke pass:

```bash
.venv/bin/python experiments/10_jepa_structured/orchestrator.py train --order topo --limit 1 --steps-per-tile 1 --batch-size 1
```

## Project Layout

- `experiments/` holds numbered research chapters.
- Root Python modules are shared platform infrastructure used across chapters.
- `tools/` holds dashboards, diagnostics, cluster helpers, and one-off utilities.
- `docs/` holds architecture, findings, plans, and legacy context.
- `data/`, `checkpoints/`, `runs/`, logs, and pid files are runtime artifacts and
  are ignored unless explicitly whitelisted.

## Reorg Note

Before the cleanup, the active chapter had stale defaults pointing at
`experiments/jepa_structured_data/`. Active scripts now derive their default
paths from their own file location.
