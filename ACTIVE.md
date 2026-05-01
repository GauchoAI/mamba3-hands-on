# Active Route

The current daily-driver experiment is:

```text
experiments/11_stack_operator_transfer/
```

This is the start of the supported active research route after the long-training
pause. Chapters 12-21 continue the same one-minute operator curriculum. The old
structured-data JEPA route remains in `experiments/10_jepa_structured/` for
reference and smoke coverage, but it is not the current daily driver.

## Commands

Use the project virtualenv:

```bash
.venv/bin/python -m pip install -e .
```

The Makefile also exports `PYTHONPATH=src`, so the commands below work from a
source checkout before installation.
Editable install exposes the `lab` CLI and utility commands such as
`lab-kappa-pack` and `lab-book`.

```bash
.venv/bin/python experiments/11_stack_operator_transfer/stack_operator.py
```

Fast one-minute smoke:

```bash
.venv/bin/python experiments/11_stack_operator_transfer/stack_operator.py \
  --epochs 150 --trials 20
```

Run the full one-minute operator curriculum:

```bash
.venv/bin/python experiments/12_raw_trace_stack/raw_trace_stack.py
.venv/bin/python experiments/13_multi_surface_stack/multi_surface_stack.py
.venv/bin/python experiments/14_operator_guided_decoding/operator_guided_decoding.py
.venv/bin/python experiments/15_comparator_transfer/comparator_transfer.py
.venv/bin/python experiments/16_trace_to_operator_search/trace_to_operator_search.py
.venv/bin/python experiments/17_operator_registry/operator_registry.py
.venv/bin/python experiments/18_operator_composition/operator_composition.py
.venv/bin/python experiments/19_language_to_role_trace/language_to_role_trace.py
.venv/bin/python experiments/20_runtime_learning_episode/runtime_learning_episode.py
.venv/bin/python experiments/21_lab_organ_demo/lab_organ_demo.py
```

The previous JEPA route can still be inspected:

```bash
.venv/bin/python experiments/10_jepa_structured/orchestrator.py status
```

Open the chapter-style dashboard:

```bash
lab book
```

## Project Layout

- `experiments/` holds numbered research chapters.
- `src/lab_platform/` holds shared platform infrastructure used across chapters.
- `tools/` holds dashboards, diagnostics, cluster helpers, and one-off utilities.
- `docs/` holds architecture, findings, plans, and legacy context.
- `data/`, `checkpoints/`, `runs/`, logs, and pid files are runtime artifacts and
  are ignored unless explicitly whitelisted.

## Reorg Note

Before the cleanup, the active chapter had stale defaults pointing at
`experiments/jepa_structured_data/`. Active scripts now derive their default
paths from their own file location.
