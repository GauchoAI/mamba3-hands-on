PYTHON ?= .venv/bin/python
ACTIVE_EXP ?= experiments/10_jepa_structured
ORCH := $(ACTIVE_EXP)/orchestrator.py
export PYTHONPATH := src:$(PYTHONPATH)

.PHONY: active-status active-gen active-train active-smoke

active-status:
	$(PYTHON) $(ORCH) status

active-gen:
	AWS_PROFILE=cc $(PYTHON) $(ORCH) gen --daemon --order topo

active-train:
	$(PYTHON) $(ORCH) train --continuous --order topo

active-smoke:
	$(PYTHON) $(ORCH) train --order topo --limit 1 --steps-per-tile 1 --batch-size 1
