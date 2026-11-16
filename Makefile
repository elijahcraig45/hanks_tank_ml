# Reproduce the NFL model comparison offline (see demo/README.md).
# The first run creates .venv-demo and installs the pinned packages in requirements-demo.txt.

PYTHON ?= python3
VENV   := .venv-demo
PY     := $(VENV)/bin/python

.PHONY: demo demo-weekly demo-clean

demo: $(VENV)/.installed
	$(PY) demo/nfl_backtest.py

demo-weekly: $(VENV)/.installed
	$(PY) demo/nfl_backtest.py --weekly

$(VENV)/.installed: requirements-demo.txt
	$(PYTHON) -m venv $(VENV)
	$(PY) -m pip install -q --upgrade pip
	$(PY) -m pip install -q -r requirements-demo.txt
	touch $@

demo-clean:
	rm -rf $(VENV)
