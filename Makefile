PYTHON ?= .venv/bin/python

install:
	$(PYTHON) -m pip install -e '.[dev]'

test:
	$(PYTHON) -m pytest -q

doctor:
	$(PYTHON) -m transformer_go.cli doctor

smoke:
	$(PYTHON) -m transformer_go.cli loop --config configs/smoke.yaml --output runs/smoke-loop --iterations 1 --games 2 --steps 3 --eval-games 2
