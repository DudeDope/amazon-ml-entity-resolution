.PHONY: install install-gpu test lint smoke doctor

install:
	python -m pip install -e ".[dev]"

install-gpu:
	python -m pip install -e ".[dev,gpu]"

test:
	python -m pytest

lint:
	python -m ruff check src tests

smoke:
	python -m amazon_er.cli smoke --config configs/smoke.yaml

doctor:
	python -m amazon_er.cli doctor --config configs/smoke.yaml

