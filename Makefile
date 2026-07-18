.PHONY: install test lint benchmark smoke

install:
	python -m pip install -e ".[dev]"

test:
	pytest

lint:
	ruff check .

benchmark:
	python scripts/run_benchmark.py --config configs/benchmark.yaml

smoke:
	python scripts/run_benchmark.py --config configs/benchmark.yaml --smoke
