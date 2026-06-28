.PHONY: setup data predict labels features monitor baselines eval figures all test lint clean

PYTHON ?= python
SEED   ?= 42

setup:
	uv sync --all-extras
	$(PYTHON) -m pip install -e .
	pre-commit install

data:
	$(PYTHON) scripts/00_download_data.py seed=$(SEED)
	$(PYTHON) scripts/01_build_manifests.py seed=$(SEED)

predict:
	$(PYTHON) scripts/03_run_predictions.py seed=$(SEED)

labels:
	$(PYTHON) scripts/04_make_failure_labels.py seed=$(SEED)

features:
	$(PYTHON) scripts/05_extract_features.py seed=$(SEED)

monitor:
	$(PYTHON) scripts/06_train_monitor.py seed=$(SEED)
	$(PYTHON) scripts/07_calibrate_conformal.py seed=$(SEED)

baselines:
	$(PYTHON) scripts/08_run_baselines.py seed=$(SEED)

eval:
	$(PYTHON) scripts/09_evaluate.py seed=$(SEED)

figures:
	$(PYTHON) scripts/10_make_figures.py seed=$(SEED)

all: data predict labels features monitor baselines eval figures

test:
	$(PYTHON) -m pytest tests/ -v

lint:
	ruff check src scripts tests
	black --check src scripts tests

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
	rm -rf .pytest_cache htmlcov .coverage
