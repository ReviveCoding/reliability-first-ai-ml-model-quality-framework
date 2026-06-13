.PHONY: install install-core install-gpu lint test smoke audit audit-full monte-carlo monte-carlo-smoke dashboard mlflow-ui gpu-preflight dataset-preflight dataset-audit

install:
	python -m pip install --upgrade pip
	pip install -r requirements.txt
	pip install -e . --no-deps

install-core:
	python -m pip install --upgrade pip
	pip install -r requirements-core.txt
	pip install -e . --no-deps

install-gpu:
	python -m pip install --upgrade pip
	pip install -r requirements.txt
	pip install -r requirements-gpu.txt
	pip install -e . --no-deps

lint:
	python -m ruff check .

test:
	python -m pytest -q

audit:
	python scripts/project_audit.py --run-smoke

audit-full:
	python scripts/project_audit.py --run-smoke --run-monte-carlo-smoke

smoke:
	python scripts/run_full_pipeline.py --use-synthetic --sample-size 800 --enable-lightgbm

dashboard:
	streamlit run dashboard/app.py

mlflow-ui:
	mlflow ui --backend-store-uri mlruns

monte-carlo:
	python scripts/run_monte_carlo.py --runs-per-scenario 6 --sample-size 800 --enable-lightgbm --jobs 3

monte-carlo-smoke:
	python scripts/run_monte_carlo.py --scenarios nominal severe --runs-per-scenario 1 --sample-size 400 --jobs 2 --output-root monte_carlo_smoke

gpu-preflight:
	python scripts/gpu_preflight.py

dataset-preflight:
	python scripts/dataset_preflight.py --archive-cache-dir data/archive_cache

dataset-audit:
	python scripts/run_dataset_audit.py --cfpb-path "$(DATASET)" --archive-cache-dir data/archive_cache --enable-lightgbm
