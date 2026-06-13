# Reliability-First AI/ML Model Quality Framework

<!-- public-release-overview:start -->

[![CI](https://github.com/ReviveCoding/reliability-first-ai-ml-model-quality-framework/actions/workflows/ci.yml/badge.svg)](https://github.com/ReviveCoding/reliability-first-ai-ml-model-quality-framework/actions/workflows/ci.yml)
[![Release contract](https://github.com/ReviveCoding/reliability-first-ai-ml-model-quality-framework/actions/workflows/release-contract.yml/badge.svg)](https://github.com/ReviveCoding/reliability-first-ai-ml-model-quality-framework/actions/workflows/release-contract.yml)

A reliability-first framework for selecting, validating, and governing AI/ML models using leakage-safe evaluation, calibration, worst-slice analysis, multi-seed stability, drift and stress testing, auditable provenance, and launch gates.

## Current model decision

| Decision component | Outcome |
|---|---|
| Selected candidate | Unweighted `transformer_text_classifier` |
| Fallback | Calibrated `tfidf_logistic_temperature_scaled` |
| Class-weighted challenger | `REJECT_WEIGHTED` |
| Launch state | `REVIEW` |
| Decision principle | Aggregate improvement is insufficient without worst-slice and seed-stability support |

## Start here

- [Architecture and evaluation flow](docs/ARCHITECTURE.md)
- [Final model decision](docs/FINAL_MODEL_DECISION.md)
- [Experimental evidence](docs/EXPERIMENTAL_EVIDENCE.md)
- [Reproducibility](docs/REPRODUCIBILITY.md)
- [Public claim boundaries](docs/CLAIM_BOUNDARIES.md)

<!-- public-release-overview:end -->

A local-runnable framework that ingests public CFPB complaint data, validates source and modeling data, trains ML/NLP models, tracks experiments with MLflow, evaluates BM25/CrossEncoder evidence quality, stress-tests calibration and drift, replays synthetic GenAI telemetry, and issues a reproducible `PASS`, `REVIEW`, or `BLOCK` decision.

## Claim boundary

This repository uses public CFPB complaint data when supplied by the user and synthetic payment/complaint-style data for offline testing. It does not use Apple, private customer, private payment, or production monitoring data. Launch decisions are portfolio-style offline quality decisions, not regulatory or production approvals.

## Why this project is different

The framework separates model development decisions from final evidence:

```text
training window
    -> fit vocabulary, rare-label policy, models
calibration window
    -> fit temperature scaling only
selection window
    -> select the champion without touching test data
untouched temporal test window
    -> final quality, uncertainty, slice, and launch-gate evaluation
```

It also records source-schema fidelity, unseen future labels, bootstrap uncertainty, selection-to-test degradation, BM25 baseline versus CrossEncoder uplift, Monte Carlo sensitivity, provenance hashes, and artifact checksums.

## Core capabilities

- Public CSV, CSV.GZ, and optional 7z ingestion
- Automatic dataset discovery from `data/raw`
- DuckDB reservoir sampling for large CSV files, with exact pandas reservoir fallback
- Safe 7z member validation and optional persistent extraction cache
- Source-schema and normalized-data quality checks
- Training-only rare-label policy with explicit future-label novelty reporting
- Four-way temporal split with overlap, chronology, and class-coverage checks
- TF-IDF Logistic Regression and held-out temperature scaling
- Optional LightGBM and GPU Transformer text classification
- Selection-only champion choice and untouched test evaluation
- Macro-F1, ROC-AUC, PR-AUC, ECE, Brier score, log loss, slice F1, and bootstrap 95% CI
- BM25 baseline and optional batched CrossEncoder reranking with uplift metrics
- PSI, KS, missingness, categorical drift, and training-serving skew checks
- Synthetic GenAI telemetry, unsupported-claim proxy, latency, regression, and human-review routing
- Severity-aware launch gates
- MLflow tracking with a filesystem fallback
- Reproducible Monte Carlo stress validation
- Streamlit dashboard, pytest, Ruff, and GitHub Actions

## Quick start on Windows

```powershell
cd "<REPO_ROOT>"
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e . --no-deps
python -m pytest -q
```

The core public-data adapter currently targets the official CFPB complaint schema. FinQA, SEC, HMDA, FUNSD, and DocVQA are suitable extensions, but they are not silently treated as CFPB inputs.

The CPU core path needs only the repository, Python dependencies, and a supported CFPB file. The optional Transformer and CrossEncoder paths additionally require pretrained model weights, either downloaded from the configured model hub or supplied from a local cache/path.

Run the documented synthetic end-to-end configuration:

```powershell
python scripts\run_full_pipeline.py `
  --use-synthetic `
  --sample-size 800 `
  --enable-lightgbm `
  --random-state 20260612
```

### Dataset-only local run

Place exactly one supported file in `data/raw`:

```text
data/raw/complaints.csv
data/raw/complaints.csv.gz
data/raw/complaints.csv.7z
```

Then validate the dataset before training:

```powershell
python scripts\dataset_preflight.py --archive-cache-dir data\archive_cache
```

Run the complete audit:

```powershell
python scripts\run_dataset_audit.py `
  --cfpb-path "data\raw\complaints.csv.7z" `
  --archive-cache-dir "data\archive_cache" `
  --sample-size 1000 `
  --sampling-strategy auto `
  --enable-lightgbm
```

Or run the pipeline directly. When `--cfpb-path` is omitted, exactly one supported file is auto-discovered from `<data-dir>/raw`:

```powershell
python scripts\run_full_pipeline.py `
  --data-dir data `
  --sample-size 1000 `
  --sampling-strategy auto `
  --enable-lightgbm
```

Run with an explicit public CFPB CSV:

```powershell
python scripts\run_full_pipeline.py `
  --cfpb-path "C:\path\to\complaints.csv" `
  --sample-size 50000 `
  --sampling-strategy auto `
  --enable-lightgbm
```

For a large 7z archive, use a persistent cache so decompression is not repeated:

```powershell
python scripts\run_full_pipeline.py `
  --cfpb-path "C:\path\to\complaints.csv.7z" `
  --archive-cache-dir "C:\path\to\cfpb_cache" `
  --sample-size 50000 `
  --sampling-strategy auto `
  --enable-lightgbm
```

The first 7z run must still decompress the large CSV. Pre-extracting the CSV is preferable when disk space permits. For CSV files larger than 100 MB, `auto` uses DuckDB's deterministic reservoir sampler. Use `reservoir` only when exact pandas streaming behavior is specifically required; it is much slower for multi-gigabyte narrative files.

On machines with many CPU cores, the scripts default BLAS/OpenMP libraries to four threads to avoid oversubscription. Override this before launching when appropriate:

```powershell
$env:MODEL_QUALITY_CPU_THREADS="8"
```

## Optional GPU path

Install a CUDA-compatible PyTorch build for the local driver, then install the optional stack:

```powershell
pip install -r requirements-gpu.txt
python scripts\gpu_preflight.py --require-cuda
```

```powershell
python scripts\run_full_pipeline.py `
  --use-synthetic `
  --sample-size 800 `
  --enable-lightgbm `
  --enable-transformer `
  --transformer-model distilbert-base-uncased `
  --transformer-epochs 2 `
  --transformer-batch-size 32 `
  --enable-cross-encoder `
  --cross-encoder-batch-size 64 `
  --device auto
```

The core models run on CPU. Transformer and CrossEncoder modules validate CUDA, MPS, or CPU and fail safely when an optional dependency or device is unavailable.

## Pipeline

```text
Public or synthetic complaint data
        -> source-schema and data-quality validation
        -> temporal train / calibration / selection / test split
        -> training-only label policy
        -> Logistic / calibrated Logistic / optional LightGBM / Transformer
        -> MLflow or filesystem experiment tracking
        -> selection-window champion choice
        -> untouched test evaluation + bootstrap uncertainty
        -> BM25 baseline + optional CrossEncoder reranking
        -> drift and training-serving-skew stress tests
        -> synthetic GenAI telemetry and review routing
        -> severity-aware PASS / REVIEW / BLOCK gate
        -> reports, cards, dashboard, manifests, and audit evidence
```

## Validated root result

The included 800-row nominal run produced:

| Metric | Result |
|---|---:|
| Launch decision | PASS |
| Champion | `tfidf_logistic_temperature_scaled` |
| Selection macro-F1 | 0.8162 |
| Untouched-test macro-F1 | 0.9138 |
| Macro-F1 95% bootstrap CI | [0.8598, 0.9589] |
| Test PR-AUC | 0.9719 |
| ECE | 0.0828 |
| Brier score | 0.1235 |
| Log loss | 0.2855 |
| Worst-slice F1 | 0.7981 |
| Evidence Recall@5 | 0.9750 |
| Context precision | 0.5717 |
| Max PSI | 0.0100 |
| Task completion | 0.9833 |

## Monte Carlo validation

Run 18 full pipelines across nominal, moderate, and severe conditions:

```powershell
python scripts\run_monte_carlo.py `
  --runs-per-scenario 6 `
  --sample-size 800 `
  --enable-lightgbm `
  --jobs 3
```

Included result:

| Scenario | PASS | REVIEW | BLOCK | Mean macro-F1 | Mean log loss | Mean max PSI | Mean task completion |
|---|---:|---:|---:|---:|---:|---:|---:|
| nominal | 100.0% | 0.0% | 0.0% | 0.9029 | 0.3398 | 0.0381 | 0.9639 |
| moderate | 0.0% | 66.7% | 33.3% | 0.6925 | 0.8664 | 0.3391 | 0.8806 |
| severe | 0.0% | 0.0% | 100.0% | 0.4118 | 1.4514 | 1.2441 | 0.1708 |

Champion selection was stable in all nominal and moderate runs. The temperature-scaled model was selected in 5 of 6 severe runs, while the unscaled Logistic model was selected once.

## Validation commands

```powershell
python -m ruff check .
python -m pytest -q
python scripts\project_audit.py --run-smoke --run-monte-carlo-smoke --run-gpu-preflight
```

Real-data `PASS`, `REVIEW`, or `BLOCK` is a quality outcome, not a process-success code. `run_dataset_audit.py` passes when ingestion, training, evaluation, and required artifact generation complete; it records the gate decision separately.

Useful Make targets:

```text
make lint
make test
make smoke
make audit-full
make monte-carlo
make gpu-preflight
```

## Main artifacts

```text
outputs/model_selection_leaderboard.csv
outputs/model_test_leaderboard.csv
outputs/calibration_summary.csv
outputs/evidence_quality_summary.json
outputs/drift_alerts.csv
outputs/human_review_queue.csv
outputs/launch_gate_result.json
outputs/pipeline_manifest.json
reports/executive_summary.md
reports/model_card.md
reports/data_card.md
reports/launch_decision_memo.md
monte_carlo/outputs/monte_carlo_scenario_summary.csv
monte_carlo/outputs/monte_carlo_champion_distribution.csv
monte_carlo/reports/monte_carlo_validation_report.md
```

## Repository boundaries

The repository intentionally excludes generated raw/intermediate data, model binaries, local MLflow runs, and Monte Carlo per-run workspaces. Summary CSV, JSON, Markdown evidence, configuration, tests, and manifests remain versionable.

See:

- `STRENGTHENING_LOOP_REPORT.md`
- `SMALL_DATA_VALIDATION.md`
- `MONTE_CARLO_VALIDATION.md`
- `LOCAL_GITHUB_RUNNABLE_AUDIT.md`
- `DATASET_RUNNABILITY_VALIDATION.md`

<!-- FINAL_MODEL_DECISION_START -->
## Final model decision

The retained configuration is the **unweighted Transformer** with a
**temperature-scaled TF-IDF Logistic fallback**. A three-seed
sqrt-balanced class-weighted challenger improved average metrics but was
rejected because it did not improve worst-slice reliability
consistently.

- [Final model decision](docs/FINAL_MODEL_DECISION.md)
- [Experimental evidence](docs/EXPERIMENTAL_EVIDENCE.md)
- [Reproducibility](docs/REPRODUCIBILITY.md)
- [GitHub release checklist](docs/GITHUB_RELEASE_CHECKLIST.md)
- [Canonical evaluation artifacts](docs/artifacts/final_evaluation)

Current launch state: **REVIEW**. No model is represented as an approved
production champion.
<!-- FINAL_MODEL_DECISION_END -->
