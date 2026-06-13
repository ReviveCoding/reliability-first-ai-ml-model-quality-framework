# Local and GitHub Runnable Audit

For the final real-dataset verification, see `DATASET_RUNNABILITY_VALIDATION.md`.

## Final v7 status

```text
Ruff static checks           PASS
Python compileall            PASS
Package import               PASS
Pytest                       45 passed
800-row nominal pipeline     PASS
Monte Carlo                  18/18, 0 failed, sensitivity PASS
Public-schema stable path    PASS
Future unseen-label path     BLOCK as designed
Real CFPB 7z preflight       PASS
Real CFPB 7z pipeline        PASS (quality gate BLOCK)
Automatic data/raw discovery PASS
```

## Repository execution contract

The tracked repository contains source, configuration, tests, lightweight reports, and summary evidence. It excludes raw data, generated feature tables, model binaries, local MLflow runs, and Monte Carlo per-run workspaces.

## Dependency layers

```text
requirements-core.txt   deterministic CPU framework
requirements-ci.txt     LightGBM, pytest, Ruff for GitHub Actions
requirements.txt        local CPU, MLflow, Streamlit, archive support
requirements-gpu.txt    optional Torch, Transformers, Datasets, Sentence Transformers
```

## Local CPU audit

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e . --no-deps
python -m ruff check .
python -m pytest -q
python scripts\project_audit.py --run-smoke --run-monte-carlo-smoke --run-gpu-preflight
```

## Public data

```powershell
python scripts\run_full_pipeline.py `
  --cfpb-path "C:\path\to\complaints.csv" `
  --sample-size 50000 `
  --sampling-strategy reservoir `
  --enable-lightgbm
```

For 7z:

```powershell
python scripts\run_full_pipeline.py `
  --cfpb-path "C:\path\to\complaints.csv.7z" `
  --archive-cache-dir "C:\path\to\cfpb_cache" `
  --sample-size 50000 `
  --sampling-strategy reservoir `
  --enable-lightgbm
```

The first 7z run still incurs one-time decompression. Subsequent runs can reuse the validated cached CSV.

## GPU audit

```powershell
pip install -r requirements-gpu.txt
python scripts\gpu_preflight.py --require-cuda
python scripts\run_full_pipeline.py `
  --use-synthetic `
  --sample-size 800 `
  --enable-lightgbm `
  --enable-transformer `
  --enable-cross-encoder `
  --device auto
```

The current build environment is CPU-only. GPU-aware code, device validation, safe fallback, batching, and preflight are tested, but CUDA throughput, VRAM, Transformer accuracy, and CrossEncoder uplift must be measured on the user's local GPU.

## GitHub Actions

The workflow runs on Python 3.10, 3.11, and 3.12 and performs:

1. dependency installation and `pip check`,
2. editable package installation,
3. Ruff static checks,
4. all tests,
5. an 800-row nominal pipeline on Python 3.11,
6. a nominal-versus-severe Monte Carlo sensitivity smoke.

## Provenance

`outputs/pipeline_manifest.json` records:

- runtime and environment versions,
- source provenance without exposing absolute local paths,
- selection protocol,
- optional-feature status,
- configuration hashes,
- artifact sizes and SHA-256 checksums.

Monte Carlo manifests additionally include the full executable code-tree hash and reject incompatible resume artifacts.

## Claim boundary

Local/GitHub runnable does not mean production deployed. This repository demonstrates offline public/proxy/synthetic model-quality engineering and launch-decision logic. It does not claim production traffic, private financial data, Apple data, regulatory approval, or operational SLA ownership.
