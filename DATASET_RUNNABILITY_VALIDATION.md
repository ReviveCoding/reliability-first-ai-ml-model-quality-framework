# Dataset-Only Local/GitHub Runnability Validation

## Scope

This validation answers a narrow question: after cloning or extracting the repository and installing its declared dependencies, does a user only need to provide a supported public dataset for the end-to-end CPU framework to run?

The currently wired public-data adapter is the CFPB Consumer Complaint schema. Other suggested datasets such as FinQA, SEC, HMDA, FUNSD, and DocVQA require dedicated adapters and are not silently interpreted as CFPB files.

## Clean-environment checks

A fresh virtual environment was created and the repository was installed in editable mode.

```text
pip check: PASS
Ruff: PASS
Pytest: 45 passed
```

## Real public dataset used

```text
CFPB complaints 20220504-20260514 complaints with narratives.csv.7z
Compressed size: 189,815,959 bytes
Extracted CSV size: 1,882,740,319 bytes
```

The archive contained one CSV member and passed safe-member-path validation.

## Real-data preflight

```text
Preflight status: PASS
Usable sample rows: 100
Required source columns missing: none
Narrative nonempty rate: 1.0
Unique products in preflight sample: 9
Unique issues in preflight sample: 20
```

## Real-data end-to-end audit

Command shape:

```powershell
python scripts\run_dataset_audit.py `
  --cfpb-path "C:\path\to\complaints.csv.7z" `
  --archive-cache-dir "data\archive_cache" `
  --sample-size 1000 `
  --sampling-strategy auto `
  --enable-lightgbm
```

Observed result after archive caching:

```text
Preflight return code: 0
Pipeline return code: 0
Required outputs present: true
Effective sampling strategy: duckdb
Pipeline runtime: 18.6791 seconds
Audit runtime: 21.70 seconds
Peak RSS: approximately 2.55 GB
Launch decision: BLOCK
Dataset audit status: PASS
```

`BLOCK` is a model-quality outcome, not a software failure. The public sample failed calibration, worst-slice, and evidence-precision thresholds, while ingestion, training, evaluation, drift, telemetry, gate generation, reports, and manifests all completed correctly.

## Automatic discovery

A supported file was placed as the only dataset under a custom `<data-dir>/raw` directory. The pipeline was then run without `--cfpb-path`.

```text
Source type: public_cfpb
Auto-discovered source file: complaints.csv.7z
Requested sampler: auto
Effective sampler: duckdb
Pipeline return code: 0
Required artifacts generated: yes
```

The framework now raises a clear error when neither `--use-synthetic` nor a unique supported dataset is available. It no longer silently falls back to synthetic input.

## Large-file sampling

The previous pandas reservoir path required a full Python-level scan of the 1.88 GB narrative CSV and was impractically slow. The `auto` strategy now selects DuckDB for files of at least 100 MB and retains pandas reservoir sampling as an exact fallback for smaller files or explicit use.

The actual 1.88 GB CSV was reservoir-sampled to 1,000 usable rows through DuckDB and then passed through the complete framework.

## CPU thread behavior

High-core environments can become much slower because BLAS/OpenMP libraries oversubscribe threads. The pipeline now defaults these libraries to one thread. Users can override the setting explicitly:

```powershell
$env:MODEL_QUALITY_CPU_THREADS="8"
```

## GitHub behavior

GitHub Actions validates:

- dependency installation and `pip check`
- Ruff
- unit/integration tests
- synthetic end-to-end smoke
- generated CFPB public-schema CSV ingestion
- Monte Carlo nominal-versus-severe sensitivity

A large real dataset is intentionally not committed to the repository. A real-data GitHub runner must download or mount the dataset before invoking the same dataset-audit command.

## Optional-feature boundary

The following CPU framework path is dataset-only after dependency installation:

```text
CFPB ingestion -> validation -> temporal splits -> Logistic/LightGBM ->
calibration -> evaluation -> evidence baseline -> drift -> telemetry ->
PASS/REVIEW/BLOCK -> reports/manifests
```

The following optional paths additionally need pretrained model weights:

```text
Transformer classifier
CrossEncoder reranker
```

Weights may be downloaded from the configured model source or supplied through a local cache/model path. Therefore, a dataset alone is sufficient for the CPU core, but not for offline first-time execution of optional pretrained-model modules unless their weights are already available locally.

## Final verdict

```text
Local CPU core with supported CFPB dataset: VERIFIED RUNNABLE
Automatic data/raw discovery: VERIFIED RUNNABLE
CSV/CSV.GZ/7z input contract: VERIFIED
Large 7z with cached extraction + DuckDB sampler: VERIFIED RUNNABLE
Required reports and manifests: VERIFIED
GitHub CI without large dataset: VERIFIED BY FIXTURE/SYNTHETIC WORKFLOW
Optional GPU/Transformer/CrossEncoder with dataset alone: NOT SUFFICIENT WITHOUT MODEL WEIGHTS
Other public datasets without adapters: NOT YET DIRECTLY RUNNABLE
```
