# Weakness Analysis and Strengthening Loop Report

## Final status

The loop stopped after the remaining changes became deployment-scope expansion or local-GPU benchmarking rather than material framework corrections.

```text
Ruff: PASS
Tests: 40 passed
Synthetic nominal pipeline: PASS
Monte Carlo: 18/18 completed, 0 failed, sensitivity PASS
Public-schema stable path: PASS
Future unseen-label path: BLOCK as designed
```

## Loop 1: Remove temporal label leakage

### Weakness

Rare labels were consolidated using counts from the full dataset. This allowed future calibration, selection, and test distributions to influence the training taxonomy.

### Fix

- Fit the label policy on the training window only.
- Apply the frozen policy to all later windows.
- Preserve genuinely unseen later labels instead of mapping them to `Other`.
- Record novel labels and novel-target rates by split.

### Validation

A future-only `Crypto service` class remained unseen, produced a 100% unknown-target rate in test, and correctly forced `BLOCK`.

## Loop 2: Remove champion-selection leakage

### Weakness

Selecting the champion from final test performance produces an optimistic sign-off estimate.

### Fix

The temporal protocol is now:

```text
train -> calibration -> selection -> untouched test
```

- training fits features and models,
- calibration fits temperature only,
- selection chooses the champion,
- test supplies final evidence only.

The gate also tracks the selection-to-test macro-F1 gap.

### Validation

The nominal 800-row run selected the calibrated Logistic model at selection macro-F1 0.8162, then reported untouched-test macro-F1 0.9138 without re-selection.

## Loop 3: Add uncertainty-aware sign-off

### Weakness

Point estimates alone can pass a small or unstable test set.

### Fix

- Added fixed-label bootstrap macro-F1 confidence intervals.
- Added minimum evaluated-row coverage.
- Added lower-CI and selection/test-gap thresholds.

### Validation

The nominal test macro-F1 was 0.9138 with a 95% CI of [0.8598, 0.9589]. A future-only unseen-class test produced zero evaluable known-label rows and was blocked rather than scored misleadingly.

## Loop 4: Preserve source-schema truth

### Weakness

Schema normalization inserts missing columns, which could make a malformed source file appear complete after normalization.

### Fix

- Preserve the original source-column set in dataframe metadata.
- Compute required-column coverage from the source schema.
- Store source-schema provenance in reports and manifests.
- Validate 7z member paths before extraction.

### Validation

Tests verify that a source missing a required field cannot obtain 100% required-column coverage merely because normalization created a placeholder column.

## Loop 5: Improve public-data ingestion

### Weaknesses

- Head-only sampling can be biased.
- Full 7z decompression is expensive and repeated work is wasteful.
- Invalid dates can distort temporal splits.

### Fixes

- Deterministic chunked reservoir sampling with O(sample_size) retained memory.
- Explicit `head` mode only for schema checks.
- Persistent extraction cache keyed by archive size and modification time.
- Invalid dates stay visible to data-quality checks but are excluded from temporal modeling splits.

### Validation

Reservoir reproducibility, source-schema fidelity, safe archive members, and temporal chronology are covered by tests.

## Loop 6: Improve calibration validity and efficiency

### Weakness

Repeated cross-validated text-model calibration was expensive and mixed training with calibration responsibilities.

### Fix

Reuse the fitted base classifier and fit a scalar temperature on the dedicated calibration window. Report ECE, Brier score, and log loss together.

### Validation

The nominal root run preserved macro-F1 while reducing ECE from 0.3606 to 0.0828 and log loss from 0.6727 to 0.2855 on the untouched test set.

## Loop 7: Strengthen CrossEncoder evidence evaluation

### Weaknesses

- Per-case CrossEncoder invocation was inefficient.
- A reranker could be enabled without proving uplift over BM25.
- Candidate bounds and empty inputs required safer handling.

### Fixes

- Batch all query-candidate pairs.
- Record BM25 baseline and reranked Recall@K, MRR, and context precision.
- Record uplift and whether CrossEncoder was requested and actually used.
- Validate candidate indices and use safe BM25 fallback.

### Validation

Unit tests verify batching, fallback, and uplift calculations. Actual GPU CrossEncoder uplift remains a local-device validation because the current environment does not include the GPU/NLP stack.

## Loop 8: Harden Monte Carlo integrity

### Weaknesses

- Resume could reuse stale outputs after code or configuration changes.
- Missing or failed runs could still leave a misleading report.
- Model-selection stability was not summarized.

### Fixes

- Run signatures include scenario, seed, sample size, model flags, config hash, pipeline hash, and full source/script tree hash.
- Missing runs, subprocess failures, or failed sensitivity checks return non-zero.
- Added champion-distribution reporting.

### Validation

```text
18 expected
18 completed
0 failed
all sensitivity checks passed
```

The calibrated Logistic model was selected in all nominal and moderate runs, and 5 of 6 severe runs.

## Loop 9: Reproducibility and CI

### Fixes

- Pipeline manifest schema v2 with runtime, environment, config hashes, provenance, champion, optional-feature status, and artifact checksums.
- Core, CI, full-CPU, and GPU dependency layers.
- Ruff static checks.
- Python 3.10, 3.11, and 3.12 GitHub Actions matrix.
- Unit, pipeline, and Monte Carlo sensitivity tests.

### Validation

```text
python -m ruff check .     PASS
python -m pytest -q        40 passed
python -m compileall       PASS
```

## Stop condition

The remaining high-value work requires the user's local hardware:

- CUDA Transformer fine-tuning benchmark,
- GPU CrossEncoder throughput and uplift,
- VRAM and latency measurement.

FastAPI, Kubernetes, remote MLflow, Spark, and cloud deployment would broaden deployment scope but would not materially improve the target Model Quality and ML Evaluation evidence. They are intentionally excluded.


## Final dataset-only verification

- Added strict `data/raw` auto-discovery with no silent synthetic fallback.
- Added DuckDB reservoir sampling for large public narrative CSVs.
- Added dataset preflight and end-to-end real-data audit commands.
- Added CPU thread controls to prevent BLAS/OpenMP oversubscription.
- Verified the uploaded 189.8 MB CFPB 7z / 1.88 GB CSV through the full pipeline.
- Final static/unit/integration status: Ruff PASS, 45 tests passed, synthetic audit PASS, Monte Carlo smoke PASS, real-data audit PASS.
