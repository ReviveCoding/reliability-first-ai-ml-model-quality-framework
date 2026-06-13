# v6 Small-Data Validation

## Objective

Confirm that the v6 protocol works beyond one nominal synthetic run and that it reacts correctly to temporal label novelty.

## Validation A: noisy nominal synthetic data

Configuration:

```text
800 rows
seed 20260612
four-way temporal split
TF-IDF Logistic + temperature scaling + LightGBM
BM25 evidence evaluation
```

Result:

```text
launch decision: PASS
champion: tfidf_logistic_temperature_scaled
selection macro-F1: 0.8162
untouched-test macro-F1: 0.9138
95% bootstrap CI: [0.8598, 0.9589]
PR-AUC: 0.9719
ECE: 0.0828
Brier: 0.1235
log loss: 0.2855
worst-slice F1: 0.7981
Evidence Recall@5: 0.9750
context precision: 0.5717
```

The result is not a perfect toy score. Calibration materially improves probability quality while the test set remains untouched for final evidence.

## Validation B: public-schema stable temporal CSV

A 1,000-row CFPB-shaped CSV with all required source columns and stable classes was loaded through the public-file path, normalized, split temporally, trained, evaluated, and gated.

```text
launch decision: PASS
source required-column coverage: 1.0
unknown-target rate: 0.0
split chronology: PASS
```

The intentionally simple text templates made prediction too easy, so this run is treated only as ingestion, schema, split, and protocol validation, not as performance evidence.

## Validation C: future unseen product class

A second 1,000-row CFPB-shaped CSV introduced `Crypto service` only in the final temporal test window.

```text
test novel-target rate: 1.0
test class coverage: 0.0
evaluated known-label rows: 0
launch decision: BLOCK
```

This verifies that the training-only label policy does not inspect future labels and does not silently remap an unseen future product into a known class.

## Historical real-public schema result

A prior 1,000-row run from the uploaded public CFPB archive completed end to end and returned `REVIEW`, with macro-F1 0.5414, ECE 0.0850, log loss 0.6763, worst-slice F1 0.3253, and Evidence Recall@5 0.8400. Because v6 changed the split and model-selection protocol, those exact numbers are retained only as historical schema-compatibility evidence, not as current v6 benchmark claims.

## Large-archive limitation

The uploaded 7z contains a roughly 1.88 GB decompressed CSV. A first run must decompress that member before reservoir sampling and exceeded the execution budget in this environment. v6 adds safe member validation and a persistent extraction cache, but it cannot eliminate the one-time decompression cost. For local use, pre-extract the CSV or provide `--archive-cache-dir`.
