# Changelog

## 0.6.0

- Added strict dataset-only execution: real data is auto-discovered from `data/raw`, and missing data no longer silently falls back to synthetic input.
- Added DuckDB reservoir sampling for multi-gigabyte public CSV files.
- Added dataset preflight and full real-dataset audit commands.
- Added CPU thread limits to prevent BLAS/OpenMP oversubscription.
- Added GitHub CI coverage for the public CFPB file-ingestion path.

## v6

- Removed model-selection leakage: champion selection now uses a dedicated selection window, followed by final evaluation on an untouched temporal test window.
- Removed label-frequency leakage: rare-label policy is fit on training data only and applied unchanged to later windows.
- Added calibration and selection sub-windows, four-way split-integrity checks, selection/test generalization-gap monitoring, and bootstrap macro-F1 confidence intervals.
- Added source-schema provenance so normalized placeholder columns cannot hide columns missing from the original public file.
- Added safe `.7z` member validation against path traversal.
- Added BM25 baseline metrics, CrossEncoder uplift metrics, and batched CrossEncoder inference.
- Added artifact checksums to the pipeline manifest and full-code-tree hashes to Monte Carlo resume signatures.
- Expanded CI to Python 3.10, 3.11, and 3.12 with `pip check`.

## v5

- Hardened Monte Carlo resume integrity, public temporal labels, calibration speed, public-data sampling, optional GPU paths, manifests, and clean-ZIP execution.
