# Public Claim Boundaries

This document defines what the repository demonstrates and what it does not claim.

## Supported claims

- The repository implements a reproducible, reliability-first AI/ML model-quality workflow.
- It includes data-quality checks, leakage-safe split semantics, calibration, worst-slice evaluation, multi-seed stability, drift and stress testing, launch gates, provenance, and compact decision evidence.
- Local validation includes passing lint and automated tests recorded in the repository documentation.
- Reported model decisions are tied to included JSON, CSV, and Markdown evidence artifacts.
- GPU training and inference paths were designed for local execution, while GitHub CI uses a smaller CPU-compatible validation path.

## Data claims

- Data should be described as public, synthetic, proxy, sampled, or locally generated according to its actual source.
- The repository does not claim access to proprietary Apple, financial-institution, advertiser, customer, or other private company data.
- Raw datasets and model checkpoints are intentionally excluded from Git.

## Deployment claims

- This is a production-style research and portfolio framework, not evidence of deployment inside a commercial production environment.
- API, dashboard, CI, monitoring, and governance components demonstrate engineering design and reproducibility, not a live service-level agreement.
- Launch-gate outputs are framework decisions, not formal regulatory approval or third-party certification.

## Metric claims

- Metrics apply only to the documented data, split, configuration, seed, and artifact version.
- Improvements smaller than the documented numerical tolerance are treated as ties.
- A higher aggregate metric does not imply promotion when worst-slice, stability, calibration, or governance conditions fail.
- The final test split must remain isolated from checkpoint selection, framework selection, threshold tuning, and reselection.

## Current model decision

- Selected candidate: unweighted `transformer_text_classifier`
- Fallback: calibrated `tfidf_logistic_temperature_scaled`
- Class-weighted challenger: `REJECT_WEIGHTED`
- Launch state: `REVIEW`

The detailed evidence and rationale are maintained in `docs/FINAL_MODEL_DECISION.md` and `docs/artifacts/final_evaluation/`.
