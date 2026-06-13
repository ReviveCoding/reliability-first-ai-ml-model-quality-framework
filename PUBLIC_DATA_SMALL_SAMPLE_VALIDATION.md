# Historical Public CFPB Small-Sample Validation

A prior 1,000-row run from the uploaded public CFPB complaints archive completed ingestion, normalization, temporal modeling, calibration, evidence evaluation, and gating.

```text
launch decision: REVIEW
macro-F1: 0.5414
PR-AUC: 0.7497
ECE: 0.0850
Brier score: 0.3165
log loss: 0.6763
worst-slice F1: 0.3253
Evidence Recall@5: 0.8400
context precision: 0.3053
```

Temperature scaling improved ECE from 0.4461 to 0.0850 and log loss from 1.2605 to 0.6763, but the gate remained `REVIEW` because slice quality and evidence precision were weak.

This run predates the v6 four-way split and selection-only champion protocol. The numbers are retained as historical real-public schema/execution evidence, not as current v6 benchmark claims. See `SMALL_DATA_VALIDATION.md` for current v6 protocol validation.
