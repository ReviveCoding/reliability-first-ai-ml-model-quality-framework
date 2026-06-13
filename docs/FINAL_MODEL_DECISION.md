# Final Model Decision

## Decision

Retain the **unweighted Transformer** as the selected promotion-candidate
configuration and retain the **temperature-scaled TF-IDF Logistic**
model as the operational fallback.

The sqrt-balanced class-weighted Transformer challenger is rejected.

The project launch state remains **REVIEW**, so no model is labeled as an
approved champion.

## Evidence

- Baseline Transformer stability verdict: `STABLE`
- Baseline seeds: `7, 17, 27`
- Frozen input rows: `19996`
- Weighted mean selection Macro-F1 delta: `+0.033361`
- Weighted mean test Macro-F1 delta: `+0.015000`
- Weighted mean selection worst-slice F1 delta: `-0.004329`
- Weighted mean test worst-slice F1 delta: `-0.037691`
- Weighted challenger decision: `REJECT_WEIGHTED`

## Why the weighted challenger was rejected

The weighted model improved average classification and ranking metrics,
but it did not improve worst-slice reliability consistently. Worst-slice
improvement was the predeclared objective, so average-metric gains were
not sufficient for adoption.

## Floating-point reporting policy

A paired delta counts as improved only when `delta > 1e-06`.
Values within `+/-1e-06` of zero are ties. This prevents numerical
noise such as `5.55e-17` from being reported as a real improvement.

## Model roles

| Role | Model | Status |
|---|---|---|
| Selection winner | `transformer_text_classifier` | Retained |
| Promotion candidate | `transformer_text_classifier` | Retained |
| Fallback | `tfidf_logistic_temperature_scaled` | Retained |
| Weighted challenger | sqrt-balanced Transformer | Rejected |
| Approved champion | None | Launch remains REVIEW |

## Canonical evidence

See [`docs/artifacts/final_evaluation`](artifacts/final_evaluation).
