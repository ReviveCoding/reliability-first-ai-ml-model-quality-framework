# Launch Decision Memo

**Decision:** REVIEW

## Failed checks

| name                               |       value |   threshold | passed   | mode   |   violation_ratio |
|:-----------------------------------|------------:|------------:|:---------|:-------|------------------:|
| model_quality.best_macro_f1        | 0.588932    |       0.6   | False    | min    |        0.0184472  |
| model_quality.worst_slice_f1       | 0.21496     |       0.45  | False    | min    |        0.522311   |
| evidence_quality.context_precision | 0.412       |       0.5   | False    | min    |        0.176      |
| drift.max_psi                      | 0.251338    |       0.25  | False    | max    |        0.00535242 |
| drift.min_ks_pvalue                | 0.000640522 |       0.001 | False    | min    |        0.359478   |

## All gate checks

| name                                               |          value |   threshold | passed   | mode   |   violation_ratio |
|:---------------------------------------------------|---------------:|------------:|:---------|:-------|------------------:|
| data_quality.required_column_coverage              |    1           |       1     | True     | min    |        0          |
| data_quality.completeness                          |    0.999514    |       0.98  | True     | min    |        0          |
| data_quality.duplicate_rate                        |    0           |       0.01  | True     | max    |        0          |
| data_quality.label_missing_rate                    |    0           |       0.02  | True     | max    |        0          |
| data_quality.date_validity                         |    1           |       0.98  | True     | min    |        0          |
| data_quality.evidence_coverage                     |    0.996699    |       0.9   | True     | min    |        0          |
| split_integrity.no_overlap                         |    1           |       1     | True     | min    |        0          |
| split_integrity.temporal_order_valid               |    1           |       1     | True     | min    |        0          |
| split_integrity.product_calibration_class_coverage |    1           |       0.8   | True     | min    |        0          |
| split_integrity.product_selection_class_coverage   |    1           |       0.8   | True     | min    |        0          |
| split_integrity.product_test_class_coverage        |    1           |       0.8   | True     | min    |        0          |
| model_quality.best_macro_f1                        |    0.588932    |       0.6   | False    | min    |        0.0184472  |
| model_quality.best_pr_auc                          |    0.698416    |       0.5   | True     | min    |        0          |
| model_quality.best_ece                             |    0.0517417   |       0.2   | True     | max    |        0          |
| model_quality.best_brier                           |    0.234173    |       0.35  | True     | max    |        0          |
| model_quality.best_log_loss                        |    0.497261    |       1.2   | True     | max    |        0          |
| model_quality.worst_slice_f1                       |    0.21496     |       0.45  | False    | min    |        0.522311   |
| model_quality.unknown_target_rate                  |    0           |       0.05  | True     | max    |        0          |
| model_quality.evaluated_rows                       | 3000           |      50     | True     | min    |        0          |
| model_quality.macro_f1_ci_low                      |    0.555091    |       0.45  | True     | min    |        0          |
| model_quality.selection_test_macro_f1_gap          |    0.00549749  |       0.15  | True     | max    |        0          |
| evidence_quality.recall_at_5                       |    0.946667    |       0.5   | True     | min    |        0          |
| evidence_quality.mrr                               |    0.531338    |       0.4   | True     | min    |        0          |
| evidence_quality.context_precision                 |    0.412       |       0.5   | False    | min    |        0.176      |
| evidence_quality.unsupported_claim_rate            |    0.0533333   |       0.1   | True     | max    |        0          |
| drift.max_psi                                      |    0.251338    |       0.25  | False    | max    |        0.00535242 |
| drift.min_ks_pvalue                                |    0.000640522 |       0.001 | False    | min    |        0.359478   |
| drift.training_serving_skew                        |    0           |       0.15  | True     | max    |        0          |
| genai_telemetry.task_completion_rate               |    0.971       |       0.8   | True     | min    |        0          |
| genai_telemetry.latency_p95_ms                     | 1660.84        |    3500     | True     | max    |        0          |
| genai_telemetry.regression_flag_rate               |    0.0493333   |       0.1   | True     | max    |        0          |
