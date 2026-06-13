<!-- selection-test-launch-v3 -->
# Selection, Untouched Test, and Launch Readiness

A model selected on the dedicated selection window is a **selection winner**.
It becomes a **promotion candidate** only when model-selection eligibility is met.
It becomes the **approved champion** only when the untouched-test launch gate returns PASS.

## Model Selection Decision

- Selection winner: `transformer_text_classifier`
- Selection policy mode: `strict`
- Selection mode: `strict_eligible_selection`
- Eligibility status: `ELIGIBLE_CANDIDATE_SELECTED`
- Eligible candidate count: `2`
- Runner-up: `tfidf_logistic_temperature_scaled`
- Primary metric: `macro_f1`
- Selection Macro-F1: `0.5944291427339397`
- Test used for selection: `false`

## Untouched Test Evaluation

- Evaluated selection winner: `transformer_text_classifier`
- Promotion candidate: `transformer_text_classifier`
- Candidate test Macro-F1: `0.588931651431394`
- Candidate test PR-AUC: `0.6984157991749774`
- Candidate test ECE: `0.051741720577081`
- Candidate test Brier: `0.2341734558245586`
- Candidate test worst-slice F1: `0.2149600580973129`
- Test used for reselection: `false`

## Launch Readiness Decision

- Decision: `REVIEW`
- Promotion candidate: `transformer_text_classifier`
- Approved champion: `None`
- Fallback model: `tfidf_logistic_temperature_scaled`

<!-- end-selection-test-launch-v3 -->

# Executive Summary

## Project
End-to-End AI/ML Model Quality Sign-Off Framework

## Launch decision
**REVIEW**

## Scope
Local-runnable public/proxy/synthetic data framework for data validation, split-integrity checks, supervised model training, MLflow experiment tracking, evidence-quality evaluation, drift testing, synthetic GenAI telemetry replay, and PASS/REVIEW/BLOCK launch gates.

## Data provenance
```json
{
  "source_type": "public_cfpb",
  "source_file": "CFPB complaints 20200518-20260518 complaints only with narratives.csv",
  "sampling_strategy_requested": "auto",
  "sampling_strategy_effective": "duckdb",
  "source_chunksize": 50000,
  "archive_cache_enabled": true,
  "requested_sample_size": 20000,
  "actual_modeling_rows": 19996,
  "min_target_count": 30,
  "product_taxonomy_version": 1,
  "product_taxonomy_changed_row_rate": 0.257001400280056,
  "source_required_columns_present": [
    "company_response_to_consumer",
    "consumer_complaint_narrative",
    "date_received",
    "issue",
    "product",
    "state",
    "timely_response"
  ]
}
```

## Legacy Selection Label (deprecated)
- **Model:** transformer_text_classifier
- **Reason:** Selected on the dedicated model-selection window. Primary task restricted to target_col='product'. No model met predictive floors; selected best available model.

## Model selection leaderboard

| model                             | target_col      |   total_rows |   evaluated_rows |   unknown_target_rate |   accuracy |   macro_f1 |   evaluation_class_coverage |   roc_auc |   pr_auc |       ece |    brier |   log_loss |   worst_slice_f1 |   macro_f1_ci_low |   macro_f1_ci_high |   bootstrap_resamples | note                                                                                                 | device   |
|:----------------------------------|:----------------|-------------:|-----------------:|----------------------:|-----------:|-----------:|----------------------------:|----------:|---------:|----------:|---------:|-----------:|-----------------:|------------------:|-------------------:|----------------------:|:-----------------------------------------------------------------------------------------------------|:---------|
| tfidf_logistic                    | product         |         1499 |             1499 |                     0 |   0.855904 |   0.533408 |                           1 |  0.945632 | 0.588823 | 0.130153  | 0.230636 |   0.529504 |         0.266667 |          nan      |         nan        |                   nan | nan                                                                                                  | nan      |
| tfidf_logistic_temperature_scaled | product         |         1499 |             1499 |                     0 |   0.855904 |   0.533408 |                           1 |  0.949143 | 0.596238 | 0.0137731 | 0.200503 |   0.44104  |         0.266667 |          nan      |         nan        |                   nan | nan                                                                                                  | nan      |
| lightgbm                          | timely_response |         1499 |             1499 |                     0 |   0.795197 |   0.461511 |                           1 |  0.963496 | 0.999708 | 0.146879  | 0.139709 |   0.45097  |         0.190476 |          nan      |         nan        |                   nan | nan                                                                                                  | nan      |
| transformer_text_classifier       | product         |         1499 |             1499 |                     0 |   0.906604 |   0.594429 |                           1 |  0.958947 | 0.634797 | 0.0269625 | 0.145833 |   0.327289 |         0.266667 |            0.5294 |           0.643097 |                   300 | Selection metrics used 1499/1499 known-label rows; final metrics used 3000/3000 rows on device=cuda. | cuda     |

## Final test leaderboard

| model                             | target_col      |   total_rows |   evaluated_rows |   unknown_target_rate |   accuracy |   macro_f1 |   evaluation_class_coverage |   roc_auc |   pr_auc |       ece |    brier |   log_loss |   worst_slice_f1 |   macro_f1_ci_low |   macro_f1_ci_high |   bootstrap_resamples |   validation_unknown_target_rate | note                                                                                                 | device   |
|:----------------------------------|:----------------|-------------:|-----------------:|----------------------:|-----------:|-----------:|----------------------------:|----------:|---------:|----------:|---------:|-----------:|-----------------:|------------------:|-------------------:|----------------------:|---------------------------------:|:-----------------------------------------------------------------------------------------------------|:---------|
| tfidf_logistic                    | product         |         3000 |             3000 |                     0 |   0.841    |   0.630589 |                           1 |  0.95045  | 0.670748 | 0.142999  | 0.270313 |   0.598051 |         0.331061 |          0.598043 |           0.659678 |                   400 |                              nan | nan                                                                                                  | nan      |
| tfidf_logistic_temperature_scaled | product         |         3000 |             3000 |                     0 |   0.841    |   0.630589 |                           1 |  0.951994 | 0.675154 | 0.0229307 | 0.239672 |   0.508589 |         0.331061 |          0.598043 |           0.659678 |                   400 |                              nan | nan                                                                                                  | nan      |
| lightgbm                          | timely_response |         3000 |             3000 |                     0 |   0.654667 |   0.411677 |                           1 |  0.953221 | 0.999512 | 0.109093  | 0.188811 |   0.557697 |         0        |          0.401975 |           0.421353 |                   400 |                              nan | nan                                                                                                  | nan      |
| transformer_text_classifier       | product         |         3000 |             3000 |                     0 |   0.845    |   0.588932 |                           1 |  0.946399 | 0.698416 | 0.0517417 | 0.234173 |   0.497261 |         0.21496  |          0.555091 |           0.620442 |                   300 |                                0 | Selection metrics used 1499/1499 known-label rows; final metrics used 3000/3000 rows on device=cuda. | cuda     |

## Split integrity

| check                 | split       | passed   |   metric | detail                                                                                                                                                                                                                                                                   |
|:----------------------|:------------|:---------|---------:|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| split_size            | train       | True     |    13997 | train_rows=13997                                                                                                                                                                                                                                                         |
| split_size            | calibration | True     |     1500 | calibration_rows=1500                                                                                                                                                                                                                                                    |
| split_size            | selection   | True     |     1499 | selection_rows=1499                                                                                                                                                                                                                                                      |
| split_size            | test        | True     |     3000 | test_rows=3000                                                                                                                                                                                                                                                           |
| split_overlap         | *           | True     |        0 | identity_key=complaint_id; record_id_overlap_count=0; content_overlap_count=2                                                                                                                                                                                            |
| split_content_overlap | *           | False    |        2 | content_overlap_count=2; fingerprint_cols=['consumer_complaint_narrative', 'product_raw', 'product', 'issue', 'date_received']                                                                                                                                           |
| temporal_order        | *           | True     |        1 | train_min=2020-05-18 00:00:00; train_max=2025-03-05 00:00:00; calibration_min=2025-03-05 00:00:00; calibration_max=2025-05-15 00:00:00; selection_min=2025-05-15 00:00:00; selection_max=2025-07-22 00:00:00; test_min=2025-07-22 00:00:00; test_max=2026-04-28 00:00:00 |
| target_class_coverage | calibration | True     |        1 | target=product; coverage=1.0000                                                                                                                                                                                                                                          |
| target_class_coverage | selection   | True     |        1 | target=product; coverage=1.0000                                                                                                                                                                                                                                          |
| target_class_coverage | test        | True     |        1 | target=product; coverage=1.0000                                                                                                                                                                                                                                          |

## Key evidence-quality metrics

```json
{
  "recall_at_5": 0.9466666666666667,
  "mrr": 0.5313380281690141,
  "context_precision": 0.412,
  "unsupported_claim_rate": 0.053333333333333344,
  "bm25_recall_at_5": 0.9466666666666667,
  "bm25_mrr": 0.639906103286385,
  "bm25_context_precision": 0.2613333333333333,
  "rerank_recall_uplift": 0.0,
  "rerank_mrr_uplift": -0.10856807511737088,
  "rerank_context_precision_uplift": 0.15066666666666667,
  "cross_encoder_requested": true,
  "used_cross_encoder": true,
  "note": "",
  "case_count": 150
}
```
