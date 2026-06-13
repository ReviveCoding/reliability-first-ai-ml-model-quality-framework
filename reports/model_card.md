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

# Model Card: Promotion Candidate and Launch Readiness

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
- **Selection reason:** Selected on the dedicated model-selection window. Primary task restricted to target_col='product'. No model met predictive floors; selected best available model.
- **Selection macro-F1:** 0.5944291427339397
- **Selection/test macro-F1 gap:** 0.005497491302545621

## Intended use
Offline portfolio-style model quality evaluation on public CFPB-style complaint data and synthetic telemetry. This is not a production financial decision system.

## Evaluation summary
- Final test macro-F1: 0.588931651431394
- Macro-F1 95% bootstrap interval: [0.5550909995643581, 0.6204419152734193]
- Final evaluated rows: 3000.0
- PR-AUC: 0.6984157991749774
- ECE: 0.05174172057708107
- Brier: 0.23417345582455862
- Log loss: 0.4972614645957947
- Worst-slice F1: 0.21496005809731297
- Unknown-target rate: 0.0

## Split integrity
```json
{
  "train_rows": 13997,
  "validation_rows": 1499,
  "selection_rows": 1499,
  "test_rows": 3000,
  "no_overlap": 1.0,
  "record_id_no_overlap": 1.0,
  "record_id_overlap_count": 0,
  "content_overlap_count": 2,
  "overlap_identity_key": "complaint_id",
  "temporal_order_valid": 1.0,
  "min_eval_fraction": 0.07496499299859972,
  "product_calibration_class_coverage": 1.0,
  "product_selection_class_coverage": 1.0,
  "product_test_class_coverage": 1.0,
  "calibration_rows": 1500
}
```

## Product taxonomy
```json
{
  "taxonomy_version": 1,
  "source": "CFPB historical product-category normalization for longitudinal modeling",
  "description": "Preserves the original CFPB product label while mapping historically renamed or reorganized product categories into a stable canonical target for temporal model development and evaluation.",
  "config_path": "configs\\product_taxonomy.yaml",
  "target_column": "product",
  "raw_column": "product_raw",
  "mapping_count": 5,
  "row_count": 19996,
  "changed_row_count": 5139,
  "changed_row_rate": 0.257001400280056,
  "raw_class_count": 14,
  "canonical_class_count": 10,
  "mapping_hits": {
    "Credit card": 598,
    "Credit card or prepaid card": 479,
    "Credit reporting, credit repair services, or other personal consumer reports": 3943,
    "Payday loan, title loan, or personal loan": 62,
    "Prepaid card": 57
  },
  "unmapped_raw_labels": [
    "Checking or savings account",
    "Credit reporting or other personal consumer reports",
    "Debt collection",
    "Debt or credit management",
    "Money transfer, virtual currency, or money service",
    "Mortgage",
    "Payday loan, title loan, personal loan, or advance loan",
    "Student loan",
    "Vehicle loan or lease"
  ]
}
```

## Label policy
```json
{
  "policy_version": 2,
  "fit_scope": "training_only",
  "target_col": "product",
  "min_count": 30,
  "other_label": "Other",
  "train_label_counts": {
    "Checking or savings account": 723,
    "Credit card / prepaid card": 845,
    "Credit reporting or other personal consumer reports": 9779,
    "Debt collection": 1284,
    "Debt or credit management": 17,
    "Money transfer, virtual currency, or money service": 523,
    "Mortgage": 388,
    "Payday loan, title loan, personal loan, or advance loan": 104,
    "Student loan": 164,
    "Vehicle loan or lease": 170
  },
  "retained_labels": [
    "Checking or savings account",
    "Credit card / prepaid card",
    "Credit reporting or other personal consumer reports",
    "Debt collection",
    "Money transfer, virtual currency, or money service",
    "Mortgage",
    "Payday loan, title loan, personal loan, or advance loan",
    "Student loan",
    "Vehicle loan or lease"
  ],
  "collapsed_labels": [
    "Debt or credit management"
  ],
  "collapsed_train_row_count": 17,
  "application": {
    "train": {
      "row_count": 13997,
      "collapsed_row_count": 17,
      "novel_target_row_count": 0,
      "novel_target_rate": 0.0,
      "novel_labels": []
    },
    "calibration": {
      "row_count": 1500,
      "collapsed_row_count": 1,
      "novel_target_row_count": 0,
      "novel_target_rate": 0.0,
      "novel_labels": []
    },
    "selection": {
      "row_count": 1499,
      "collapsed_row_count": 3,
      "novel_target_row_count": 0,
      "novel_target_rate": 0.0,
      "novel_labels": []
    },
    "test": {
      "row_count": 3000,
      "collapsed_row_count": 10,
      "novel_target_row_count": 0,
      "novel_target_rate": 0.0,
      "novel_labels": []
    }
  }
}
```

## Limitations
CFPB public complaints are not a representative sample of all consumer experiences, and synthetic telemetry is used only to simulate GenAI workflow quality signals. Launch decisions are local/offline quality gates, not production approvals.
