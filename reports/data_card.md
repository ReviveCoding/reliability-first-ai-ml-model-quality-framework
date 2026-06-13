# Data Card

## Data sources
- Public CFPB-style complaint records when a CFPB CSV is supplied.
- Synthetic CFPB-style fallback data for local smoke tests.
- Synthetic payment-style GenAI telemetry for evidence/grounding and workflow-quality evaluation.

## Provenance
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

## Product taxonomy normalization
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

## Label consolidation policy
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

## Quality checks
| check              | column                       | passed   |   metric | detail                                 |
|:-------------------|:-----------------------------|:---------|---------:|:---------------------------------------|
| required_column    | product                      | True     | 1        | missing_like_rate=0.0000               |
| required_column    | issue                        | True     | 1        | missing_like_rate=0.0000               |
| required_column    | consumer_complaint_narrative | True     | 1        | missing_like_rate=0.0000               |
| required_column    | company_response_to_consumer | True     | 1        | missing_like_rate=0.0000               |
| required_column    | timely_response              | True     | 1        | missing_like_rate=0.0000               |
| required_column    | date_received                | True     | 1        | missing_like_rate=0.0000               |
| required_column    | state                        | True     | 0.996599 | missing_like_rate=0.0034               |
| duplicate_rate     | *                            | True     | 0        | duplicate_rate=0.0000                  |
| completeness       | *                            | True     | 0.999514 | missing-like-aware completeness=0.9995 |
| label_missing_rate | product                      | True     | 0        | label_missing_like_rate=0.0000         |
| label_missing_rate | timely_response              | True     | 0        | label_missing_like_rate=0.0000         |
| date_validity      | date_received                | True     | 1        | valid_date_rate=1.0000                 |
| evidence_coverage  | consumer_complaint_narrative | True     | 0.996699 | evidence_coverage=0.9967               |

## Split checks
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

## Claim boundary
This project does not use Apple, private customer, private payment, or production monitoring data. Synthetic telemetry is not evidence of real production behavior.
