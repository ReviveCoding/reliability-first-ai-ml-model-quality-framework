# Monte Carlo End-to-End Validation Report

## Design
Repeated end-to-end runs across nominal, moderate, and severe scenarios. Each run uses an independent child seed, scenario-specific synthetic data generation, drift injection, telemetry generation, model training, quality evaluation, and PASS/REVIEW/BLOCK launch gating.

## Gate distribution

| scenario   |   BLOCK |   PASS |   REVIEW |
|:-----------|--------:|-------:|---------:|
| moderate   |       0 |      0 |        2 |
| nominal    |       0 |      2 |        0 |
| severe     |       2 |      0 |        0 |

## Scenario summary

| scenario   |   runs |   pass_rate |   review_rate |   block_rate |   macro_f1_mean |   macro_f1_ci95_low |   macro_f1_ci95_high |   log_loss_mean |   max_psi_mean |   task_completion_rate_mean |   latency_p95_ms_mean |   regression_flag_rate_mean |
|:-----------|-------:|------------:|--------------:|-------------:|----------------:|--------------------:|---------------------:|----------------:|---------------:|----------------------------:|----------------------:|----------------------------:|
| nominal    |      2 |           1 |             0 |            0 |        0.911883 |            0.88485  |             0.938916 |        0.344524 |      0.0539706 |                    0.966667 |               1711.32 |                   0.0666667 |
| moderate   |      2 |           0 |             1 |            0 |        0.711712 |            0.707127 |             0.716297 |        0.855378 |      0.307875  |                    0.858333 |               3356.4  |                   0.179167  |
| severe     |      2 |           0 |             0 |            1 |        0.436435 |            0.381014 |             0.491856 |        1.3974   |      1.00855   |                    0.154167 |               5852.38 |                   0.875     |

## Champion-selection stability

| scenario   | champion_model                    |   count |   rate |
|:-----------|:----------------------------------|--------:|-------:|
| moderate   | tfidf_logistic_temperature_scaled |       2 |      1 |
| nominal    | tfidf_logistic_temperature_scaled |       2 |      1 |
| severe     | tfidf_logistic_temperature_scaled |       2 |      1 |

## Monotonicity and sensitivity checks

```json
{
  "metric_checks": [
    {
      "metric": "macro_f1_mean",
      "direction": "nonincreasing",
      "values": [
        0.9118829026556117,
        0.7117119522861067,
        0.4364348610429114
      ],
      "passed": true
    },
    {
      "metric": "pr_auc_mean",
      "direction": "nonincreasing",
      "values": [
        0.9479014747865899,
        0.7759640325715078,
        0.49642359485534265
      ],
      "passed": true
    },
    {
      "metric": "worst_slice_f1_mean",
      "direction": "nonincreasing",
      "values": [
        0.7063257575757576,
        0.5767396655631949,
        0.26964285714285713
      ],
      "passed": true
    },
    {
      "metric": "evidence_recall_at_5_mean",
      "direction": "nonincreasing",
      "values": [
        0.9624999999999999,
        0.8916666666666666,
        0.6916666666666667
      ],
      "passed": true
    },
    {
      "metric": "context_precision_mean",
      "direction": "nonincreasing",
      "values": [
        0.5875,
        0.4066666666666667,
        0.3125
      ],
      "passed": true
    },
    {
      "metric": "task_completion_rate_mean",
      "direction": "nonincreasing",
      "values": [
        0.9666666666666667,
        0.8583333333333334,
        0.15416666666666667
      ],
      "passed": true
    },
    {
      "metric": "log_loss_mean",
      "direction": "nondecreasing",
      "values": [
        0.3445243938118487,
        0.8553782426968207,
        1.3974007307972887
      ],
      "passed": true
    },
    {
      "metric": "unsupported_claim_rate_mean",
      "direction": "nondecreasing",
      "values": [
        0.03750000000000003,
        0.10833333333333334,
        0.30833333333333335
      ],
      "passed": true
    },
    {
      "metric": "max_psi_mean",
      "direction": "nondecreasing",
      "values": [
        0.05397061899385278,
        0.3078754567229651,
        1.0085461650719245
      ],
      "passed": true
    },
    {
      "metric": "latency_p95_ms_mean",
      "direction": "nondecreasing",
      "values": [
        1711.3187018663584,
        3356.3991203607443,
        5852.383626962369
      ],
      "passed": true
    },
    {
      "metric": "regression_flag_rate_mean",
      "direction": "nondecreasing",
      "values": [
        0.06666666666666667,
        0.17916666666666664,
        0.875
      ],
      "passed": true
    },
    {
      "metric": "human_review_rate_mean",
      "direction": "nondecreasing",
      "values": [
        0.06666666666666667,
        0.22083333333333333,
        0.8916666666666666
      ],
      "passed": true
    }
  ],
  "gate_checks": [
    {
      "name": "nominal_block_rate_le_0.20",
      "value": 0.0,
      "passed": true
    },
    {
      "name": "severe_block_rate_ge_0.80",
      "value": 1.0,
      "passed": true
    }
  ],
  "all_passed": true,
  "completion_checks": [
    {
      "name": "completed_run_count_matches_expected",
      "expected": 6,
      "actual": 6,
      "passed": true
    },
    {
      "name": "nominal_run_count_matches_expected",
      "expected": 2,
      "actual": 2,
      "passed": true
    },
    {
      "name": "moderate_run_count_matches_expected",
      "expected": 2,
      "actual": 2,
      "passed": true
    },
    {
      "name": "severe_run_count_matches_expected",
      "expected": 2,
      "actual": 2,
      "passed": true
    },
    {
      "name": "pipeline_failures_zero",
      "expected": 0,
      "actual": 0,
      "passed": true
    }
  ],
  "pipeline_failures": []
}
```

## Interpretation
- Predictive quality should decline as ambiguity and label conflict increase.
- Log loss, drift, latency, regression flags, and human-review demand should increase with stress.
- Nominal runs should avoid frequent BLOCK decisions, while severe runs should usually BLOCK.
- Confidence intervals summarize run-to-run uncertainty; they do not imply production guarantees.
