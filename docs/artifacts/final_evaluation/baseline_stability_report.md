# Transformer 3-Seed Stability Audit

## Experimental control

- Seeds: `7, 17, 27`
- Frozen input rows: `19996`
- Frozen input SHA-256: `59648e6c775a54bf6f3d8c5f2b63cb8948f2bd270018d2619ff410c46414de66`
- Class weighting: `disabled`
- Checkpoint selection: calibration Macro-F1
- Framework model selection: dedicated selection split
- Final evaluation: untouched test split
- Test used for tuning or reselection: `false`

## Per-seed results

|   seed | selection_winner            | eligibility_status          | launch_decision   |   best_checkpoint_macro_f1 |   selection_macro_f1 |   test_macro_f1 |   test_pr_auc |   test_ece |   test_worst_slice_f1 | best_model_checkpoint                                                                                                                   |   selection_rows |   test_rows |   fallback_test_macro_f1 |   transformer_minus_fallback_test_macro_f1 |
|-------:|:----------------------------|:----------------------------|:------------------|---------------------------:|---------------------:|----------------:|--------------:|-----------:|----------------------:|:----------------------------------------------------------------------------------------------------------------------------------------|-----------------:|------------:|-------------------------:|-------------------------------------------:|
|      7 | transformer_text_classifier | ELIGIBLE_CANDIDATE_SELECTED | REVIEW            |                   0.590056 |             0.570345 |        0.607612 |      0.678773 |  0.0552976 |              0.311275 | <REPO_ROOT>\sa\s_20260612_234110\seed_7\outputs\transformer_model\checkpoint-876  |             1499 |        3000 |                 0.626406 |                                 -0.0187939 |
|     17 | transformer_text_classifier | ELIGIBLE_CANDIDATE_SELECTED | REVIEW            |                   0.632802 |             0.562552 |        0.600164 |      0.692586 |  0.0582088 |              0.336275 | <REPO_ROOT>\sa\s_20260612_234110\seed_17\outputs\transformer_model\checkpoint-876 |             1499 |        3000 |                 0.627316 |                                 -0.0271522 |
|     27 | transformer_text_classifier | ELIGIBLE_CANDIDATE_SELECTED | REVIEW            |                   0.610266 |             0.599672 |        0.609417 |      0.694227 |  0.0627041 |              0.286275 | <REPO_ROOT>\sa\s_20260612_234110\seed_27\outputs\transformer_model\checkpoint-876 |             1499 |        3000 |                 0.627265 |                                 -0.0178485 |

## Stability summary

| metric                                   |   n |       mean |   sample_std |        min |        max |      range |   coefficient_of_variation |
|:-----------------------------------------|----:|-----------:|-------------:|-----------:|-----------:|-----------:|---------------------------:|
| best_checkpoint_macro_f1                 |   3 |  0.611041  |   0.0213837  |  0.590056  |  0.632802  | 0.0427463  |                 0.0349955  |
| selection_macro_f1                       |   3 |  0.577523  |   0.0195736  |  0.562552  |  0.599672  | 0.0371203  |                 0.0338923  |
| test_macro_f1                            |   3 |  0.605731  |   0.00490458 |  0.600164  |  0.609417  | 0.00925238 |                 0.00809696 |
| test_pr_auc                              |   3 |  0.688529  |   0.00848871 |  0.678773  |  0.694227  | 0.0154545  |                 0.0123288  |
| test_ece                                 |   3 |  0.0587368 |   0.00373133 |  0.0552976 |  0.0627041 | 0.00740641 |                 0.0635263  |
| test_worst_slice_f1                      |   3 |  0.311275  |   0.025      |  0.286275  |  0.336275  | 0.05       |                 0.080315   |
| transformer_minus_fallback_test_macro_f1 |   3 | -0.0212648 |   0.00512043 | -0.0271522 | -0.0178485 | 0.00930365 |                 0.240793   |

## Guardrail checks

| name                           |      value | operator   |   threshold | passed   |
|:-------------------------------|-----------:|:-----------|------------:|:---------|
| selection_macro_f1.sample_std  | 0.0195736  | <=         |    0.02     | True     |
| test_macro_f1.sample_std       | 0.00490458 | <=         |    0.02     | True     |
| test_macro_f1.range            | 0.00925238 | <=         |    0.05     | True     |
| test_pr_auc.sample_std         | 0.00848871 | <=         |    0.01     | True     |
| test_ece.sample_std            | 0.00373133 | <=         |    0.02     | True     |
| test_worst_slice_f1.sample_std | 0.025      | <=         |    0.05     | True     |
| test_worst_slice_f1.range      | 0.05       | <=         |    0.12     | True     |
| transformer_selection_rate     | 1          | >=         |    0.666667 | True     |

## Decision

**STABLE**

Proceed to a class-weighted Transformer challenger experiment.

This is a three-seed engineering stability audit. It characterizes run-to-run sensitivity but is not a high-powered statistical study.
