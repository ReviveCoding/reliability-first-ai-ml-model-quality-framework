# Class-Weighted Transformer 3-Seed Challenger

## Final decision

- Selection decision: `REJECT_CHALLENGER`
- Final decision: `REJECT_WEIGHTED`
- Retained model: `transformer_text_classifier` (unweighted)
- Fallback: `tfidf_logistic_temperature_scaled`
- Approved champion: `null` because the full launch state remains `REVIEW`

## Floating-point comparison policy

A paired delta is counted as an improvement only when it is strictly greater than `1e-06`. Values within `+/-1e-06` of zero are treated as ties.

This correction changes reporting only. It does not change the predeclared thresholds or the validated model decision.

## Experimental control

- Seeds: `7, 17, 27`
- Epochs: `2`
- Frozen input rows: `19996`
- Frozen input SHA-256: `59648e6c775a54bf6f3d8c5f2b63cb8948f2bd270018d2619ff410c46414de66`
- Weighting mode: `sqrt_balanced`
- Weight source: train split only
- Checkpoint selection: calibration Macro-F1
- Challenger selection: dedicated selection split
- Test role: post-selection confirmation only
- Test used for challenger selection: `false`

## Paired baseline versus weighted results

|   seed |   baseline_selection_macro_f1 |   weighted_selection_macro_f1 |   delta_selection_macro_f1 |   baseline_selection_pr_auc |   weighted_selection_pr_auc |   delta_selection_pr_auc |   baseline_selection_ece |   weighted_selection_ece |   delta_selection_ece |   baseline_selection_worst_slice_f1 |   weighted_selection_worst_slice_f1 |   delta_selection_worst_slice_f1 |   baseline_test_macro_f1 |   weighted_test_macro_f1 |   delta_test_macro_f1 |   baseline_test_pr_auc |   weighted_test_pr_auc |   delta_test_pr_auc |   baseline_test_ece |   weighted_test_ece |   delta_test_ece |   baseline_test_worst_slice_f1 |   weighted_test_worst_slice_f1 |   delta_test_worst_slice_f1 |
|-------:|------------------------------:|------------------------------:|---------------------------:|----------------------------:|----------------------------:|-------------------------:|-------------------------:|-------------------------:|----------------------:|------------------------------------:|------------------------------------:|---------------------------------:|-------------------------:|-------------------------:|----------------------:|-----------------------:|-----------------------:|--------------------:|--------------------:|--------------------:|-----------------:|-------------------------------:|-------------------------------:|----------------------------:|
|      7 |                      0.570345 |                      0.613428 |                 0.0430828  |                    0.634404 |                    0.669952 |               0.035548   |                0.024172  |                0.0255673 |            0.00139532 |                            0.266667 |                            0.214286 |                     -0.052381    |                 0.607612 |                 0.626886 |           0.019274    |               0.678773 |               0.698096 |         0.0193235   |           0.0552976 |           0.0511609 |      -0.00413671 |                       0.311275 |                       0.257787 |                  -0.0534872 |
|     17 |                      0.562552 |                      0.612223 |                 0.0496708  |                    0.641228 |                    0.666583 |               0.0253553  |                0.0319146 |                0.0252022 |           -0.00671242 |                            0.266667 |                            0.266667 |                      5.55112e-17 |                 0.600164 |                 0.625052 |           0.0248872   |               0.692586 |               0.70742  |         0.0148336   |           0.0582088 |           0.051185  |      -0.00702373 |                       0.336275 |                       0.215741 |                  -0.120534  |
|     27 |                      0.599672 |                      0.607002 |                 0.00732984 |                    0.670993 |                    0.663047 |              -0.00794662 |                0.034015  |                0.0327657 |           -0.00124927 |                            0.227273 |                            0.266667 |                      0.0393939   |                 0.609417 |                 0.610255 |           0.000838096 |               0.694227 |               0.694464 |         0.000236901 |           0.0627041 |           0.0559197 |      -0.00678435 |                       0.286275 |                       0.347222 |                   0.0609477 |

## Tolerance-corrected delta summary

| metric                   |   mean_delta |   sample_std_delta |    min_delta |   max_delta |   improvement_tolerance |   improved_seed_count |   non_degraded_seed_count |
|:-------------------------|-------------:|-------------------:|-------------:|------------:|------------------------:|----------------------:|--------------------------:|
| selection_macro_f1       |   0.0333611  |         0.0227832  |  0.00732984  |  0.0496708  |                   1e-06 |                     3 |                         3 |
| selection_pr_auc         |   0.0176522  |         0.0227475  | -0.00794662  |  0.035548   |                   1e-06 |                     2 |                         2 |
| selection_ece            |  -0.00218879 |         0.00413472 | -0.00671242  |  0.00139532 |                   1e-06 |                     1 |                         2 |
| selection_worst_slice_f1 |  -0.004329   |         0.0460403  | -0.052381    |  0.0393939  |                   1e-06 |                     1 |                         2 |
| test_macro_f1            |   0.0149998  |         0.0125814  |  0.000838096 |  0.0248872  |                   1e-06 |                     3 |                         3 |
| test_pr_auc              |   0.0114647  |         0.00997933 |  0.000236901 |  0.0193235  |                   1e-06 |                     3 |                         3 |
| test_ece                 |  -0.0059816  |         0.00160219 | -0.00702373  | -0.00413671 |                   1e-06 |                     0 |                         1 |
| test_worst_slice_f1      |  -0.0376911  |         0.0917661  | -0.120534    |  0.0609477  |                   1e-06 |                     1 |                         1 |

## Selection checks

| name                                      |       value | operator   |   threshold | passed   |
|:------------------------------------------|------------:|:-----------|------------:|:---------|
| selection.mean_macro_f1_delta             |  0.0333611  | >=         |      -0.005 | True     |
| selection.macro_f1_improved_seed_count    |  3          | >=         |       2     | True     |
| selection.mean_worst_slice_f1_delta       | -0.004329   | >=         |       0.04  | False    |
| selection.worst_slice_improved_seed_count |  1          | >=         |       2     | False    |
| selection.mean_ece_delta                  | -0.00218879 | <=         |       0.02  | True     |
| selection.mean_pr_auc_delta               |  0.0176522  | >=         |      -0.01  | True     |

## Post-selection test confirmation

| name                                              |      value | operator   |   threshold | passed   |
|:--------------------------------------------------|-----------:|:-----------|------------:|:---------|
| test_confirmation.mean_macro_f1_delta             |  0.0149998 | >=         |      -0.005 | True     |
| test_confirmation.mean_worst_slice_f1_delta       | -0.0376911 | >=         |       0.02  | False    |
| test_confirmation.worst_slice_improved_seed_count |  1         | >=         |       2     | False    |
| test_confirmation.mean_ece_delta                  | -0.0059816 | <=         |       0.02  | True     |
| test_confirmation.mean_pr_auc_delta               |  0.0114647 | >=         |      -0.01  | True     |

## Interpretation

The weighted challenger improved average Macro-F1, PR-AUC, and calibration, but it did not improve worst-slice F1 consistently. Because worst-slice reliability was the stated objective, the challenger remains rejected.
