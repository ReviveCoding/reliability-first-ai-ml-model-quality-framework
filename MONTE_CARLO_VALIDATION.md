# Monte Carlo End-to-End Validation

## Design

- Scenarios: nominal, moderate, severe
- Runs per scenario: 6
- Total full pipelines: 18
- Rows per run: 800
- Independent child seeds from one reproducible base seed
- Models: TF-IDF Logistic, held-out temperature-scaled Logistic, LightGBM auxiliary task
- Full flow per run: generation, source/data checks, four-way split, training, calibration, selection, untouched test, evidence evaluation, drift, telemetry, and launch gate

## Completion

```text
expected: 18
completed: 18
failed: 0
sensitivity checks: PASS
```

## Results

| Scenario | PASS | REVIEW | BLOCK | Mean macro-F1 | Mean log loss | Mean max PSI | Mean task completion |
|---|---:|---:|---:|---:|---:|---:|---:|
| nominal | 100.0% | 0.0% | 0.0% | 0.9029 | 0.3398 | 0.0381 | 0.9639 |
| moderate | 0.0% | 66.7% | 33.3% | 0.6925 | 0.8664 | 0.3391 | 0.8806 |
| severe | 0.0% | 0.0% | 100.0% | 0.4118 | 1.4514 | 1.2441 | 0.1708 |

Additional monotonic trends:

- PR-AUC: 0.9537 -> 0.7889 -> 0.4537
- Worst-slice F1: 0.6883 -> 0.4918 -> 0.2114
- Evidence Recall@5: 0.9875 -> 0.8958 -> 0.7931
- Unsupported-claim rate: 0.0125 -> 0.1042 -> 0.2069
- Human-review rate: 0.0806 -> 0.1917 -> 0.8792
- p95 latency: 1715 ms -> 3038 ms -> 5673 ms

## Champion stability

| Scenario | Champion | Count | Rate |
|---|---|---:|---:|
| nominal | `tfidf_logistic_temperature_scaled` | 6 | 100.0% |
| moderate | `tfidf_logistic_temperature_scaled` | 6 | 100.0% |
| severe | `tfidf_logistic_temperature_scaled` | 5 | 83.3% |
| severe | `tfidf_logistic` | 1 | 16.7% |

## Interpretation

The framework is directionally sensitive rather than merely runnable. Predictive, slice, evidence, and task-completion quality decline as designed stress increases. Probability loss, drift, latency, unsupported claims, regression flags, and review demand increase. Nominal runs avoid false blocks, moderate runs produce mixed escalation, and severe runs consistently block.

ECE is not required to worsen monotonically by itself because an uninformative low-confidence model can appear superficially calibrated. The framework therefore evaluates ECE together with Brier score and log loss.

## Reproduction

```powershell
python scripts\run_monte_carlo.py `
  --runs-per-scenario 6 `
  --sample-size 800 `
  --enable-lightgbm `
  --jobs 3
```

Run signatures prevent stale resume reuse after source, configuration, sample-size, seed, or model-flag changes.
