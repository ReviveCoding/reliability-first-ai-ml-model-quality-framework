# Reproducibility

## Environment

- Python virtual environment: `.venv`
- GPU path: CUDA-enabled PyTorch
- Transformer: `distilbert-base-uncased`
- Fixed seeds: `7, 17, 27`
- Frozen input SHA-256: `59648e6c775a54bf6f3d8c5f2b63cb8948f2bd270018d2619ff410c46414de66`
- Frozen input rows: `19996`

## Primary commands

Full end-to-end run:

```powershell
.\run_end_to_end_model_quality.ps1 -Mode Full -SkipTests
```

Three-seed baseline stability audit:

```powershell
.\05_run_transformer_3seed_stability_audit.ps1
```

Class-weighted challenger:

```powershell
.\06_run_class_weighted_transformer_3seed_challenger.ps1 `
  -BaselineAuditRoot ".\sa\s_20260612_234110"
```

Final documentation and cleanup:

```powershell
.\07_finalize_model_decision_and_github_cleanup.ps1
```

## Split roles

| Split | Role |
|---|---|
| Train | Parameter fitting and train-only class-weight calculation |
| Calibration | Best-checkpoint selection |
| Selection | Model/challenger selection |
| Test | Post-selection confirmation only |

The test split is not used for checkpoint selection, framework model
selection, or reselection.
