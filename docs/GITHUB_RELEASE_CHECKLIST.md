# GitHub Release Checklist

## Required

- [ ] `python -m ruff check .` passes
- [ ] `python -m pytest -q` passes
- [ ] `git diff --check` passes
- [ ] No secrets, tokens, or local absolute paths are committed
- [ ] Public datasets and generated model artifacts are ignored
- [ ] No tracked file approaches GitHub's 100 MB hard file limit
- [ ] README links to the final decision and reproducibility documents
- [ ] Final evidence bundle is present under `docs/artifacts/final_evaluation`
- [ ] Claims are labeled local/offline/public-data/portfolio evidence
- [ ] Weighted challenger is documented as rejected, not silently removed

## Recommended repository contents

Commit source code, tests, configuration, scripts, compact reports, and
small canonical CSV/JSON evidence. Do not commit local virtual
environments, raw datasets, checkpoints, run directories, MLflow stores,
or large generated outputs.
