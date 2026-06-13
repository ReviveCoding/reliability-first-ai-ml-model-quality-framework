# Contributing

Contributions that improve reproducibility, evaluation quality, failure analysis, documentation, or test coverage are welcome.

## Development setup

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -r requirements-ci.txt
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

## Required checks

Before opening a pull request, run:

```bash
python -m ruff check .
python -m pytest -q
python scripts/verify_release_contract.py --repo .
```

## Evaluation and claim discipline

- Do not use the final test split for checkpoint selection, framework selection, threshold tuning, or reselection.
- Preserve calibration, worst-slice, multi-seed, and provenance evidence when changing model-selection logic.
- Label synthetic, public, proxy, and offline data accurately.
- Do not describe this repository as a production deployment, formal certification, or use of proprietary company data.
- Update relevant evidence and decision artifacts when changing reported metrics.

## Pull requests

Keep pull requests focused. Explain the problem, implementation, validation performed, and any change to model decisions or claim boundaries.
