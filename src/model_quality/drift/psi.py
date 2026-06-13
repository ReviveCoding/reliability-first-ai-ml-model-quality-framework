from __future__ import annotations

import numpy as np
import pandas as pd


def psi(expected, actual, buckets=10) -> float:
    expected = pd.Series(expected).dropna()
    actual = pd.Series(actual).dropna()
    if expected.empty or actual.empty:
        return 0.0
    if expected.dtype == object or str(expected.dtype).startswith('category'):
        cats = sorted(set(expected.astype(str)) | set(actual.astype(str)))
        e = expected.astype(str).value_counts(normalize=True).reindex(cats, fill_value=0.0001)
        a = actual.astype(str).value_counts(normalize=True).reindex(cats, fill_value=0.0001)
    else:
        breaks = np.unique(np.quantile(expected, np.linspace(0,1,buckets+1)))
        if len(breaks) < 3:
            return 0.0
        e = pd.cut(expected, breaks, include_lowest=True).value_counts(normalize=True).sort_index().replace(0,0.0001)
        a = pd.cut(actual, breaks, include_lowest=True).value_counts(normalize=True).reindex(e.index, fill_value=0.0001).replace(0,0.0001)
    return float(((a-e)*np.log(a/e)).sum())


def psi_report(reference: pd.DataFrame, current: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame([{'column':c, 'psi':psi(reference[c], current[c])} for c in columns if c in reference.columns and c in current.columns])
