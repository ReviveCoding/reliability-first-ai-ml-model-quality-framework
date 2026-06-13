from __future__ import annotations

import pandas as pd
from scipy.stats import ks_2samp


def ks_report(reference: pd.DataFrame, current: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows=[]
    for c in columns:
        if c not in reference.columns or c not in current.columns:
            continue
        if not pd.api.types.is_numeric_dtype(reference[c]):
            continue
        r = reference[c].dropna()
        a = current[c].dropna()
        if len(r) < 2 or len(a) < 2:
            continue
        stat, p = ks_2samp(r, a, method='asymp')
        rows.append({'column':c, 'ks_stat':float(stat), 'ks_pvalue':float(p)})
    return pd.DataFrame(rows)
