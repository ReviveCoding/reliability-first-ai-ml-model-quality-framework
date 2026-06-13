from __future__ import annotations

import pandas as pd
from sklearn.metrics import f1_score

from .label_alignment import known_label_mask


def worst_slice_f1(model, df: pd.DataFrame, target_col: str, slice_col: str = 'state', min_count: int = 10) -> dict:
    rows = []
    for key, group in df.groupby(slice_col, dropna=False):
        mask, unknown_rate = known_label_mask(model, group, target_col)
        evaluated = group.loc[mask]
        if len(evaluated) < min_count:
            continue
        y = evaluated[target_col].astype(str).values
        pred = model.predict(evaluated)
        rows.append({
            'slice_col': slice_col,
            'slice': key,
            'n': int(len(group)),
            'evaluated_n': int(len(evaluated)),
            'unknown_target_rate': unknown_rate,
            'macro_f1': float(f1_score(y, pred, average='macro', zero_division=0)),
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return {'worst_slice_f1': None, 'slice_metrics': out}
    return {'worst_slice_f1': float(out['macro_f1'].min()), 'slice_metrics': out}
