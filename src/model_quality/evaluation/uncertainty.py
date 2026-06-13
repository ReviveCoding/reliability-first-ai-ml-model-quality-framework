from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from .label_alignment import known_label_mask


def bootstrap_macro_f1_interval(
    model,
    df: pd.DataFrame,
    target_col: str,
    *,
    n_resamples: int = 500,
    confidence_level: float = 0.95,
    random_state: int = 7,
) -> dict:
    """Estimate a percentile bootstrap interval for macro-F1.

    The interval is intentionally lightweight and deterministic. It quantifies
    finite-holdout uncertainty without turning the project into a full
    statistical inference package.
    """
    mask, unknown_rate = known_label_mask(model, df, target_col)
    eval_df = df.loc[mask].copy()
    if eval_df.empty:
        return {
            'macro_f1_ci_low': None,
            'macro_f1_ci_high': None,
            'bootstrap_resamples': int(n_resamples),
            'evaluated_rows': 0,
            'unknown_target_rate': unknown_rate,
        }
    y_true = eval_df[target_col].astype(str).to_numpy()
    y_pred = np.asarray(model.predict(eval_df)).astype(str)
    fixed_labels = [str(x) for x in getattr(model.label_encoder, 'classes_', sorted(set(y_true) | set(y_pred)))]
    n = len(eval_df)
    if n == 1:
        score = float(f1_score(y_true, y_pred, labels=fixed_labels, average='macro', zero_division=0))
        return {
            'macro_f1_ci_low': score,
            'macro_f1_ci_high': score,
            'bootstrap_resamples': int(n_resamples),
            'evaluated_rows': int(n),
            'unknown_target_rate': unknown_rate,
        }

    rng = np.random.default_rng(random_state)
    scores = np.empty(max(1, int(n_resamples)), dtype=float)
    for i in range(len(scores)):
        idx = rng.integers(0, n, size=n)
        scores[i] = f1_score(y_true[idx], y_pred[idx], labels=fixed_labels, average='macro', zero_division=0)
    alpha = (1.0 - float(confidence_level)) / 2.0
    return {
        'macro_f1_ci_low': float(np.quantile(scores, alpha)),
        'macro_f1_ci_high': float(np.quantile(scores, 1.0 - alpha)),
        'bootstrap_resamples': int(len(scores)),
        'evaluated_rows': int(n),
        'unknown_target_rate': unknown_rate,
    }
