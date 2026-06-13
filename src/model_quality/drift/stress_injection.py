from __future__ import annotations

import numpy as np
import pandas as pd


def inject_missingness(df: pd.DataFrame, column: str, rate: float = 0.15, random_state: int = 7) -> pd.DataFrame:
    out = df.copy()
    rng = np.random.default_rng(random_state)
    if column not in out.columns:
        return out
    mask = rng.random(len(out)) < float(np.clip(rate, 0.0, 1.0))
    out.loc[mask, column] = None
    return out


def inject_feature_drift(
    df: pd.DataFrame,
    column: str,
    rate: float = 0.20,
    random_state: int = 7,
    target_value: object | None = None,
) -> pd.DataFrame:
    """Inject a real categorical distribution shift.

    Earlier versions re-sampled from the original support and often preserved
    the marginal distribution. This implementation moves selected rows toward a
    target category, which creates a detectable population shift.
    """
    out = df.copy()
    rng = np.random.default_rng(random_state)
    if column not in out.columns or out.empty:
        return out
    values = pd.Series(out[column]).dropna().astype(str)
    if values.nunique() < 2:
        return out
    if target_value is None:
        # Prefer a minority category so the resulting shift is not trivially the
        # same as the reference distribution's dominant category.
        target_value = values.value_counts().sort_values().index[0]
    mask = rng.random(len(out)) < float(np.clip(rate, 0.0, 1.0))
    out.loc[mask, column] = target_value
    return out


def inject_numeric_shift(
    df: pd.DataFrame,
    column: str,
    location_shift: float = 0.0,
    scale_multiplier: float = 1.0,
) -> pd.DataFrame:
    out = df.copy()
    if column not in out.columns:
        return out
    x = pd.to_numeric(out[column], errors='coerce')
    out[column] = x * float(scale_multiplier) + float(location_shift)
    return out


def simulate_training_serving_skew(reference: pd.DataFrame, serving: pd.DataFrame) -> dict:
    ref_cols = set(reference.columns)
    serving_cols = set(serving.columns)
    missing = sorted(ref_cols - serving_cols)
    extra = sorted(serving_cols - ref_cols)
    mismatch_rate = len(missing) / max(1, len(ref_cols))
    return {'missing_columns': missing, 'extra_columns': extra, 'training_serving_skew': float(mismatch_rate)}
