from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split


def temporal_split(
    df: pd.DataFrame,
    date_col: str = 'date_received',
    train_frac: float = 0.7,
    val_frac: float = 0.15,
):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce', format='mixed')
    # Invalid dates are measured by the data-quality layer but excluded from a
    # temporal split; silently sorting NaT to the end would contaminate test.
    df = df[df[date_col].notna()].sort_values(date_col).reset_index(drop=True)
    n = len(df)
    if n < 10:
        raise ValueError(f'Not enough valid dated rows for temporal splitting: {n}.')
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    if n_train <= 0 or n_val < 2 or n_train + n_val >= n:
        raise ValueError(f'Invalid temporal split sizes for n={n}, train_frac={train_frac}, val_frac={val_frac}.')
    train = df.iloc[:n_train].copy()
    val = df.iloc[n_train:n_train+n_val].copy()
    test = df.iloc[n_train+n_val:].copy()
    return train, val, test


def split_validation_window(
    validation: pd.DataFrame,
    date_col: str = 'date_received',
    calibration_fraction: float = 0.5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the validation window into calibration and model-selection sets.

    Temperature scaling is fitted on the earlier calibration window. Champion
    selection is performed on the later selection window. The final test window
    remains untouched until a model has been selected.
    """
    if not 0.2 <= calibration_fraction <= 0.8:
        raise ValueError('calibration_fraction must be between 0.2 and 0.8.')
    frame = validation.copy()
    if date_col in frame.columns:
        frame[date_col] = pd.to_datetime(frame[date_col], errors='coerce', format='mixed')
        frame = frame.sort_values(date_col).reset_index(drop=True)
    if len(frame) < 4:
        raise ValueError('Validation window needs at least 4 rows for calibration/selection splitting.')
    cut = int(round(len(frame) * calibration_fraction))
    cut = min(max(cut, 2), len(frame) - 2)
    return frame.iloc[:cut].copy(), frame.iloc[cut:].copy()


def random_split(df: pd.DataFrame, target_col: str, random_state: int = 7):
    train, tmp = train_test_split(
        df,
        test_size=0.3,
        random_state=random_state,
        stratify=df[target_col] if df[target_col].nunique() > 1 else None,
    )
    val, test = train_test_split(
        tmp,
        test_size=0.5,
        random_state=random_state,
        stratify=tmp[target_col] if tmp[target_col].nunique() > 1 else None,
    )
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)
