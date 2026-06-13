from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

from .label_alignment import known_label_mask


def expected_calibration_error(y_true_int, proba, n_bins=10):
    conf = np.max(proba, axis=1)
    pred = np.argmax(proba, axis=1)
    acc = (pred == y_true_int).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (conf > lo) & (conf <= hi)
        if mask.sum() == 0:
            continue
        bin_acc = float(acc[mask].mean())
        bin_conf = float(conf[mask].mean())
        weight = float(mask.mean())
        ece += weight * abs(bin_acc - bin_conf)
        rows.append({'bin_low': lo, 'bin_high': hi, 'count': int(mask.sum()), 'accuracy': bin_acc, 'confidence': bin_conf})
    return float(ece), pd.DataFrame(rows)


def calibration_report(model, df: pd.DataFrame, target_col: str) -> tuple[dict, pd.DataFrame]:
    mask, unknown_rate = known_label_mask(model, df, target_col)
    eval_df = df.loc[mask].copy()
    base = {
        'model': model.name,
        'total_rows': int(len(df)),
        'evaluated_rows': int(len(eval_df)),
        'unknown_target_rate': unknown_rate,
    }
    if eval_df.empty:
        return {**base, 'ece': None, 'brier': None, 'log_loss': None}, pd.DataFrame()
    proba = model.predict_proba(eval_df)
    if proba is None:
        return {**base, 'ece': None, 'brier': None, 'log_loss': None}, pd.DataFrame()
    y_true = model.label_encoder.transform(eval_df[target_col].astype(str))
    ece, table = expected_calibration_error(y_true, proba)
    if proba.shape[1] == 2:
        brier = float(brier_score_loss(y_true, proba[:, 1]))
    else:
        onehot = np.eye(proba.shape[1])[y_true]
        brier = float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))
    ll = float(log_loss(y_true, proba, labels=np.arange(proba.shape[1])))
    return {**base, 'ece': ece, 'brier': brier, 'log_loss': ll}, table
