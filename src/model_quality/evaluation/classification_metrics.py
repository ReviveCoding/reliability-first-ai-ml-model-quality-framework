from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score

from .label_alignment import known_label_mask


def classification_report_dict(model, df: pd.DataFrame, target_col: str) -> dict:
    mask, unknown_rate = known_label_mask(model, df, target_col)
    eval_df = df.loc[mask].copy()
    out = {
        'model': model.name,
        'target_col': target_col,
        'total_rows': int(len(df)),
        'evaluated_rows': int(len(eval_df)),
        'unknown_target_rate': unknown_rate,
    }
    if eval_df.empty:
        out.update({
            'accuracy': None, 'macro_f1': None, 'roc_auc': None, 'pr_auc': None,
            'evaluation_class_coverage': 0.0,
        })
        return out

    y_true = eval_df[target_col].astype(str).values
    y_pred = model.predict(eval_df)
    proba = model.predict_proba(eval_df)
    labels = list(model.label_encoder.classes_)
    y_int = model.label_encoder.transform(y_true)
    present = np.unique(y_int)
    out.update({
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'macro_f1': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        'evaluation_class_coverage': float(len(present) / max(1, len(labels))),
    })

    if proba is None or len(present) < 2:
        out['roc_auc'] = None
        out['pr_auc'] = None
        return out

    try:
        if len(labels) == 2:
            out['roc_auc'] = float(roc_auc_score(y_int, proba[:, 1]))
            out['pr_auc'] = float(average_precision_score(y_int, proba[:, 1]))
        else:
            # Compute one-vs-rest metrics only for classes that actually appear
            # in the evaluation window. This avoids dropping all multiclass
            # metrics when a temporal holdout omits a rare training class.
            roc_values = []
            ap_values = []
            for class_index in present:
                binary = (y_int == class_index).astype(int)
                if binary.min() == binary.max():
                    continue
                roc_values.append(roc_auc_score(binary, proba[:, class_index]))
                ap_values.append(average_precision_score(binary, proba[:, class_index]))
            out['roc_auc'] = float(np.mean(roc_values)) if roc_values else None
            out['pr_auc'] = float(np.mean(ap_values)) if ap_values else None
    except (ValueError, IndexError):
        out['roc_auc'] = None
        out['pr_auc'] = None
    return out
