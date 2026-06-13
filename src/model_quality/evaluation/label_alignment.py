from __future__ import annotations

import numpy as np
import pandas as pd


def known_label_mask(model, df: pd.DataFrame, target_col: str) -> tuple[np.ndarray, float]:
    """Return rows whose target labels were observed during training.

    Temporal splits can legitimately contain classes not present in the training
    window. Those rows cannot be transformed by a closed-set classifier's label
    encoder, so they are reported as unknown-target coverage rather than silently
    mapped to an arbitrary known class.
    """
    values = df[target_col].astype(str).to_numpy()
    known = set(map(str, model.label_encoder.classes_))
    mask = np.array([value in known for value in values], dtype=bool)
    unknown_rate = float(1.0 - mask.mean()) if len(mask) else 0.0
    return mask, unknown_rate
