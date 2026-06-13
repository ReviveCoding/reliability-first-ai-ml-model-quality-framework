from __future__ import annotations

from typing import Any

import pandas as pd


def fit_label_policy(
    train_df: pd.DataFrame,
    target_col: str = 'product',
    min_count: int = 20,
    other_label: str = 'Other',
) -> dict[str, Any]:
    """Fit a rare-label policy using the training window only.

    Fitting on the full dataset would leak future label frequencies into model
    development. The returned policy is therefore derived exclusively from the
    training split and can then be applied unchanged to calibration, selection,
    and test windows.
    """
    if target_col not in train_df.columns:
        raise KeyError(f'Missing target column: {target_col}')
    labels = train_df[target_col].astype(str)
    counts = labels.value_counts(dropna=False).sort_index()
    if min_count <= 1:
        rare: list[str] = []
    else:
        rare = sorted(counts[counts < int(min_count)].index.astype(str).tolist())
    retained = sorted(counts[counts >= int(min_count)].index.astype(str).tolist())
    return {
        'policy_version': 2,
        'fit_scope': 'training_only',
        'target_col': target_col,
        'min_count': int(min_count),
        'other_label': other_label,
        'train_label_counts': {str(k): int(v) for k, v in counts.items()},
        'retained_labels': retained,
        'collapsed_labels': rare,
        'collapsed_train_row_count': int(labels.isin(rare).sum()),
    }


def apply_label_policy(
    df: pd.DataFrame,
    policy: dict[str, Any],
    *,
    preserve_raw: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply a training-fitted label policy without hiding novel labels.

    Labels that were rare in training are mapped to ``Other``. Labels never
    seen during training are left unchanged so downstream evaluation can report
    an explicit unknown/novel-target rate instead of silently folding future
    classes into a known bucket.
    """
    target_col = str(policy['target_col'])
    if target_col not in df.columns:
        raise KeyError(f'Missing target column: {target_col}')
    out = df.copy()
    raw_col = f'{target_col}_raw'
    if preserve_raw and raw_col not in out.columns:
        out[raw_col] = out[target_col].astype(str)

    labels = out[target_col].astype(str)
    rare = set(map(str, policy.get('collapsed_labels', [])))
    known_train = set(map(str, policy.get('train_label_counts', {}).keys()))
    other_label = str(policy.get('other_label', 'Other'))

    collapsed_mask = labels.isin(rare)
    novel_mask = ~labels.isin(known_train)
    out.loc[collapsed_mask, target_col] = other_label

    stats = {
        'row_count': int(len(out)),
        'collapsed_row_count': int(collapsed_mask.sum()),
        'novel_target_row_count': int(novel_mask.sum()),
        'novel_target_rate': float(novel_mask.mean()) if len(out) else 0.0,
        'novel_labels': sorted(labels[novel_mask].unique().astype(str).tolist()),
    }
    return out, stats


def consolidate_rare_labels(
    df: pd.DataFrame,
    target_col: str = 'product',
    min_count: int = 20,
    other_label: str = 'Other',
) -> tuple[pd.DataFrame, dict]:
    """Backward-compatible single-frame helper.

    New pipeline code should call :func:`fit_label_policy` on the training split
    and :func:`apply_label_policy` on every split to avoid temporal leakage.
    """
    policy = fit_label_policy(df, target_col=target_col, min_count=min_count, other_label=other_label)
    out, stats = apply_label_policy(df, policy)
    merged = {**policy, **stats, 'collapsed_row_count': stats['collapsed_row_count']}
    return out, merged
