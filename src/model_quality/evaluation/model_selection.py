from __future__ import annotations

import pandas as pd


def select_champion_model(
    leaderboard: pd.DataFrame,
    primary_target: str = 'product',
    min_macro_f1: float = 0.60,
    min_pr_auc: float = 0.50,
) -> dict:
    """Select a champion only among models evaluated on the same primary task.

    Comparing a product classifier with a timely-response model directly is not
    statistically meaningful. The primary target is therefore filtered first,
    then calibration, slice stability, and predictive quality are considered.
    """
    if leaderboard is None or leaderboard.empty:
        return {'champion_model': None, 'reason': 'No models were evaluated.'}
    df = leaderboard.copy()
    if 'target_col' in df.columns and (df['target_col'].astype(str) == primary_target).any():
        df = df[df['target_col'].astype(str) == primary_target].copy()
        task_note = f'Primary task restricted to target_col={primary_target!r}. '
    else:
        task_note = f'No rows matched primary target {primary_target!r}; used all available models. '

    defaults = {
        'macro_f1': -1.0,
        'pr_auc': -1.0,
        'ece': 999.0,
        'brier': 999.0,
        'log_loss': 999.0,
        'worst_slice_f1': -1.0,
        'unknown_target_rate': 1.0,
    }
    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = default
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(default)

    candidates = df[(df['macro_f1'] >= min_macro_f1) & (df['pr_auc'] >= min_pr_auc)].copy()
    if candidates.empty:
        candidates = df.sort_values(
            ['macro_f1', 'pr_auc', 'worst_slice_f1', 'log_loss', 'ece', 'brier'],
            ascending=[False, False, False, True, True, True],
        )
        reason = task_note + 'No model met predictive floors; selected best available model.'
    else:
        candidates = candidates.sort_values(
            ['unknown_target_rate', 'log_loss', 'ece', 'brier', 'worst_slice_f1', 'macro_f1', 'pr_auc'],
            ascending=[True, True, True, True, False, False, False],
        )
        reason = task_note + 'Selected among models passing predictive floors by target coverage, log loss, ECE, Brier, slice stability, macro-F1, and PR-AUC.'

    row = candidates.iloc[0].to_dict()
    def num(name):
        value = row.get(name)
        return float(value) if value is not None and pd.notna(value) else None
    return {
        'champion_model': row.get('model'),
        'target_col': row.get('target_col'),
        'macro_f1': num('macro_f1'),
        'pr_auc': num('pr_auc'),
        'ece': num('ece'),
        'brier': num('brier'),
        'log_loss': num('log_loss'),
        'worst_slice_f1': num('worst_slice_f1'),
        'unknown_target_rate': num('unknown_target_rate'),
        'reason': reason,
    }
