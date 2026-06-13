from __future__ import annotations

import hashlib
from itertools import combinations
from typing import Iterable

import pandas as pd


def _row_fingerprint(df: pd.DataFrame, columns: Iterable[str]) -> pd.Series:
    cols = [c for c in columns if c in df.columns]
    if not cols:
        return pd.Series([], dtype=str)
    joined = df[cols].fillna('').astype(str).agg('||'.join, axis=1)
    return joined.map(lambda x: hashlib.sha256(x.encode('utf-8')).hexdigest())


def _normalized_nonempty_values(frame: pd.DataFrame, column: str) -> set[str]:
    values = frame[column].dropna().astype(str).str.strip()
    return set(values[values.ne('')])


def _pairwise_overlap_count(value_sets: dict[str, set[str]]) -> int:
    count = 0
    for left, right in combinations(value_sets, 2):
        count += len(value_sets[left] & value_sets[right])
    return count


def split_integrity_report(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    date_col: str = 'date_received',
    target_cols: list[str] | None = None,
    fingerprint_cols: list[str] | None = None,
    calibration: pd.DataFrame | None = None,
    record_id_candidates: list[str] | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Audit temporal model-development splits.

    Record-identity overlap is the hard leakage check whenever a stable record
    identifier is available in every split. Exact content overlap is reported
    separately as a diagnostic because distinct records can legitimately share
    templated or repeated text.

    If no stable ID exists in every split, content fingerprints are used as the
    conservative fallback hard check.
    """
    target_cols = target_cols or ['product']
    fingerprint_cols = fingerprint_cols or [
        'consumer_complaint_narrative',
        'product_raw',
        'product',
        'issue',
        'date_received',
    ]
    record_id_candidates = record_id_candidates or [
        'complaint_id',
        'record_id',
        'row_id',
    ]

    split_frames: list[tuple[str, pd.DataFrame]] = [('train', train)]
    if calibration is not None:
        split_frames.append(('calibration', calibration))
        selection_name = 'selection'
    else:
        selection_name = 'validation'
    split_frames.extend([(selection_name, validation), ('test', test)])

    rows: list[dict] = []
    sizes = {name: len(frame) for name, frame in split_frames}
    total = max(1, sum(sizes.values()))

    for split, n in sizes.items():
        rows.append({
            'check': 'split_size',
            'split': split,
            'passed': n > 0,
            'metric': n,
            'detail': f'{split}_rows={n}',
        })

    common_id_col = next(
        (
            candidate
            for candidate in record_id_candidates
            if all(candidate in frame.columns for _, frame in split_frames)
        ),
        None,
    )

    fingerprints = {
        name: set(_row_fingerprint(frame, fingerprint_cols))
        for name, frame in split_frames
    }
    content_overlap_count = _pairwise_overlap_count(fingerprints)

    if common_id_col is not None:
        identity_sets = {
            name: _normalized_nonempty_values(frame, common_id_col)
            for name, frame in split_frames
        }
        record_id_overlap_count = _pairwise_overlap_count(identity_sets)
        no_overlap = record_id_overlap_count == 0
        overlap_detail = (
            f'identity_key={common_id_col}; '
            f'record_id_overlap_count={record_id_overlap_count}; '
            f'content_overlap_count={content_overlap_count}'
        )
    else:
        record_id_overlap_count = None
        no_overlap = content_overlap_count == 0
        overlap_detail = (
            'identity_key=content_fingerprint_fallback; '
            f'content_overlap_count={content_overlap_count}'
        )

    # Backward-compatible hard gate name. Its semantics are now record identity
    # when a stable ID is available.
    rows.append({
        'check': 'split_overlap',
        'split': '*',
        'passed': no_overlap,
        'metric': 0.0 if no_overlap else (
            record_id_overlap_count
            if record_id_overlap_count is not None
            else content_overlap_count
        ),
        'detail': overlap_detail,
    })

    # Diagnostic only: repeated content with distinct record IDs is useful to
    # monitor, but is not automatically record leakage.
    rows.append({
        'check': 'split_content_overlap',
        'split': '*',
        'passed': content_overlap_count == 0,
        'metric': float(content_overlap_count),
        'detail': (
            f'content_overlap_count={content_overlap_count}; '
            f'fingerprint_cols={fingerprint_cols}'
        ),
    })

    temporal_order_valid = True
    temporal_parts: list[str] = []
    previous_max = None
    for name, frame in split_frames:
        if date_col not in frame.columns:
            temporal_order_valid = False
            temporal_parts.append(f'{name}:missing_date_column')
            continue
        dates = pd.to_datetime(frame[date_col], errors='coerce')
        current_min, current_max = dates.min(), dates.max()
        temporal_parts.append(f'{name}_min={current_min}; {name}_max={current_max}')
        if pd.isna(current_min) or pd.isna(current_max):
            temporal_order_valid = False
        if previous_max is not None and pd.notna(current_min) and previous_max > current_min:
            temporal_order_valid = False
        previous_max = current_max

    rows.append({
        'check': 'temporal_order',
        'split': '*',
        'passed': temporal_order_valid,
        'metric': 1.0 if temporal_order_valid else 0.0,
        'detail': '; '.join(temporal_parts),
    })

    coverage_metrics: dict[str, float] = {}
    for target in target_cols:
        if target not in train.columns:
            continue
        train_classes = set(train[target].dropna().astype(str).unique())
        denom = max(1, len(train_classes))
        for split_name, split_df in split_frames[1:]:
            if target not in split_df.columns:
                cov = 0.0
            else:
                split_classes = set(split_df[target].dropna().astype(str).unique())
                cov = len(train_classes & split_classes) / denom
            coverage_metrics[f'{target}_{split_name}_class_coverage'] = float(cov)
            rows.append({
                'check': 'target_class_coverage',
                'split': split_name,
                'passed': cov >= 0.80,
                'metric': float(cov),
                'detail': f'target={target}; coverage={cov:.4f}',
            })

    eval_sizes = [sizes[name] / total for name in sizes if name != 'train']
    metrics = {
        'train_rows': int(sizes['train']),
        'validation_rows': int(sizes[selection_name]),
        'selection_rows': int(sizes[selection_name]),
        'test_rows': int(sizes['test']),
        'no_overlap': float(no_overlap),
        'record_id_no_overlap': float(no_overlap) if common_id_col is not None else None,
        'record_id_overlap_count': (
            int(record_id_overlap_count)
            if record_id_overlap_count is not None
            else None
        ),
        'content_overlap_count': int(content_overlap_count),
        'overlap_identity_key': common_id_col or 'content_fingerprint_fallback',
        'temporal_order_valid': float(temporal_order_valid),
        'min_eval_fraction': float(min(eval_sizes)) if eval_sizes else 0.0,
        **coverage_metrics,
    }
    if calibration is not None:
        metrics['calibration_rows'] = int(sizes['calibration'])

    return pd.DataFrame(rows), metrics