from __future__ import annotations

import pandas as pd

MISSING_LIKE_VALUES = {'', 'unknown', 'none', 'nan', 'na', 'n/a', 'null'}


def missing_like_rate(series: pd.Series) -> float:
    """Treat nulls and common placeholder values as missing-like.

    CFPB-style loaders often need placeholder values for modeling stability, but
    model-quality gates should still recognize placeholders such as "Unknown" as
    data-quality risk signals.
    """
    if series.empty:
        return 1.0
    s = series.astype('string').str.strip().str.lower()
    return float(series.isna().mean() + ((~series.isna()) & s.isin(MISSING_LIKE_VALUES)).mean())


def valid_date_rate(series: pd.Series) -> float:
    if series.empty:
        return 0.0
    parsed = pd.to_datetime(series, errors='coerce')
    return float(parsed.notna().mean())


def data_quality_report(df: pd.DataFrame, required_columns: list[str], label_columns: list[str] | None = None) -> tuple[pd.DataFrame, dict]:
    label_columns = label_columns or []
    rows=[]
    missing_required=[]
    source_present = set(df.attrs.get('source_required_columns_present', required_columns if set(required_columns).issubset(df.columns) else []))
    for col in required_columns:
        exists = col in source_present if 'source_required_columns_present' in df.attrs else col in df.columns
        miss_rate = missing_like_rate(df[col]) if exists else 1.0
        rows.append({'check':'required_column', 'column':col, 'passed':exists and miss_rate < 1.0, 'metric':1.0-miss_rate, 'detail':f'missing_like_rate={miss_rate:.4f}'})
        if not exists:
            missing_required.append(col)

    duplicate_rate = float(df.duplicated().mean()) if len(df) else 0.0
    rows.append({'check':'duplicate_rate', 'column':'*', 'passed':duplicate_rate <= 0.01, 'metric':duplicate_rate, 'detail':f'duplicate_rate={duplicate_rate:.4f}'})

    if set(required_columns).issubset(df.columns):
        completeness = float(pd.Series([1.0 - missing_like_rate(df[c]) for c in required_columns]).mean())
    else:
        completeness = 0.0
    rows.append({'check':'completeness', 'column':'*', 'passed':completeness >= 0.98, 'metric':completeness, 'detail':f'missing-like-aware completeness={completeness:.4f}'})

    label_missing_rates = []
    for col in label_columns:
        if col in df.columns:
            miss = missing_like_rate(df[col])
            label_missing_rates.append(miss)
            rows.append({'check':'label_missing_rate', 'column':col, 'passed':miss <= 0.02, 'metric':miss, 'detail':f'label_missing_like_rate={miss:.4f}'})
    label_missing_rate = float(max(label_missing_rates)) if label_missing_rates else 0.0

    if 'date_received' in df.columns:
        date_validity = valid_date_rate(df['date_received'])
        rows.append({'check':'date_validity', 'column':'date_received', 'passed':date_validity >= 0.98, 'metric':date_validity, 'detail':f'valid_date_rate={date_validity:.4f}'})
    else:
        date_validity = 0.0

    if 'consumer_complaint_narrative' in df.columns:
        text = df['consumer_complaint_narrative'].fillna('').astype(str)
        evidence_coverage = float((text.str.len() > 50).mean())
    else:
        evidence_coverage = 0.0
    rows.append({'check':'evidence_coverage', 'column':'consumer_complaint_narrative', 'passed':evidence_coverage >= 0.90, 'metric':evidence_coverage, 'detail':f'evidence_coverage={evidence_coverage:.4f}'})

    metrics = {
        'row_count': int(len(df)),
        'required_column_coverage': float(1 - len(missing_required) / max(1, len(required_columns))),
        'duplicate_rate': duplicate_rate,
        'completeness': completeness,
        'label_missing_rate': label_missing_rate,
        'date_validity': date_validity,
        'evidence_coverage': evidence_coverage,
    }
    return pd.DataFrame(rows), metrics
