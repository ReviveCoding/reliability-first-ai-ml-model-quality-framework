from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from model_quality.ingestion.load_cfpb import (
    _duckdb_reservoir_sample_csv,
    discover_cfpb_path,
    load_cfpb,
)


def _public_frame(rows: int = 80) -> pd.DataFrame:
    products = ['Credit card', 'Mortgage', 'Credit reporting']
    issues = ['Billing issue', 'Payment issue', 'Incorrect information']
    return pd.DataFrame({
        'Date received': pd.date_range('2024-01-01', periods=rows, freq='D').strftime('%m/%d/%y'),
        'Product': [products[i % len(products)] for i in range(rows)],
        'Issue': [issues[i % len(issues)] for i in range(rows)],
        'Consumer complaint narrative': [
            f'This is a detailed complaint narrative number {i} with enough text for validation.'
            for i in range(rows)
        ],
        'Company response to consumer': ['Closed with explanation'] * rows,
        'Timely response?': ['Yes' if i % 7 else 'No' for i in range(rows)],
        'State': ['NJ' if i % 2 else 'NY' for i in range(rows)],
        'Submitted via': ['Web'] * rows,
        'Company': ['Example Company'] * rows,
    })


def test_discover_exactly_one_supported_dataset(tmp_path: Path):
    raw = tmp_path / 'raw'
    raw.mkdir()
    expected = raw / 'complaints.csv'
    _public_frame().to_csv(expected, index=False)
    assert discover_cfpb_path(raw) == expected


def test_discover_rejects_zero_or_multiple_datasets(tmp_path: Path):
    raw = tmp_path / 'raw'
    raw.mkdir()
    with pytest.raises(FileNotFoundError):
        discover_cfpb_path(raw)
    _public_frame().to_csv(raw / 'a.csv', index=False)
    _public_frame().to_csv(raw / 'b.csv', index=False)
    with pytest.raises(ValueError, match='Multiple supported datasets'):
        discover_cfpb_path(raw)


def test_real_path_is_required_without_synthetic_flag():
    with pytest.raises(ValueError, match='CFPB path is required'):
        load_cfpb(None, use_synthetic=False)


def test_duckdb_sampler_reads_public_schema(tmp_path: Path):
    path = tmp_path / 'complaints.csv'
    _public_frame(120).to_csv(path, index=False)
    sample = _duckdb_reservoir_sample_csv(path, sample_size=30, random_state=11)
    assert len(sample) == 30
    assert sample.attrs['effective_sampling_strategy'] == 'duckdb'
    assert sample['consumer_complaint_narrative'].str.len().gt(20).all()
