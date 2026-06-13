from pathlib import Path

import pandas as pd

from model_quality.features.split_builder import temporal_split
from model_quality.ingestion.load_cfpb import load_cfpb


def _public_frame(n=100):
    return pd.DataFrame({
        'Product': ['Credit card'] * n,
        'Issue': ['Incorrect information'] * n,
        'Consumer complaint narrative': [f'Complaint narrative number {i} with enough detail for testing.' for i in range(n)],
        'Company response to consumer': ['Closed with explanation'] * n,
        'Timely response?': ['Yes'] * n,
        'Date received': pd.date_range('2024-01-01', periods=n).astype(str),
        'State': ['NJ'] * n,
    })


def test_reservoir_sampling_is_deterministic_and_not_head_only(tmp_path: Path):
    path = tmp_path / 'complaints.csv'
    _public_frame().to_csv(path, index=False)
    a = load_cfpb(path, sample_size=12, random_state=11, sampling_strategy='reservoir', source_chunksize=20)
    b = load_cfpb(path, sample_size=12, random_state=11, sampling_strategy='reservoir', source_chunksize=20)
    assert a['consumer_complaint_narrative'].tolist() == b['consumer_complaint_narrative'].tolist()
    sampled_ids = {int(text.split('number ')[1].split(' ')[0]) for text in a['consumer_complaint_narrative']}
    assert sampled_ids != set(range(12))
    assert max(sampled_ids) > 20


def test_temporal_split_excludes_invalid_dates():
    frame = _public_frame(30).rename(columns={
        'Product': 'product', 'Issue': 'issue', 'Consumer complaint narrative': 'consumer_complaint_narrative',
        'Company response to consumer': 'company_response_to_consumer', 'Timely response?': 'timely_response',
        'Date received': 'date_received', 'State': 'state',
    })
    frame.loc[0, 'date_received'] = 'not-a-date'
    train, val, test = temporal_split(frame)
    combined = pd.concat([train, val, test])
    assert combined['date_received'].notna().all()
    assert len(combined) == 29
