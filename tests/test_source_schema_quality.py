import pandas as pd

from model_quality.ingestion.load_cfpb import _validate_archive_members, normalize_cfpb
from model_quality.validation.quality_metrics import data_quality_report


def test_missing_source_column_is_not_hidden_by_normalization():
    raw = pd.DataFrame({
        'Product': ['Credit card'],
        # Issue intentionally absent.
        'Consumer complaint narrative': ['This is a sufficiently long complaint narrative for testing.'],
        'Company response to consumer': ['Closed with explanation'],
        'Timely response?': ['Yes'],
        'Date received': ['2024-01-01'],
        'State': ['NJ'],
    })
    normalized = normalize_cfpb(raw)
    _, metrics = data_quality_report(
        normalized,
        [
            'product', 'issue', 'consumer_complaint_narrative',
            'company_response_to_consumer', 'timely_response',
            'date_received', 'state',
        ],
    )
    assert 'issue' in normalized.columns
    assert metrics['required_column_coverage'] < 1.0


def test_archive_member_path_validation_rejects_traversal():
    _validate_archive_members(['folder/data.csv'])
    try:
        _validate_archive_members(['../escape.csv'])
    except ValueError:
        pass
    else:
        raise AssertionError('Expected parent traversal to be rejected.')
