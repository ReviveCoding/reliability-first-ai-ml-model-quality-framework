import pandas as pd

from model_quality.validation.split_integrity import split_integrity_report


def _frame(complaint_id: int, text: str, date: str) -> pd.DataFrame:
    return pd.DataFrame({
        'complaint_id': [complaint_id],
        'consumer_complaint_narrative': [text],
        'product_raw': ['A'],
        'product': ['A'],
        'issue': ['x'],
        'date_received': pd.to_datetime([date]),
    })


def test_same_content_with_distinct_record_ids_is_diagnostic_not_hard_overlap():
    train = _frame(1, 'same repeated template text', '2024-01-01')
    validation = _frame(2, 'same repeated template text', '2024-01-01')
    test = _frame(3, 'different text', '2024-01-03')

    report, metrics = split_integrity_report(train, validation, test)

    assert metrics['no_overlap'] == 1.0
    assert metrics['record_id_overlap_count'] == 0
    assert metrics['content_overlap_count'] == 1
    hard_row = report.loc[report['check'] == 'split_overlap'].iloc[0]
    diagnostic_row = report.loc[report['check'] == 'split_content_overlap'].iloc[0]
    assert bool(hard_row['passed']) is True
    assert bool(diagnostic_row['passed']) is False


def test_same_record_id_across_splits_is_hard_overlap():
    train = _frame(1, 'text one', '2024-01-01')
    validation = _frame(1, 'text two', '2024-01-02')
    test = _frame(3, 'text three', '2024-01-03')

    _, metrics = split_integrity_report(train, validation, test)

    assert metrics['no_overlap'] == 0.0
    assert metrics['record_id_overlap_count'] == 1
