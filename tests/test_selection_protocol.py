import pandas as pd

from model_quality.features.split_builder import split_validation_window
from model_quality.validation.split_integrity import split_integrity_report


def _frame(start: str, n: int):
    return pd.DataFrame({
        'date_received': pd.date_range(start, periods=n, freq='D'),
        'consumer_complaint_narrative': [f'narrative {i} with enough text' for i in range(n)],
        'product': ['A', 'B'] * (n // 2) + (['A'] if n % 2 else []),
        'issue': ['I'] * n,
    })


def test_calibration_selection_split_preserves_temporal_order():
    val = _frame('2024-02-01', 10)
    calibration, selection = split_validation_window(val)
    assert calibration['date_received'].max() <= selection['date_received'].min()
    assert len(calibration) == 5
    assert len(selection) == 5


def test_four_way_integrity_report_detects_valid_order():
    train = _frame('2024-01-01', 10)
    calibration = _frame('2024-02-01', 4)
    selection = _frame('2024-03-01', 4)
    test = _frame('2024-04-01', 6)
    _, metrics = split_integrity_report(
        train,
        selection,
        test,
        calibration=calibration,
        target_cols=['product'],
    )
    assert metrics['temporal_order_valid'] == 1.0
    assert metrics['no_overlap'] == 1.0
    assert metrics['calibration_rows'] == 4
    assert metrics['selection_rows'] == 4
