import pandas as pd

from model_quality.validation.split_integrity import split_integrity_report


def test_split_integrity_detects_temporal_order_and_coverage():
    train = pd.DataFrame({
        'consumer_complaint_narrative': ['a long text 1', 'a long text 2'],
        'product': ['A', 'B'], 'issue': ['x', 'y'],
        'date_received': pd.to_datetime(['2024-01-01', '2024-01-02'])
    })
    val = pd.DataFrame({
        'consumer_complaint_narrative': ['a long text 3'],
        'product': ['A'], 'issue': ['x'],
        'date_received': pd.to_datetime(['2024-01-03'])
    })
    test = pd.DataFrame({
        'consumer_complaint_narrative': ['a long text 4', 'a long text 5'],
        'product': ['A', 'B'], 'issue': ['x', 'y'],
        'date_received': pd.to_datetime(['2024-01-04', '2024-01-05'])
    })
    report, metrics = split_integrity_report(train, val, test, target_cols=['product'])
    assert metrics['no_overlap'] == 1.0
    assert metrics['temporal_order_valid'] == 1.0
    assert metrics['product_test_class_coverage'] == 1.0
    assert not report.empty
