import pandas as pd

from model_quality.evaluation.calibration_metrics import calibration_report
from model_quality.evaluation.classification_metrics import classification_report_dict
from model_quality.evaluation.slice_metrics import worst_slice_f1
from model_quality.models.train_logistic import train_tfidf_logistic


def _train_df():
    return pd.DataFrame({
        'consumer_complaint_narrative': ['card fee', 'card charge', 'loan payment', 'loan balance'] * 4,
        'product': ['Card', 'Card', 'Loan', 'Loan'] * 4,
        'state': ['NJ', 'NY', 'NJ', 'NY'] * 4,
    })


def test_metrics_report_unseen_temporal_labels_without_crashing():
    model = train_tfidf_logistic(_train_df(), random_state=1)
    test = pd.DataFrame({
        'consumer_complaint_narrative': ['card dispute', 'crypto transfer', 'loan issue', 'crypto wallet'],
        'product': ['Card', 'Crypto', 'Loan', 'Crypto'],
        'state': ['NJ', 'NJ', 'NY', 'NY'],
    })
    metrics = classification_report_dict(model, test, 'product')
    calibration, _ = calibration_report(model, test, 'product')
    slices = worst_slice_f1(model, test, 'product', min_count=1)
    assert metrics['evaluated_rows'] == 2
    assert metrics['unknown_target_rate'] == 0.5
    assert calibration['evaluated_rows'] == 2
    assert calibration['unknown_target_rate'] == 0.5
    assert slices['worst_slice_f1'] is not None
