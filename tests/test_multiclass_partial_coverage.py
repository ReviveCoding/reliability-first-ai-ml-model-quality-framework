import pandas as pd

from model_quality.evaluation.classification_metrics import classification_report_dict
from model_quality.models.train_logistic import train_tfidf_logistic


def test_multiclass_auc_and_pr_auc_work_when_holdout_omits_train_class():
    train = pd.DataFrame({
        'consumer_complaint_narrative': ['card fee'] * 10 + ['loan payment'] * 10 + ['transfer delay'] * 10,
        'product': ['Card'] * 10 + ['Loan'] * 10 + ['Transfer'] * 10,
    })
    test = pd.DataFrame({
        'consumer_complaint_narrative': ['card charge'] * 5 + ['loan balance'] * 5,
        'product': ['Card'] * 5 + ['Loan'] * 5,
    })
    model = train_tfidf_logistic(train, random_state=1)
    metrics = classification_report_dict(model, test, 'product')
    assert metrics['evaluation_class_coverage'] == 2 / 3
    assert metrics['roc_auc'] is not None
    assert metrics['pr_auc'] is not None
