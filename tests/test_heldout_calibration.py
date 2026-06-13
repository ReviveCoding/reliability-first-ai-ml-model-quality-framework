import pandas as pd

from model_quality.models.train_logistic import train_calibrated_tfidf_logistic, train_tfidf_logistic


def test_heldout_calibration_reuses_fitted_vectorizer():
    train = pd.DataFrame({
        'consumer_complaint_narrative': ['card billing fee'] * 12 + ['loan payment balance'] * 12,
        'product': ['Card'] * 12 + ['Loan'] * 12,
    })
    val = pd.DataFrame({
        'consumer_complaint_narrative': ['card charge dispute'] * 4 + ['loan monthly payment'] * 4,
        'product': ['Card'] * 4 + ['Loan'] * 4,
    })
    base = train_tfidf_logistic(train, random_state=1)
    calibrated = train_calibrated_tfidf_logistic(train, calibration_df=val, base_model=base, random_state=1)
    assert calibrated.name == 'tfidf_logistic_temperature_scaled'
    assert calibrated.pipeline.base_pipeline is base.pipeline
    assert calibrated.predict_proba(val).shape == (8, 2)
