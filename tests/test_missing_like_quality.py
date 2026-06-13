import pandas as pd

from model_quality.validation.quality_metrics import data_quality_report, missing_like_rate


def test_missing_like_rate_counts_unknown_placeholders():
    s = pd.Series(['A', 'Unknown', '', None, 'NA'])
    assert missing_like_rate(s) == 0.8


def test_data_quality_treats_unknown_labels_as_missing_like():
    df = pd.DataFrame({
        'product':['Credit card','Unknown'],
        'issue':['Issue','Issue'],
        'consumer_complaint_narrative':['This is a long enough complaint narrative about evidence.']*2,
        'company_response_to_consumer':['Closed','Closed'],
        'timely_response':['Yes','Unknown'],
        'date_received':['2024-01-01','bad date'],
        'state':['NJ','NY'],
    })
    _, metrics = data_quality_report(df, list(df.columns), label_columns=['product','timely_response'])
    assert metrics['label_missing_rate'] >= 0.5
    assert metrics['date_validity'] == 0.5
