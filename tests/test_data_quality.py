from model_quality.ingestion.load_cfpb import make_synthetic_cfpb
from model_quality.validation.quality_metrics import data_quality_report


def test_data_quality_synthetic():
    df = make_synthetic_cfpb(100)
    required = ['product','issue','consumer_complaint_narrative','company_response_to_consumer','timely_response','date_received','state']
    report, metrics = data_quality_report(df, required, label_columns=['product'])
    assert metrics['row_count'] == 100
    assert metrics['required_column_coverage'] == 1.0
    assert not report.empty
