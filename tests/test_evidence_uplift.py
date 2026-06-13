import pandas as pd

from model_quality.retrieval.evidence_metrics import evaluate_evidence_quality


def test_evidence_metrics_report_baseline_and_uplift_fields():
    df = pd.DataFrame({
        'product': ['Credit card', 'Mortgage', 'Credit card'],
        'issue': ['Incorrect information', 'Managing account', 'Incorrect information'],
        'company_response_to_consumer': ['Closed', 'In progress', 'Closed'],
        'consumer_complaint_narrative': [
            'card billing wrong record and inaccurate balance details',
            'mortgage escrow account access and statement review details',
            'merchant charge incorrect information on billing statement',
        ],
    })
    metrics, _ = evaluate_evidence_quality(df, use_cross_encoder=False, top_k_final=2)
    assert metrics['bm25_recall_at_5'] == metrics['recall_at_5']
    assert metrics['rerank_recall_uplift'] == 0.0
    assert metrics['cross_encoder_requested'] is False
