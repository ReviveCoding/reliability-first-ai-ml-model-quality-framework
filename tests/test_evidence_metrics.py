import pandas as pd

from model_quality.retrieval.evidence_metrics import build_evidence_cases, evaluate_evidence_quality


def test_evidence_queries_do_not_leak_exact_narrative_text():
    df = pd.DataFrame({
        'product': ['Credit card', 'Mortgage'],
        'issue': ['Incorrect information', 'Managing account'],
        'state': ['NJ', 'NY'],
        'company_response_to_consumer': ['Closed with explanation', 'In progress'],
        'consumer_complaint_narrative': ['secret phrase alpha unique narrative', 'secret phrase beta unique narrative'],
    })
    cases = build_evidence_cases(df, n_cases=2)
    assert 'secret phrase alpha' not in cases['query'].iloc[0]
    metrics, detail = evaluate_evidence_quality(df)
    assert 0.0 <= metrics['context_precision'] <= 1.0
    assert 0.0 <= metrics['unsupported_claim_rate'] <= 1.0
    assert 'precision_at_k' in detail.columns
