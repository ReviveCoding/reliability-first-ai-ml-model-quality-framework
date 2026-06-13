import pandas as pd

from model_quality.risk.human_review import build_human_review_queue


def test_human_review_queue_prioritizes_risky_cases():
    tel = pd.DataFrame({
        'workflow_id':['a','b'], 'case_id':[1,2], 'prompt_type':['x','x'],
        'unsupported_claim_proxy':[1,0], 'evidence_coverage':[0.3,0.9],
        'task_completion':[0,1], 'latency_ms':[2000,1000], 'regression_flag':[1,0]
    })
    q = build_human_review_queue(tel)
    assert len(q) == 1
    assert q.iloc[0]['workflow_id'] == 'a'
    assert 'unsupported_claim' in q.iloc[0]['review_reason']
