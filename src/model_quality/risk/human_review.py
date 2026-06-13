from __future__ import annotations

import pandas as pd


def build_human_review_queue(telemetry: pd.DataFrame, evidence_detail: pd.DataFrame | None = None, max_rows: int = 200) -> pd.DataFrame:
    """Create a compact queue of cases needing human/model-quality review."""
    tel = telemetry.copy()
    tel['review_reason'] = ''
    tel.loc[tel['unsupported_claim_proxy'].astype(bool), 'review_reason'] += 'unsupported_claim;'
    tel.loc[tel['regression_flag'].astype(bool), 'review_reason'] += 'regression_flag;'
    tel.loc[tel['evidence_coverage'] < 0.60, 'review_reason'] += 'low_evidence_coverage;'
    tel.loc[tel['latency_ms'] > tel['latency_ms'].quantile(0.95), 'review_reason'] += 'high_latency;'
    queue = tel[tel['review_reason'].str.len() > 0].copy()
    queue['priority_score'] = (
        queue['unsupported_claim_proxy'].astype(float) * 3.0 +
        queue['regression_flag'].astype(float) * 2.0 +
        (1.0 - queue['evidence_coverage']).clip(lower=0) +
        (queue['latency_ms'] / max(float(tel['latency_ms'].max()), 1.0))
    )
    cols = ['workflow_id','case_id','prompt_type','review_reason','priority_score','unsupported_claim_proxy','evidence_coverage','task_completion','latency_ms','regression_flag']
    return queue.sort_values('priority_score', ascending=False)[cols].head(max_rows).reset_index(drop=True)
