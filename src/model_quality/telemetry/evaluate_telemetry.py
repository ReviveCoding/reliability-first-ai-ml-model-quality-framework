from __future__ import annotations

import pandas as pd


def telemetry_metrics(tel: pd.DataFrame) -> dict:
    return {
        'unsupported_claim_rate': float(tel['unsupported_claim_proxy'].mean()),
        'task_completion_rate': float(tel['task_completion'].mean()),
        'latency_p50_ms': float(tel['latency_ms'].quantile(0.50)),
        'latency_p95_ms': float(tel['latency_ms'].quantile(0.95)),
        'regression_flag_rate': float(tel['regression_flag'].mean()),
        'human_review_rate': float(tel['human_review_required'].mean()),
    }
