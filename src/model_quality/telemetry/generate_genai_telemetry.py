from __future__ import annotations

import numpy as np
import pandas as pd


def generate_telemetry(
    df: pd.DataFrame,
    random_state: int = 7,
    scenario_params: dict | None = None,
) -> pd.DataFrame:
    """Generate reproducible synthetic GenAI workflow telemetry.

    Scenario parameters alter evidence quality, latency, unsupported-claim risk,
    regression frequency, and completion criteria so Monte Carlo runs can test
    whether launch gates respond monotonically to increasing stress.
    """
    p = (scenario_params or {}).get('telemetry', scenario_params or {})
    rng = np.random.default_rng(random_state)
    n = len(df)
    alpha = float(p.get('evidence_alpha', 8.0))
    beta = float(p.get('evidence_beta', 2.0))
    latency_mean = float(p.get('latency_log_mean', 7.0))
    latency_sigma = float(p.get('latency_log_sigma', 0.35))
    unsupported_base = float(p.get('unsupported_base_rate', 0.03))
    regression_base = float(p.get('regression_base_rate', 0.05))
    evidence_min = float(p.get('completion_evidence_min', 0.45))
    latency_max = float(p.get('completion_latency_max_ms', 3500))

    evidence_coverage = rng.beta(alpha, beta, size=n)
    latency = rng.lognormal(mean=latency_mean, sigma=latency_sigma, size=n)
    unsupported = (evidence_coverage < 0.55) | (rng.random(n) < unsupported_base)
    task_completion = (evidence_coverage > evidence_min) & (latency < latency_max) & (~unsupported)
    regression_flag = (rng.random(n) < regression_base) | unsupported | (~task_completion)
    coherence = np.clip(rng.beta(max(1.2, alpha), max(1.2, beta), size=n), 0.0, 1.0)
    return pd.DataFrame({
        'workflow_id': [f'wf_{i:06d}' for i in range(n)],
        'case_id': list(range(n)),
        'prompt_type': rng.choice(['complaint_summary', 'response_check', 'evidence_lookup', 'risk_routing'], size=n),
        'retrieved_evidence_count': rng.integers(0, 8, size=n),
        'evidence_coverage': evidence_coverage,
        'unsupported_claim_proxy': unsupported.astype(int),
        'output_coherence_proxy': coherence,
        'task_completion': task_completion.astype(int),
        'latency_ms': latency,
        'regression_flag': regression_flag.astype(int),
        'human_review_required': (unsupported | regression_flag | (evidence_coverage < 0.6)).astype(int),
    })
