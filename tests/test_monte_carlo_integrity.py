import json
from pathlib import Path

import pandas as pd

from scripts.run_monte_carlo import _completion_checks, _resume_is_valid


def test_completion_checks_fail_on_missing_runs_and_failures():
    raw = pd.DataFrame([
        {'scenario': 'nominal'},
        {'scenario': 'severe'},
    ])
    checks = _completion_checks(raw, ['nominal', 'severe'], runs_per=2, failures=[{'seed': 1}])
    by_name = {c['name']: c for c in checks}
    assert not by_name['completed_run_count_matches_expected']['passed']
    assert not by_name['nominal_run_count_matches_expected']['passed']
    assert not by_name['severe_run_count_matches_expected']['passed']
    assert not by_name['pipeline_failures_zero']['passed']


def test_resume_requires_exact_signature_and_complete_outputs(tmp_path: Path):
    run_root = tmp_path / 'run'
    outputs = run_root / 'outputs'
    outputs.mkdir(parents=True)
    signature = {'scenario': 'nominal', 'seed': 7, 'sample_size': 100}
    (outputs / 'all_metrics.json').write_text('{}', encoding='utf-8')
    (outputs / 'launch_gate_result.json').write_text('{}', encoding='utf-8')
    (run_root / 'run_manifest.json').write_text(
        json.dumps({'completed': True, 'signature': signature}), encoding='utf-8'
    )
    assert _resume_is_valid(run_root, signature)
    assert not _resume_is_valid(run_root, {**signature, 'sample_size': 200})


def test_monotonicity_supports_single_scenario_runs():
    from scripts.run_monte_carlo import _monotonicity
    summary = pd.DataFrame([{
        'scenario': 'nominal',
        'block_rate': 0.0,
        'macro_f1_mean': 0.8,
        'pr_auc_mean': 0.8,
        'worst_slice_f1_mean': 0.7,
        'evidence_recall_at_5_mean': 0.8,
        'context_precision_mean': 0.6,
        'task_completion_rate_mean': 0.9,
        'log_loss_mean': 0.5,
        'unsupported_claim_rate_mean': 0.1,
        'max_psi_mean': 0.05,
        'latency_p95_ms_mean': 1500,
        'regression_flag_rate_mean': 0.03,
        'human_review_rate_mean': 0.1,
    }])
    checks = _monotonicity(summary, ['nominal'])
    assert checks['all_passed']
    assert len(checks['gate_checks']) == 1
