from __future__ import annotations

import math

import yaml


def _check(name, value, threshold, mode):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return {
            'name': name, 'value': None, 'threshold': float(threshold), 'passed': False,
            'mode': mode, 'violation_ratio': 1.0,
        }
    value = float(value)
    threshold = float(threshold)
    passed = value >= threshold if mode == 'min' else value <= threshold
    if passed:
        violation = 0.0
    elif mode == 'min':
        violation = (threshold - value) / max(abs(threshold), 1e-12)
    else:
        violation = (value - threshold) / max(abs(threshold), 1e-12)
    return {
        'name': name, 'value': value, 'threshold': threshold, 'passed': bool(passed),
        'mode': mode, 'violation_ratio': float(max(0.0, violation)),
    }


def evaluate_gate(metrics: dict, thresholds_path: str = 'configs/launch_gate.yaml') -> dict:
    """Evaluate quality gates with severity-aware PASS/REVIEW/BLOCK logic.

    Slight misses are REVIEW rather than automatic BLOCK. BLOCK is reserved for
    hard integrity failures, a very large single miss, or accumulated material
    violations across multiple quality dimensions.
    """
    with open(thresholds_path, 'r', encoding='utf-8') as f:
        th = yaml.safe_load(f)
    mapping = {
        'data_quality.required_column_coverage_min': ('data_quality.required_column_coverage', 'min'),
        'data_quality.completeness_min': ('data_quality.completeness', 'min'),
        'data_quality.duplicate_rate_max': ('data_quality.duplicate_rate', 'max'),
        'data_quality.label_missing_rate_max': ('data_quality.label_missing_rate', 'max'),
        'data_quality.date_validity_min': ('data_quality.date_validity', 'min'),
        'data_quality.evidence_coverage_min': ('data_quality.evidence_coverage', 'min'),
        'split_integrity.no_overlap_min': ('split_integrity.no_overlap', 'min'),
        'split_integrity.temporal_order_valid_min': ('split_integrity.temporal_order_valid', 'min'),
        'split_integrity.product_calibration_class_coverage_min': ('split_integrity.product_calibration_class_coverage', 'min'),
        'split_integrity.product_selection_class_coverage_min': ('split_integrity.product_selection_class_coverage', 'min'),
        'split_integrity.product_test_class_coverage_min': ('split_integrity.product_test_class_coverage', 'min'),
        'model_quality.macro_f1_min': ('model_quality.best_macro_f1', 'min'),
        'model_quality.pr_auc_min': ('model_quality.best_pr_auc', 'min'),
        'model_quality.ece_max': ('model_quality.best_ece', 'max'),
        'model_quality.brier_max': ('model_quality.best_brier', 'max'),
        'model_quality.log_loss_max': ('model_quality.best_log_loss', 'max'),
        'model_quality.worst_slice_f1_min': ('model_quality.worst_slice_f1', 'min'),
        'model_quality.unknown_target_rate_max': ('model_quality.unknown_target_rate', 'max'),
        'model_quality.evaluated_rows_min': ('model_quality.evaluated_rows', 'min'),
        'model_quality.macro_f1_ci_low_min': ('model_quality.macro_f1_ci_low', 'min'),
        'model_quality.selection_test_macro_f1_gap_max': ('model_quality.selection_test_macro_f1_gap', 'max'),
        'evidence_quality.recall_at_5_min': ('evidence_quality.recall_at_5', 'min'),
        'evidence_quality.mrr_min': ('evidence_quality.mrr', 'min'),
        'evidence_quality.context_precision_min': ('evidence_quality.context_precision', 'min'),
        'evidence_quality.unsupported_claim_rate_max': ('evidence_quality.unsupported_claim_rate', 'max'),
        'drift.max_psi': ('drift.max_psi', 'max'),
        'drift.ks_pvalue_min': ('drift.min_ks_pvalue', 'min'),
        'drift.training_serving_skew_max': ('drift.training_serving_skew', 'max'),
        'genai_telemetry.task_completion_rate_min': ('genai_telemetry.task_completion_rate', 'min'),
        'genai_telemetry.p95_latency_ms_max': ('genai_telemetry.latency_p95_ms', 'max'),
        'genai_telemetry.regression_flag_rate_max': ('genai_telemetry.regression_flag_rate', 'max'),
    }
    flat = {}
    def flatten(prefix, obj):
        for k, v in obj.items():
            key = f'{prefix}.{k}' if prefix else k
            if isinstance(v, dict):
                flatten(key, v)
            else:
                flat[key] = v
    flatten('', metrics)

    checks = []
    for th_key, (metric_key, mode) in mapping.items():
        group, name = th_key.split('.')
        if group in th and name in th[group]:
            checks.append(_check(metric_key, flat.get(metric_key), th[group][name], mode))

    failed = [c for c in checks if not c['passed']]
    hard_integrity_names = {
        'data_quality.required_column_coverage',
        'split_integrity.no_overlap',
        'split_integrity.temporal_order_valid',
    }
    hard_failures = [c for c in failed if c['name'] in hard_integrity_names]

    # A metric more than 100% beyond its allowed distance is severe. For
    # bounded rates, a 2x threshold breach is also severe.
    severe_single = [c for c in failed if c['violation_ratio'] >= 1.0]
    weighted_total = 0.0
    for c in failed:
        weight = 1.5 if c['name'].startswith(('model_quality.', 'evidence_quality.', 'drift.', 'genai_telemetry.')) else 1.0
        weighted_total += weight * min(c['violation_ratio'], 2.0)

    if not failed:
        status = 'PASS'
    elif hard_failures or severe_single or weighted_total >= 2.5:
        status = 'BLOCK'
    else:
        status = 'REVIEW'

    return {
        'status': status,
        'checks': checks,
        'failed_checks': failed,
        'hard_failures': hard_failures,
        'severity_summary': {
            'failed_count': len(failed),
            'max_violation_ratio': float(max((c['violation_ratio'] for c in failed), default=0.0)),
            'weighted_total_violation': float(weighted_total),
        },
    }
