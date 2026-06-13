import yaml

from model_quality.gates.quality_gate import evaluate_gate


def test_quality_gate_pass(tmp_path):
    th = {
        'data_quality': {'completeness_min':0.9,'duplicate_rate_max':0.1,'label_missing_rate_max':0.1,'evidence_coverage_min':0.8},
        'model_quality': {'macro_f1_min':0.5,'pr_auc_min':0.5,'ece_max':0.2,'brier_max':0.3,'worst_slice_f1_min':0.4},
        'evidence_quality': {'recall_at_5_min':0.5,'mrr_min':0.3,'context_precision_min':0.5,'unsupported_claim_rate_max':0.2},
        'drift': {'max_psi':0.3,'training_serving_skew_max':0.2},
        'genai_telemetry': {'task_completion_rate_min':0.8,'p95_latency_ms_max':3000,'regression_flag_rate_max':0.1},
    }
    p = tmp_path / 'gate.yaml'
    p.write_text(yaml.safe_dump(th))
    metrics = {
        'data_quality': {'completeness':.99,'duplicate_rate':0,'label_missing_rate':0,'evidence_coverage':.95},
        'model_quality': {'best_macro_f1':.7,'best_pr_auc':.7,'best_ece':.1,'best_brier':.2,'worst_slice_f1':.5},
        'evidence_quality': {'recall_at_5':.7,'mrr':.5,'context_precision':.7,'unsupported_claim_rate':.05},
        'drift': {'max_psi':.1,'training_serving_skew':.0},
        'genai_telemetry': {'task_completion_rate':.9,'latency_p95_ms':2000,'regression_flag_rate':.05},
    }
    assert evaluate_gate(metrics, str(p))['status'] == 'PASS'
