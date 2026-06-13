import yaml

from model_quality.gates.quality_gate import evaluate_gate


def _thresholds(tmp_path):
    th = {
        'model_quality': {'macro_f1_min': 0.60, 'log_loss_max': 1.20},
        'genai_telemetry': {'task_completion_rate_min': 0.80},
    }
    p = tmp_path / 'gate.yaml'
    p.write_text(yaml.safe_dump(th))
    return str(p)


def test_small_gate_misses_review_not_block(tmp_path):
    metrics = {
        'model_quality': {'best_macro_f1': 0.58, 'best_log_loss': 1.25},
        'genai_telemetry': {'task_completion_rate': 0.78},
    }
    result = evaluate_gate(metrics, _thresholds(tmp_path))
    assert result['status'] == 'REVIEW'
    assert result['severity_summary']['failed_count'] == 3


def test_large_gate_miss_blocks(tmp_path):
    metrics = {
        'model_quality': {'best_macro_f1': 0.10, 'best_log_loss': 3.0},
        'genai_telemetry': {'task_completion_rate': 0.20},
    }
    result = evaluate_gate(metrics, _thresholds(tmp_path))
    assert result['status'] == 'BLOCK'
