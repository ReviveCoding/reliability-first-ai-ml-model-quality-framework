from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_FILES = [
    'README.md', 'requirements.txt', 'requirements-core.txt', 'requirements-ci.txt', 'requirements-gpu.txt', 'pyproject.toml', 'Makefile',
    'configs/launch_gate.yaml', 'configs/product_taxonomy.yaml', 'scripts/run_full_pipeline.py',
    'src/model_quality/ingestion/load_cfpb.py', 'src/model_quality/features/product_taxonomy.py', 'src/model_quality/models/train_logistic.py',
    'src/model_quality/retrieval/cross_encoder_reranker.py', 'src/model_quality/tracking/mlflow_logger.py',
    'tests/test_quality_gate.py', 'tests/test_split_integrity.py', 'tests/test_monte_carlo_scenarios.py',
    'scripts/run_monte_carlo.py', 'scripts/gpu_preflight.py', 'scripts/dataset_preflight.py',
    'scripts/run_dataset_audit.py', 'configs/monte_carlo.yaml', 'MONTE_CARLO_VALIDATION.md',
    'STRENGTHENING_LOOP_REPORT.md', 'SMALL_DATA_VALIDATION.md',
    'LOCAL_GITHUB_RUNNABLE_AUDIT.md', 'dashboard/app.py'
]
REQUIRED_OUTPUTS = [
    'outputs/model_selection_leaderboard.csv', 'outputs/model_test_leaderboard.csv', 'outputs/model_leaderboard.csv', 'outputs/data_quality_metrics.csv', 'outputs/calibration_summary.csv',
    'outputs/evidence_quality_metrics.csv', 'outputs/drift_alerts.csv', 'outputs/telemetry_metrics.csv',
    'outputs/human_review_queue.csv', 'outputs/split_integrity_metrics.csv', 'outputs/product_taxonomy.json', 'outputs/label_policy.json',
    'outputs/pipeline_manifest.json', 'outputs/launch_gate_result.json', 'outputs/evidence_quality_summary.json',
    'reports/executive_summary.md', 'reports/launch_decision_memo.md', 'reports/split_integrity_report.md', 'reports/model_card.md', 'reports/data_card.md'
]


def check_files(paths):
    missing = [p for p in paths if not (ROOT / p).exists()]
    return {'passed': not missing, 'missing': missing}


def run_cmd(cmd):
    p = subprocess.run(cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return {'cmd': ' '.join(cmd), 'returncode': p.returncode, 'output_tail': p.stdout[-4000:]}


def _sanitize_paths(obj):
    if isinstance(obj, dict):
        return {key: _sanitize_paths(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_paths(value) for value in obj]
    if isinstance(obj, str):
        return obj.replace(str(ROOT), '<PROJECT_ROOT>')
    return obj


def main():
    run_smoke = '--run-smoke' in sys.argv
    run_mc_smoke = '--run-monte-carlo-smoke' in sys.argv
    run_gpu_preflight = '--run-gpu-preflight' in sys.argv
    result = {'structure': check_files(REQUIRED_FILES)}
    result['ruff'] = run_cmd([sys.executable, '-m', 'ruff', 'check', '.'])
    result['compileall'] = run_cmd([sys.executable, '-m', 'compileall', '-q', 'src', 'scripts', 'dashboard'])
    result['package_import'] = run_cmd([sys.executable, '-c', "import sys; sys.path.insert(0, 'src'); import model_quality; print(model_quality.__file__)"])
    result['pytest'] = run_cmd([sys.executable, '-m', 'pytest', '-q'])
    if run_smoke:
        result['smoke_pipeline'] = run_cmd([sys.executable, 'scripts/run_full_pipeline.py', '--use-synthetic', '--sample-size', '800', '--enable-lightgbm', '--random-state', '20260612', '--disable-mlflow'])
        result['outputs'] = check_files(REQUIRED_OUTPUTS)
        gate_path = ROOT / 'outputs/launch_gate_result.json'
        if gate_path.exists():
            result['launch_gate'] = json.loads(gate_path.read_text(encoding='utf-8'))
    else:
        result['outputs'] = check_files(REQUIRED_OUTPUTS) if (ROOT / 'outputs').exists() else {'passed': None, 'missing': 'Run with --run-smoke to validate generated outputs.'}


    if run_gpu_preflight:
        result['gpu_preflight'] = run_cmd([sys.executable, 'scripts/gpu_preflight.py'])

    if run_mc_smoke:
        result['monte_carlo_smoke'] = run_cmd([
            sys.executable, 'scripts/run_monte_carlo.py',
            '--scenarios', 'nominal', 'severe',
            '--runs-per-scenario', '1', '--sample-size', '400', '--jobs', '2',
            '--output-root', 'monte_carlo_smoke'
        ])
        manifest_path = ROOT / 'monte_carlo_smoke' / 'outputs' / 'monte_carlo_manifest.json'
        checks_path = ROOT / 'monte_carlo_smoke' / 'outputs' / 'monte_carlo_validation_checks.json'
        result['monte_carlo_outputs'] = {
            'passed': manifest_path.exists() and checks_path.exists(),
            'manifest_exists': manifest_path.exists(),
            'checks_exists': checks_path.exists(),
        }
        if checks_path.exists():
            result['monte_carlo_checks'] = json.loads(checks_path.read_text(encoding='utf-8'))
        if manifest_path.exists():
            result['monte_carlo_manifest'] = json.loads(manifest_path.read_text(encoding='utf-8'))
    result['passed'] = bool(
        result['structure']['passed'] and
        result['ruff']['returncode'] == 0 and
        result['compileall']['returncode'] == 0 and
        result['package_import']['returncode'] == 0 and
        result['pytest']['returncode'] == 0 and
        (not run_smoke or (result.get('smoke_pipeline', {}).get('returncode') == 0 and result['outputs']['passed'])) and
        (not run_gpu_preflight or result.get('gpu_preflight', {}).get('returncode') == 0) and
        (not run_mc_smoke or (
            result.get('monte_carlo_smoke', {}).get('returncode') == 0 and
            result.get('monte_carlo_outputs', {}).get('passed') and
            result.get('monte_carlo_checks', {}).get('all_passed')
        ))
    )
    out_dir = ROOT / 'outputs'
    out_dir.mkdir(exist_ok=True)
    sanitized = _sanitize_paths(result)
    (out_dir / 'project_audit.json').write_text(json.dumps(sanitized, indent=2), encoding='utf-8')
    print(json.dumps(sanitized, indent=2))
    raise SystemExit(0 if result['passed'] else 1)


if __name__ == '__main__':
    main()
