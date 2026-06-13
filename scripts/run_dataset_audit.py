from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_OUTPUTS = [
    'outputs/model_selection_leaderboard.csv',
    'outputs/model_test_leaderboard.csv',
    'outputs/data_quality_metrics.csv',
    'outputs/evidence_quality_summary.json',
    'outputs/drift_alerts.csv',
    'outputs/telemetry_metrics.csv',
    'outputs/launch_gate_result.json',
    'outputs/pipeline_manifest.json',
    'reports/executive_summary.md',
    'reports/launch_decision_memo.md',
]


def _run(command: list[str]) -> dict:
    process = subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return {
        'command': ' '.join(command),
        'returncode': process.returncode,
        'output_tail': process.stdout[-6000:],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description='Run a real-dataset preflight and end-to-end audit.')
    parser.add_argument('--cfpb-path', required=True)
    parser.add_argument('--archive-cache-dir', default='data/archive_cache')
    parser.add_argument('--sample-size', type=int, default=1000)
    parser.add_argument('--sampling-strategy', choices=['auto', 'duckdb', 'reservoir', 'head'], default='auto')
    parser.add_argument('--output-root', default='dataset_audit')
    parser.add_argument('--enable-lightgbm', action='store_true')
    args = parser.parse_args()

    output_root = Path(args.output_root)
    preflight_output = output_root / 'outputs' / 'dataset_preflight.json'
    preflight = _run([
        sys.executable,
        'scripts/dataset_preflight.py',
        '--cfpb-path', args.cfpb_path,
        '--archive-cache-dir', args.archive_cache_dir,
        '--sample-size', '100',
        '--sampling-strategy', 'head',
        '--output', str(preflight_output),
    ])

    pipeline_command = [
        sys.executable,
        'scripts/run_full_pipeline.py',
        '--cfpb-path', args.cfpb_path,
        '--archive-cache-dir', args.archive_cache_dir,
        '--sample-size', str(args.sample_size),
        '--sampling-strategy', args.sampling_strategy,
        '--disable-mlflow',
        '--data-dir', str(output_root / 'data'),
        '--outputs-dir', str(output_root / 'outputs'),
        '--reports-dir', str(output_root / 'reports'),
        '--tracking-uri', str(output_root / 'mlruns'),
    ]
    if args.enable_lightgbm:
        pipeline_command.append('--enable-lightgbm')
    pipeline = _run(pipeline_command) if preflight['returncode'] == 0 else {
        'command': ' '.join(pipeline_command),
        'returncode': None,
        'output_tail': 'Skipped because dataset preflight failed.',
    }

    missing = [path for path in REQUIRED_OUTPUTS if not (output_root / path).exists()]
    gate_path = output_root / 'outputs' / 'launch_gate_result.json'
    manifest_path = output_root / 'outputs' / 'pipeline_manifest.json'
    result = {
        'preflight': preflight,
        'pipeline': pipeline,
        'required_outputs_present': not missing,
        'missing_outputs': missing,
        'launch_gate': json.loads(gate_path.read_text(encoding='utf-8')) if gate_path.exists() else None,
        'pipeline_manifest': json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.exists() else None,
    }
    result['passed'] = bool(
        preflight['returncode'] == 0
        and pipeline['returncode'] == 0
        and result['required_outputs_present']
    )
    report_path = output_root / 'outputs' / 'dataset_audit.json'
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['passed'] else 1)


if __name__ == '__main__':
    main()
