from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import bootstrap

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from model_quality.simulation.scenarios import load_monte_carlo_config, spawn_run_seeds


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def _code_tree_sha256() -> str:
    """Hash executable project code used by Monte Carlo runs.

    Resume validity must change when imported modules change, not only when the
    top-level pipeline script changes.
    """
    h = hashlib.sha256()
    roots = [ROOT / 'src', ROOT / 'scripts']
    paths = []
    for root in roots:
        paths.extend(root.rglob('*.py'))
    for path in sorted(paths, key=lambda item: item.relative_to(ROOT).as_posix()):
        rel = path.relative_to(ROOT).as_posix().encode('utf-8')
        h.update(rel)
        h.update(b'\0')
        h.update(path.read_bytes())
        h.update(b'\0')
    return h.hexdigest()


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _run_signature(*, scenario: str, seed: int, sample_size: int, enable_lightgbm: bool, config_path: Path) -> dict:
    pipeline_path = ROOT / 'scripts' / 'run_full_pipeline.py'
    return {
        'scenario': scenario,
        'seed': int(seed),
        'sample_size': int(sample_size),
        'enable_lightgbm': bool(enable_lightgbm),
        'config_sha256': _sha256(config_path),
        'pipeline_sha256': _sha256(pipeline_path),
        'code_tree_sha256': _code_tree_sha256(),
    }


def _resume_is_valid(run_root: Path, expected_signature: dict) -> bool:
    manifest_path = run_root / 'run_manifest.json'
    out_dir = run_root / 'outputs'
    if not (manifest_path.exists() and (out_dir / 'all_metrics.json').exists() and (out_dir / 'launch_gate_result.json').exists()):
        return False
    try:
        manifest = _read_json(manifest_path)
    except Exception:
        return False
    return bool(manifest.get('completed') and manifest.get('signature') == expected_signature)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def _extract_row(scenario: str, seed: int, outputs: Path) -> dict:
    metrics = _read_json(outputs / 'all_metrics.json')
    gate = _read_json(outputs / 'launch_gate_result.json')
    model = metrics['model_quality']
    evidence = metrics['evidence_quality']
    drift = metrics['drift']
    telemetry = metrics['genai_telemetry']
    return {
        'scenario': scenario,
        'seed': seed,
        'gate_status': gate['status'],
        'failed_gate_count': len(gate.get('failed_checks', [])),
        'weighted_gate_violation': gate.get('severity_summary', {}).get('weighted_total_violation', 0.0),
        'champion_model': model.get('champion_model'),
        'macro_f1': model.get('best_macro_f1'),
        'pr_auc': model.get('best_pr_auc'),
        'ece': model.get('best_ece'),
        'brier': model.get('best_brier'),
        'log_loss': model.get('best_log_loss'),
        'worst_slice_f1': model.get('worst_slice_f1'),
        'macro_f1_ci_low': model.get('macro_f1_ci_low'),
        'selection_test_macro_f1_gap': model.get('selection_test_macro_f1_gap'),
        'evaluated_rows': model.get('evaluated_rows'),
        'evidence_recall_at_5': evidence.get('recall_at_5'),
        'context_precision': evidence.get('context_precision'),
        'unsupported_claim_rate': evidence.get('unsupported_claim_rate'),
        'max_psi': drift.get('max_psi'),
        'min_ks_pvalue': drift.get('min_ks_pvalue'),
        'training_serving_skew': drift.get('training_serving_skew'),
        'task_completion_rate': telemetry.get('task_completion_rate'),
        'latency_p95_ms': telemetry.get('latency_p95_ms'),
        'regression_flag_rate': telemetry.get('regression_flag_rate'),
        'human_review_rate': telemetry.get('human_review_rate'),
    }


def _bootstrap_ci(values: pd.Series, seed: int = 0) -> tuple[float, float]:
    x = pd.to_numeric(values, errors='coerce').dropna().to_numpy(dtype=float)
    if len(x) == 0:
        return float('nan'), float('nan')
    if len(x) == 1 or np.allclose(x, x[0]):
        return float(x[0]), float(x[0])
    res = bootstrap((x,), np.mean, confidence_level=0.95, n_resamples=2000, method='BCa', rng=np.random.default_rng(seed))
    return float(res.confidence_interval.low), float(res.confidence_interval.high)


def _build_summary(raw: pd.DataFrame, scenario_order: list[str], base_seed: int) -> pd.DataFrame:
    metrics = [
        'macro_f1', 'pr_auc', 'ece', 'brier', 'log_loss', 'worst_slice_f1',
        'macro_f1_ci_low', 'selection_test_macro_f1_gap', 'evaluated_rows',
        'evidence_recall_at_5', 'context_precision', 'unsupported_claim_rate',
        'max_psi', 'training_serving_skew', 'task_completion_rate',
        'latency_p95_ms', 'regression_flag_rate', 'human_review_rate',
    ]
    rows = []
    for scenario in scenario_order:
        g = raw[raw['scenario'] == scenario]
        gate_counts = g['gate_status'].value_counts(normalize=True)
        base = {
            'scenario': scenario,
            'runs': int(len(g)),
            'pass_rate': float(gate_counts.get('PASS', 0.0)),
            'review_rate': float(gate_counts.get('REVIEW', 0.0)),
            'block_rate': float(gate_counts.get('BLOCK', 0.0)),
        }
        for i, metric in enumerate(metrics):
            s = pd.to_numeric(g[metric], errors='coerce')
            lo, hi = _bootstrap_ci(s, seed=base_seed + i)
            base[f'{metric}_mean'] = float(s.mean())
            base[f'{metric}_std'] = float(s.std(ddof=1)) if s.notna().sum() > 1 else 0.0
            base[f'{metric}_p05'] = float(s.quantile(0.05))
            base[f'{metric}_p95'] = float(s.quantile(0.95))
            base[f'{metric}_ci95_low'] = lo
            base[f'{metric}_ci95_high'] = hi
        rows.append(base)
    return pd.DataFrame(rows)


def _champion_distribution(raw: pd.DataFrame) -> pd.DataFrame:
    required = {'scenario', 'champion_model'}
    if raw.empty or not required.issubset(raw.columns):
        return pd.DataFrame(columns=['scenario', 'champion_model', 'count', 'rate'])
    out = (
        raw.groupby(['scenario', 'champion_model'], dropna=False)
        .size()
        .reset_index(name='count')
    )
    totals = out.groupby('scenario')['count'].transform('sum')
    out['rate'] = out['count'] / totals.where(totals > 0, 1)
    return out.sort_values(['scenario', 'count', 'champion_model'], ascending=[True, False, True]).reset_index(drop=True)


def _monotonicity(summary: pd.DataFrame, order: list[str]) -> dict:
    indexed = summary.set_index('scenario').loc[order]
    decreasing = ['macro_f1_mean', 'pr_auc_mean', 'worst_slice_f1_mean', 'evidence_recall_at_5_mean', 'context_precision_mean', 'task_completion_rate_mean']
    increasing = ['log_loss_mean', 'unsupported_claim_rate_mean', 'max_psi_mean', 'latency_p95_ms_mean', 'regression_flag_rate_mean', 'human_review_rate_mean']
    checks = []
    for metric in decreasing:
        vals = indexed[metric].to_list()
        passed = all(vals[i] >= vals[i + 1] - 0.02 for i in range(len(vals) - 1))
        checks.append({'metric': metric, 'direction': 'nonincreasing', 'values': vals, 'passed': bool(passed)})
    for metric in increasing:
        vals = indexed[metric].to_list()
        passed = all(vals[i] <= vals[i + 1] + 1e-9 for i in range(len(vals) - 1))
        checks.append({'metric': metric, 'direction': 'nondecreasing', 'values': vals, 'passed': bool(passed)})
    gate_checks = []
    if 'nominal' in indexed.index:
        nominal_block = float(indexed.loc['nominal', 'block_rate'])
        gate_checks.append({'name': 'nominal_block_rate_le_0.20', 'value': nominal_block, 'passed': bool(nominal_block <= 0.20)})
    if 'severe' in indexed.index:
        severe_block = float(indexed.loc['severe', 'block_rate'])
        gate_checks.append({'name': 'severe_block_rate_ge_0.80', 'value': severe_block, 'passed': bool(severe_block >= 0.80)})
    return {
        'metric_checks': checks,
        'gate_checks': gate_checks,
        'all_passed': bool(all(x['passed'] for x in checks + gate_checks)),
    }


def _completion_checks(raw: pd.DataFrame, scenarios: list[str], runs_per: int, failures: list[dict]) -> list[dict]:
    checks = []
    expected_total = runs_per * len(scenarios)
    checks.append({
        'name': 'completed_run_count_matches_expected',
        'expected': expected_total,
        'actual': int(len(raw)),
        'passed': bool(len(raw) == expected_total),
    })
    for scenario in scenarios:
        actual = int((raw['scenario'] == scenario).sum()) if not raw.empty and 'scenario' in raw else 0
        checks.append({
            'name': f'{scenario}_run_count_matches_expected',
            'expected': int(runs_per),
            'actual': actual,
            'passed': bool(actual == runs_per),
        })
    checks.append({
        'name': 'pipeline_failures_zero',
        'expected': 0,
        'actual': int(len(failures)),
        'passed': bool(not failures),
    })
    return checks


def _markdown_report(raw: pd.DataFrame, summary: pd.DataFrame, champion_distribution: pd.DataFrame, checks: dict) -> str:
    gate = raw.groupby(['scenario', 'gate_status']).size().unstack(fill_value=0)
    key_cols = [
        'scenario', 'runs', 'pass_rate', 'review_rate', 'block_rate',
        'macro_f1_mean', 'macro_f1_ci95_low', 'macro_f1_ci95_high',
        'log_loss_mean', 'max_psi_mean', 'task_completion_rate_mean',
        'latency_p95_ms_mean', 'regression_flag_rate_mean',
    ]
    champion_table = champion_distribution.to_markdown(index=False) if not champion_distribution.empty else 'No completed champion selections.'
    return f"""# Monte Carlo End-to-End Validation Report

## Design
Repeated end-to-end runs across nominal, moderate, and severe scenarios. Each run uses an independent child seed, scenario-specific synthetic data generation, drift injection, telemetry generation, model training, quality evaluation, and PASS/REVIEW/BLOCK launch gating.

## Gate distribution

{gate.to_markdown()}

## Scenario summary

{summary[key_cols].to_markdown(index=False)}

## Champion-selection stability

{champion_table}

## Monotonicity and sensitivity checks

```json
{json.dumps(checks, indent=2)}
```

## Interpretation
- Predictive quality should decline as ambiguity and label conflict increase.
- Log loss, drift, latency, regression flags, and human-review demand should increase with stress.
- Nominal runs should avoid frequent BLOCK decisions, while severe runs should usually BLOCK.
- Confidence intervals summarize run-to-run uncertainty; they do not imply production guarantees.
"""


def _slim_run(run_root: Path, keep_full: bool) -> None:
    if keep_full:
        return
    for pattern in ('*.joblib', 'telemetry_cases.csv', 'human_review_queue.csv', 'calibration_bins.csv'):
        for p in (run_root / 'outputs').glob(pattern):
            p.unlink(missing_ok=True)
    shutil.rmtree(run_root / 'data', ignore_errors=True)
    shutil.rmtree(run_root / 'reports', ignore_errors=True)
    shutil.rmtree(run_root / 'tracking', ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/monte_carlo.yaml')
    ap.add_argument('--runs-per-scenario', type=int, default=None)
    ap.add_argument('--sample-size', type=int, default=None)
    ap.add_argument('--base-seed', type=int, default=None)
    ap.add_argument('--scenarios', nargs='+', default=['nominal', 'moderate', 'severe'])
    ap.add_argument('--output-root', default='monte_carlo')
    ap.add_argument('--enable-lightgbm', action='store_true')
    ap.add_argument('--retain-all', action='store_true')
    ap.add_argument('--resume', action='store_true', help='Reuse completed run directories instead of deleting the output root.')
    ap.add_argument('--jobs', type=int, default=1, help='Number of concurrent subprocess runs.')
    args = ap.parse_args()

    config_path = (ROOT / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config).resolve()
    cfg = load_monte_carlo_config(config_path)
    runs_per = args.runs_per_scenario or int(cfg.get('runs_per_scenario', 6))
    sample_size = args.sample_size or int(cfg.get('sample_size', 800))
    base_seed = args.base_seed if args.base_seed is not None else int(cfg.get('base_seed', 20260612))
    scenarios = args.scenarios
    missing = [x for x in scenarios if x not in cfg['scenarios']]
    if missing:
        raise KeyError(f'Unknown scenarios: {missing}')

    output_root = ROOT / args.output_root
    runs_root = output_root / 'runs'
    outputs_root = output_root / 'outputs'
    reports_root = output_root / 'reports'
    if not args.resume:
        shutil.rmtree(output_root, ignore_errors=True)
    runs_root.mkdir(parents=True, exist_ok=True)
    outputs_root.mkdir(parents=True, exist_ok=True)
    reports_root.mkdir(parents=True, exist_ok=True)

    seeds = spawn_run_seeds(base_seed, runs_per * len(scenarios))
    rows = []
    failures = []
    tasks = []
    idx = 0
    for scenario in scenarios:
        for run_idx in range(runs_per):
            seed = seeds[idx]
            idx += 1
            tasks.append((scenario, run_idx, seed))

    def execute(task):
        scenario, run_idx, seed = task
        run_root = runs_root / scenario / f'run_{run_idx + 1:02d}_seed_{seed}'
        data_dir = run_root / 'data'
        out_dir = run_root / 'outputs'
        rep_dir = run_root / 'reports'
        tracking_dir = run_root / 'tracking'
        signature = _run_signature(
            scenario=scenario, seed=seed, sample_size=sample_size,
            enable_lightgbm=args.enable_lightgbm, config_path=config_path,
        )
        if args.resume and _resume_is_valid(run_root, signature):
            return ('row', _extract_row(scenario, seed, out_dir))
        if run_root.exists():
            shutil.rmtree(run_root)
        cmd = [
            sys.executable, 'scripts/run_full_pipeline.py',
            '--use-synthetic', '--sample-size', str(sample_size),
            '--random-state', str(seed), '--scenario', scenario,
            '--scenario-config', str(config_path),
            '--data-dir', str(data_dir), '--outputs-dir', str(out_dir),
            '--reports-dir', str(rep_dir), '--tracking-uri', str(tracking_dir),
            '--disable-mlflow',
        ]
        if args.enable_lightgbm:
            cmd.append('--enable-lightgbm')
        env = dict(os.environ)
        env.update({'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'NUMEXPR_NUM_THREADS': '1'})
        proc = subprocess.run(cmd, cwd=ROOT, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (run_root / 'pipeline.log').parent.mkdir(parents=True, exist_ok=True)
        (run_root / 'pipeline.log').write_text(proc.stdout, encoding='utf-8')
        if proc.returncode != 0:
            return ('failure', {'scenario': scenario, 'seed': seed, 'returncode': proc.returncode, 'output_tail': proc.stdout[-3000:]})
        row = _extract_row(scenario, seed, out_dir)
        run_manifest = {
            'completed': True,
            'signature': signature,
            'gate_status': row.get('gate_status'),
            'champion_model': row.get('champion_model'),
        }
        (run_root / 'run_manifest.json').write_text(json.dumps(run_manifest, indent=2), encoding='utf-8')
        _slim_run(run_root, keep_full=args.retain_all or run_idx == 0)
        return ('row', row)

    if args.jobs <= 1:
        results = [execute(task) for task in tasks]
    else:
        results = []
        with ThreadPoolExecutor(max_workers=args.jobs) as ex:
            futures = {ex.submit(execute, task): task for task in tasks}
            for future in as_completed(futures):
                results.append(future.result())
    for kind, payload in results:
        if kind == 'row':
            rows.append(payload)
        else:
            failures.append(payload)
    rows.sort(key=lambda r: (scenarios.index(r['scenario']), int(r['seed'])))

    raw = pd.DataFrame(rows)
    raw.to_csv(outputs_root / 'monte_carlo_raw_results.csv', index=False)
    if failures:
        (outputs_root / 'monte_carlo_failures.json').write_text(json.dumps(failures, indent=2), encoding='utf-8')
    if raw.empty:
        raise RuntimeError('All Monte Carlo runs failed.')

    summary = _build_summary(raw, scenarios, base_seed)
    summary.to_csv(outputs_root / 'monte_carlo_scenario_summary.csv', index=False)
    gate_distribution = raw.groupby(['scenario', 'gate_status']).size().reset_index(name='count')
    gate_distribution.to_csv(outputs_root / 'monte_carlo_gate_distribution.csv', index=False)
    champion_distribution = _champion_distribution(raw)
    champion_distribution.to_csv(outputs_root / 'monte_carlo_champion_distribution.csv', index=False)
    checks = _monotonicity(summary, scenarios)
    completion = _completion_checks(raw, scenarios, runs_per, failures)
    checks['completion_checks'] = completion
    checks['pipeline_failures'] = failures
    checks['all_passed'] = bool(checks.get('all_passed') and all(c['passed'] for c in completion))
    (outputs_root / 'monte_carlo_validation_checks.json').write_text(json.dumps(checks, indent=2), encoding='utf-8')
    report = _markdown_report(raw, summary, champion_distribution, checks)
    (reports_root / 'monte_carlo_validation_report.md').write_text(report, encoding='utf-8')
    manifest = {
        'base_seed': base_seed,
        'runs_per_scenario': runs_per,
        'sample_size': sample_size,
        'scenarios': scenarios,
        'run_count_expected': int(runs_per * len(scenarios)),
        'run_count_completed': int(len(raw)),
        'run_count_failed': int(len(failures)),
        'validation_checks_passed': checks['all_passed'],
        'config_sha256': _sha256(config_path),
        'pipeline_sha256': _sha256(ROOT / 'scripts' / 'run_full_pipeline.py'),
        'code_tree_sha256': _code_tree_sha256(),
        'environment': {
            'python': platform.python_version(),
            'platform': platform.platform(),
            'numpy': _package_version('numpy'),
            'pandas': _package_version('pandas'),
            'scikit_learn': _package_version('scikit-learn'),
            'scipy': _package_version('scipy'),
            'lightgbm': _package_version('lightgbm'),
        },
    }
    (outputs_root / 'monte_carlo_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(manifest, indent=2))
    print(summary[['scenario', 'pass_rate', 'review_rate', 'block_rate', 'macro_f1_mean', 'log_loss_mean', 'max_psi_mean', 'task_completion_rate_mean']].to_string(index=False))
    raise SystemExit(0 if checks['all_passed'] else 1)


if __name__ == '__main__':
    main()
