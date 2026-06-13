from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Prevent BLAS/OpenMP oversubscription on high-core local and CI machines.
_DEFAULT_CPU_THREADS = os.environ.get('MODEL_QUALITY_CPU_THREADS', '1')
for _env_name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_env_name] = _DEFAULT_CPU_THREADS

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from model_quality.drift.ks_test import ks_report
from model_quality.drift.psi import psi_report
from model_quality.drift.stress_injection import (
    inject_feature_drift,
    inject_missingness,
    simulate_training_serving_skew,
)
from model_quality.evaluation.calibration_metrics import calibration_report
from model_quality.evaluation.classification_metrics import classification_report_dict
from model_quality.evaluation.model_selection import select_champion_model
from model_quality.evaluation.slice_metrics import worst_slice_f1
from model_quality.evaluation.uncertainty import bootstrap_macro_f1_interval
from model_quality.features.label_policy import apply_label_policy, fit_label_policy
from model_quality.features.product_taxonomy import canonicalize_product_taxonomy
from model_quality.features.split_builder import split_validation_window, temporal_split
from model_quality.gates.quality_gate import evaluate_gate
from model_quality.ingestion.load_cfpb import discover_cfpb_path, load_cfpb
from model_quality.models.train_lightgbm import train_lightgbm_or_fallback
from model_quality.models.train_logistic import train_calibrated_tfidf_logistic, train_tfidf_logistic
from model_quality.models.train_transformer import train_transformer_classifier
from model_quality.reporting.write_reports import write_reports
from model_quality.retrieval.evidence_metrics import evaluate_evidence_quality
from model_quality.risk.human_review import build_human_review_queue
from model_quality.simulation.scenarios import get_scenario
from model_quality.telemetry.evaluate_telemetry import telemetry_metrics
from model_quality.telemetry.generate_genai_telemetry import generate_telemetry
from model_quality.tracking.mlflow_logger import ExperimentLogger
from model_quality.utils import ensure_dirs, environment_info, save_json, sha256_file
from model_quality.validation.quality_metrics import data_quality_report
from model_quality.validation.split_integrity import split_integrity_report


def _evaluate_sklearn_model(model, frame: pd.DataFrame, *, random_state: int, with_uncertainty: bool) -> tuple[dict, pd.DataFrame]:
    target = model.target_col
    metrics = classification_report_dict(model, frame, target)
    calibration, table = calibration_report(model, frame, target)
    slice_metrics = worst_slice_f1(model, frame, target, slice_col='state', min_count=5)
    metrics.update({
        'ece': calibration.get('ece'),
        'brier': calibration.get('brier'),
        'log_loss': calibration.get('log_loss'),
        'worst_slice_f1': slice_metrics.get('worst_slice_f1'),
    })
    if with_uncertainty:
        metrics.update(bootstrap_macro_f1_interval(
            model,
            frame,
            target,
            n_resamples=400,
            random_state=random_state,
        ))
    return metrics, table


def _row_for_model(df: pd.DataFrame, name: str) -> dict:
    if df.empty or 'model' not in df.columns:
        return {}
    matched = df[df['model'].astype(str) == str(name)]
    return matched.iloc[0].to_dict() if not matched.empty else {}


def _number(row: dict, key: str):
    value = row.get(key)
    return float(value) if value is not None and pd.notna(value) else None


def _artifact_inventory(paths: list[Path]) -> list[dict]:
    rows = []
    for path in sorted(set(paths), key=lambda p: str(p)):
        if not (path.exists() and path.is_file()):
            continue
        try:
            display_path = path.resolve().relative_to(ROOT.resolve()).as_posix()
        except ValueError:
            display_path = f'{path.parent.name}/{path.name}'
        rows.append({
            'path': display_path,
            'size_bytes': int(path.stat().st_size),
            'sha256': sha256_file(path),
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfpb-path', type=str, default=None)
    ap.add_argument('--use-synthetic', action='store_true')
    ap.add_argument('--sample-size', type=int, default=2500)
    ap.add_argument(
        '--sampling-strategy',
        choices=['auto', 'duckdb', 'reservoir', 'head'],
        default='auto',
        help='auto uses DuckDB for large CSVs and pandas reservoir sampling for smaller files.',
    )
    ap.add_argument('--source-chunksize', type=int, default=50000)
    ap.add_argument('--archive-cache-dir', type=str, default=None, help='Optional persistent cache for extracted .7z CSVs.')
    ap.add_argument('--min-target-count', type=int, default=20)
    ap.add_argument(
        '--product-taxonomy-path',
        type=str,
        default='configs/product_taxonomy.yaml',
        help='Versioned CFPB product taxonomy mapping applied before temporal splitting.',
    )
    ap.add_argument('--random-state', type=int, default=7)
    ap.add_argument('--scenario', type=str, default='nominal')
    ap.add_argument('--scenario-config', type=str, default='configs/monte_carlo.yaml')
    ap.add_argument('--enable-lightgbm', action='store_true')
    ap.add_argument('--enable-transformer', action='store_true')
    ap.add_argument('--enable-cross-encoder', action='store_true')
    ap.add_argument('--transformer-model', type=str, default='distilbert-base-uncased')
    ap.add_argument('--transformer-epochs', type=int, default=2)
    ap.add_argument('--transformer-batch-size', type=int, default=32)
    ap.add_argument('--cross-encoder-model', type=str, default='cross-encoder/ms-marco-MiniLM-L-6-v2')
    ap.add_argument('--cross-encoder-batch-size', type=int, default=64)
    ap.add_argument('--device', type=str, default='auto')
    ap.add_argument('--data-dir', type=str, default='data')
    ap.add_argument('--outputs-dir', type=str, default='outputs')
    ap.add_argument('--reports-dir', type=str, default='reports')
    ap.add_argument('--tracking-uri', type=str, default='mlruns')
    ap.add_argument('--disable-mlflow', action='store_true')
    ap.add_argument('--thresholds-path', type=str, default='configs/launch_gate.yaml')
    args = ap.parse_args()
    started_at = time.perf_counter()

    outputs = Path(args.outputs_dir)
    reports = Path(args.reports_dir)
    data_dir = Path(args.data_dir)
    processed_dir = data_dir / 'processed'
    golden_dir = data_dir / 'golden'
    ensure_dirs(outputs, reports, processed_dir, golden_dir, args.tracking_uri)

    resolved_cfpb_path = args.cfpb_path
    if not args.use_synthetic and resolved_cfpb_path is None:
        resolved_cfpb_path = str(discover_cfpb_path(data_dir / 'raw'))

    scenario_params = get_scenario(args.scenario, args.scenario_config)
    df = load_cfpb(
        resolved_cfpb_path,
        sample_size=args.sample_size,
        random_state=args.random_state,
        use_synthetic=args.use_synthetic,
        scenario=args.scenario,
        scenario_config_path=args.scenario_config,
        sampling_strategy=args.sampling_strategy,
        source_chunksize=args.source_chunksize,
        archive_cache_dir=args.archive_cache_dir,
    )
    df, product_taxonomy = canonicalize_product_taxonomy(df, args.product_taxonomy_path)
    source_attrs = dict(df.attrs)
    save_json(product_taxonomy, outputs / 'product_taxonomy.json')
    required = [
        'product', 'issue', 'consumer_complaint_narrative',
        'company_response_to_consumer', 'timely_response',
        'date_received', 'state',
    ]
    dq_df, dq_metrics = data_quality_report(df, required, label_columns=['product', 'timely_response'])
    dq_df.to_csv(outputs / 'data_quality_metrics.csv', index=False)

    # Split first, then fit target policy on training only. This avoids using
    # future/test label frequencies during model development.
    raw_train, raw_validation, raw_test = temporal_split(df)
    raw_calibration, raw_selection = split_validation_window(raw_validation)
    policy = fit_label_policy(raw_train, target_col='product', min_count=args.min_target_count)
    train, train_policy_stats = apply_label_policy(raw_train, policy)
    calibration, calibration_policy_stats = apply_label_policy(raw_calibration, policy)
    selection, selection_policy_stats = apply_label_policy(raw_selection, policy)
    test, test_policy_stats = apply_label_policy(raw_test, policy)
    label_policy = {
        **policy,
        'application': {
            'train': train_policy_stats,
            'calibration': calibration_policy_stats,
            'selection': selection_policy_stats,
            'test': test_policy_stats,
        },
    }
    save_json(label_policy, outputs / 'label_policy.json')

    combined = pd.concat([
        train.assign(_split='train'),
        calibration.assign(_split='calibration'),
        selection.assign(_split='selection'),
        test.assign(_split='test'),
    ], ignore_index=True)
    combined.to_csv(processed_dir / 'modeling_dataset.csv', index=False)
    train.to_csv(golden_dir / 'train.csv', index=False)
    calibration.to_csv(golden_dir / 'calibration.csv', index=False)
    selection.to_csv(golden_dir / 'selection.csv', index=False)
    test.to_csv(golden_dir / 'test.csv', index=False)

    split_df, split_metrics = split_integrity_report(
        train,
        selection,
        test,
        target_cols=['product'],
        calibration=calibration,
    )
    split_df.to_csv(outputs / 'split_integrity_metrics.csv', index=False)

    logger = ExperimentLogger(
        experiment_name=f'model_quality_signoff_{args.scenario}',
        tracking_uri=args.tracking_uri,
        use_mlflow=not args.disable_mlflow,
    )

    models = []
    logistic = train_tfidf_logistic(train, target_col='product', random_state=args.random_state)
    models.append(logistic)
    try:
        calibrated_logistic = train_calibrated_tfidf_logistic(
            train,
            target_col='product',
            random_state=args.random_state,
            calibration_df=calibration,
            base_model=logistic,
        )
        models.append(calibrated_logistic)
    except Exception as exc:
        print(f'[WARN] Calibrated logistic skipped: {exc}')
    if args.enable_lightgbm:
        try:
            models.append(train_lightgbm_or_fallback(
                train,
                target_col='timely_response',
                random_state=args.random_state,
            ))
        except Exception as exc:
            print(f'[WARN] LightGBM/fallback training skipped: {exc}')

    selection_rows: list[dict] = []
    test_rows: list[dict] = []
    calibration_rows: list[dict] = []
    calibration_tables: list[pd.DataFrame] = []
    model_objects = {}

    for model in models:
        selection_metrics, _ = _evaluate_sklearn_model(
            model,
            selection,
            random_state=args.random_state + 11,
            with_uncertainty=False,
        )
        test_metrics, table = _evaluate_sklearn_model(
            model,
            test,
            random_state=args.random_state + 29,
            with_uncertainty=True,
        )
        selection_rows.append(selection_metrics)
        test_rows.append(test_metrics)
        calibration_rows.append({
            'model': model.name,
            'ece': test_metrics.get('ece'),
            'brier': test_metrics.get('brier'),
            'log_loss': test_metrics.get('log_loss'),
            'evaluated_rows': test_metrics.get('evaluated_rows'),
            'unknown_target_rate': test_metrics.get('unknown_target_rate'),
        })
        if table is not None and not table.empty:
            table = table.copy()
            table['model'] = model.name
            calibration_tables.append(table)
        model_path = outputs / f'{model.name}.joblib'
        model.save(model_path)
        model_objects[model.name] = model
        logger.log_run(
            model.name,
            {'target_col': model.target_col, 'scenario': args.scenario, 'random_state': args.random_state},
            {k: v for k, v in test_metrics.items() if isinstance(v, (int, float)) and v is not None},
            [str(model_path)],
        )

    if args.enable_transformer:
        transformer = train_transformer_classifier(
            train,
            calibration,
            selection_df=selection,
            test_df=test,
            model_name=args.transformer_model,
            device=args.device,
            out_dir=str(outputs / 'transformer_model'),
            random_state=args.random_state,
            epochs=args.transformer_epochs,
            batch_size=args.transformer_batch_size,
        )
        if transformer.enabled:
            selection_rows.append({
                'model': 'transformer_text_classifier',
                'target_col': 'product',
                **(transformer.selection_metrics or {}),
                'note': transformer.note,
                'device': transformer.device,
            })
            test_rows.append({
                'model': 'transformer_text_classifier',
                'target_col': 'product',
                **transformer.metrics,
                'note': transformer.note,
                'device': transformer.device,
            })
            logger.log_run(
                'transformer_text_classifier',
                {
                    'model_name': args.transformer_model,
                    'device': transformer.device,
                    'scenario': args.scenario,
                    'random_state': args.random_state,
                },
                {k: v for k, v in transformer.metrics.items() if isinstance(v, (int, float))},
                [transformer.artifact_dir] if transformer.artifact_dir else [],
            )
        else:
            print(f'[WARN] {transformer.note}')

    selection_leaderboard = pd.DataFrame(selection_rows)
    test_leaderboard = pd.DataFrame(test_rows)
    selection_leaderboard.to_csv(outputs / 'model_selection_leaderboard.csv', index=False)
    test_leaderboard.to_csv(outputs / 'model_test_leaderboard.csv', index=False)
    # Compatibility alias retained for existing dashboard/report consumers.
    test_leaderboard.to_csv(outputs / 'model_leaderboard.csv', index=False)
    calibration_df = pd.DataFrame(calibration_rows)
    calibration_df.to_csv(outputs / 'calibration_summary.csv', index=False)
    if calibration_tables:
        pd.concat(calibration_tables, ignore_index=True).to_csv(outputs / 'calibration_bins.csv', index=False)

    selection_champion = select_champion_model(selection_leaderboard, primary_target='product')
    champion_name = selection_champion.get('champion_model')
    champion_test = _row_for_model(test_leaderboard, champion_name)
    selection_test_gap = None
    if selection_champion.get('macro_f1') is not None and _number(champion_test, 'macro_f1') is not None:
        selection_test_gap = abs(float(selection_champion['macro_f1']) - float(champion_test['macro_f1']))

    evidence_metrics, evidence_detail = evaluate_evidence_quality(
        test.reset_index(drop=True),
        use_cross_encoder=args.enable_cross_encoder,
        device=args.device,
        cross_encoder_model=args.cross_encoder_model,
        cross_encoder_batch_size=args.cross_encoder_batch_size,
    )
    evidence_detail.to_csv(outputs / 'evidence_quality_metrics.csv', index=False)
    save_json(evidence_metrics, outputs / 'evidence_quality_summary.json')

    drift_missing = float(scenario_params.get('drift_missingness_rate', 0.10))
    drift_shift = float(scenario_params.get('drift_categorical_shift', 0.20))
    drift_current = inject_missingness(
        test,
        'consumer_complaint_narrative',
        drift_missing,
        random_state=args.random_state + 101,
    )
    drift_current = inject_feature_drift(
        drift_current,
        'product',
        drift_shift,
        random_state=args.random_state + 202,
    )
    psi_df = psi_report(test, drift_current, ['product', 'issue', 'state'])
    ref_len = test['consumer_complaint_narrative'].fillna('').astype(str).str.len()
    cur_len = drift_current['consumer_complaint_narrative'].fillna('').astype(str).str.len()
    ks_df = ks_report(test.assign(text_len=ref_len), drift_current.assign(text_len=cur_len), ['text_len'])
    drift_df = psi_df.merge(ks_df, how='outer', on='column') if not ks_df.empty else psi_df

    skew_strength = float(scenario_params.get('drift_categorical_shift', 0.20))
    drop_cols = []
    if skew_strength >= 0.25:
        drop_cols.append('state')
    if skew_strength >= 0.50:
        drop_cols.append('submitted_via')
    serving_df = test.drop(columns=drop_cols, errors='ignore')
    skew = simulate_training_serving_skew(test, serving_df)
    drift_df.to_csv(outputs / 'drift_alerts.csv', index=False)
    save_json(skew, outputs / 'training_serving_skew.json')

    telemetry = generate_telemetry(
        test.reset_index(drop=True),
        random_state=args.random_state + 303,
        scenario_params=scenario_params,
    )
    telemetry_summary = telemetry_metrics(telemetry)
    telemetry.to_csv(outputs / 'telemetry_cases.csv', index=False)
    pd.DataFrame([telemetry_summary]).to_csv(outputs / 'telemetry_metrics.csv', index=False)
    review_queue = build_human_review_queue(telemetry, evidence_detail)
    review_queue.to_csv(outputs / 'human_review_queue.csv', index=False)

    max_psi = float(drift_df['psi'].dropna().max()) if 'psi' in drift_df and drift_df['psi'].notna().any() else 0.0
    min_ks_pvalue = float(drift_df['ks_pvalue'].dropna().min()) if 'ks_pvalue' in drift_df and drift_df['ks_pvalue'].notna().any() else 1.0

    all_metrics = {
        'simulation': {
            'scenario': args.scenario,
            'random_state': int(args.random_state),
            'sample_size': int(len(df)),
        },
        'product_taxonomy': product_taxonomy,
        'label_policy': label_policy,
        'data_provenance': {
            'source_type': 'synthetic' if args.use_synthetic else 'public_cfpb',
            'source_file': Path(resolved_cfpb_path).name if resolved_cfpb_path else None,
            'sampling_strategy_requested': args.sampling_strategy,
            'sampling_strategy_effective': source_attrs.get('effective_sampling_strategy'),
            'source_chunksize': int(args.source_chunksize),
            'archive_cache_enabled': bool(args.archive_cache_dir),
            'requested_sample_size': int(args.sample_size),
            'actual_modeling_rows': int(len(combined)),
            'min_target_count': int(args.min_target_count),
            'product_taxonomy_version': product_taxonomy.get('taxonomy_version'),
            'product_taxonomy_changed_row_rate': product_taxonomy.get('changed_row_rate'),
            'source_required_columns_present': source_attrs.get('source_required_columns_present', []),
        },
        'data_quality': dq_metrics,
        'split_integrity': split_metrics,
        'model_quality': {
            'champion_model': champion_name,
            'champion_target': selection_champion.get('target_col'),
            'champion_reason': (
                'Selected on the dedicated model-selection window. '
                + str(selection_champion.get('reason', ''))
            ),
            'selection_macro_f1': selection_champion.get('macro_f1'),
            'selection_pr_auc': selection_champion.get('pr_auc'),
            'selection_test_macro_f1_gap': selection_test_gap,
            'best_macro_f1': _number(champion_test, 'macro_f1'),
            'best_pr_auc': _number(champion_test, 'pr_auc'),
            'best_ece': _number(champion_test, 'ece'),
            'best_brier': _number(champion_test, 'brier'),
            'best_log_loss': _number(champion_test, 'log_loss'),
            'worst_slice_f1': _number(champion_test, 'worst_slice_f1'),
            'unknown_target_rate': _number(champion_test, 'unknown_target_rate'),
            'evaluated_rows': _number(champion_test, 'evaluated_rows'),
            'macro_f1_ci_low': _number(champion_test, 'macro_f1_ci_low'),
            'macro_f1_ci_high': _number(champion_test, 'macro_f1_ci_high'),
        },
        'evidence_quality': evidence_metrics,
        'drift': {
            'max_psi': max_psi,
            'min_ks_pvalue': min_ks_pvalue,
            'training_serving_skew': skew['training_serving_skew'],
        },
        'genai_telemetry': telemetry_summary,
    }
    save_json(all_metrics, outputs / 'all_metrics.json')
    gate_result = evaluate_gate(all_metrics, thresholds_path=args.thresholds_path)
    save_json(gate_result, outputs / 'launch_gate_result.json')

    write_reports(
        str(reports),
        str(outputs),
        dq_df,
        test_leaderboard,
        calibration_df,
        evidence_metrics,
        drift_df,
        telemetry_summary,
        gate_result,
        split_df=split_df,
        all_metrics=all_metrics,
        selection_leaderboard=selection_leaderboard,
    )
    logger.log_run(
        'launch_gate',
        {'pipeline': 'full', 'scenario': args.scenario, 'random_state': args.random_state},
        {'status_numeric': {'PASS': 2, 'REVIEW': 1, 'BLOCK': 0}[gate_result['status']]},
        [str(outputs / 'launch_gate_result.json'), str(reports / 'executive_summary.md')],
    )

    artifact_paths = [
        outputs / 'model_selection_leaderboard.csv',
        outputs / 'model_test_leaderboard.csv',
        outputs / 'data_quality_metrics.csv',
        outputs / 'product_taxonomy.json',
        outputs / 'split_integrity_metrics.csv',
        outputs / 'evidence_quality_summary.json',
        outputs / 'drift_alerts.csv',
        outputs / 'telemetry_metrics.csv',
        outputs / 'launch_gate_result.json',
        reports / 'executive_summary.md',
        reports / 'model_card.md',
        reports / 'launch_decision_memo.md',
    ]
    pipeline_manifest = {
        'schema_version': 3,
        'runtime_seconds': round(time.perf_counter() - started_at, 4),
        'cpu_thread_limit': int(_DEFAULT_CPU_THREADS),
        'scenario': args.scenario,
        'random_state': int(args.random_state),
        'launch_gate': gate_result['status'],
        'champion_model': champion_name,
        'selection_protocol': 'train -> calibration -> selection -> untouched test',
        'data_provenance': all_metrics['data_provenance'],
        'optional_features': {
            'lightgbm_requested': bool(args.enable_lightgbm),
            'transformer_requested': bool(args.enable_transformer),
            'cross_encoder_requested': bool(args.enable_cross_encoder),
            'cross_encoder_effective': bool(evidence_metrics.get('used_cross_encoder')),
            'device': args.device,
            'mlflow_requested': bool(not args.disable_mlflow),
        },
        'config_hashes': {
            'scenario_config': sha256_file(args.scenario_config),
            'launch_gate': sha256_file(args.thresholds_path),
            'product_taxonomy': sha256_file(args.product_taxonomy_path),
        },
        'artifact_inventory': _artifact_inventory(artifact_paths),
        'environment': environment_info(),
    }
    save_json(pipeline_manifest, outputs / 'pipeline_manifest.json')

    print(
        f"Pipeline completed. Scenario={args.scenario} seed={args.random_state} "
        f"launch_gate={gate_result['status']} champion={champion_name}"
    )
    print(f"Reports: {reports.resolve()}")
    print(f"Outputs: {outputs.resolve()}")


if __name__ == '__main__':
    main()
