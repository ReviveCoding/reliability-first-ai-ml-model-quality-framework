from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def md_table(df: pd.DataFrame, max_rows=20):
    if df is None or df.empty:
        return '_No rows._'
    return df.head(max_rows).to_markdown(index=False)


def _safe_read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def write_reports(
    reports_dir: str,
    outputs_dir: str,
    data_quality_df: pd.DataFrame,
    model_leaderboard: pd.DataFrame,
    calibration_df: pd.DataFrame,
    evidence_metrics: dict,
    drift_df: pd.DataFrame,
    telemetry_metrics: dict,
    gate_result: dict,
    split_df: pd.DataFrame | None = None,
    all_metrics: dict | None = None,
    selection_leaderboard: pd.DataFrame | None = None,
):
    reports = Path(reports_dir)
    outputs = Path(outputs_dir)
    reports.mkdir(parents=True, exist_ok=True)
    outputs.mkdir(parents=True, exist_ok=True)
    all_metrics = all_metrics or {}

    (reports/'data_quality_report.md').write_text('# Data Quality Report\n\n'+md_table(data_quality_df, max_rows=80)+'\n', encoding='utf-8')
    (reports/'split_integrity_report.md').write_text('# Split Integrity Report\n\n'+md_table(split_df, max_rows=80)+'\n', encoding='utf-8')
    selection_section = '# Model Selection Leaderboard (validation selection window)\n\n' + md_table(selection_leaderboard, max_rows=80) + '\n\n'
    test_section = '# Final Test Leaderboard (untouched test window)\n\n' + md_table(model_leaderboard, max_rows=80) + '\n'
    (reports/'model_quality_report.md').write_text(selection_section + test_section, encoding='utf-8')
    (reports/'calibration_report.md').write_text('# Calibration Report\n\n'+md_table(calibration_df, max_rows=80)+'\n', encoding='utf-8')
    (reports/'evidence_quality_report.md').write_text('# Evidence Quality Report\n\n```json\n'+json.dumps(evidence_metrics, indent=2)+'\n```\n', encoding='utf-8')
    (reports/'drift_report.md').write_text('# Drift Report\n\n'+md_table(drift_df, max_rows=80)+'\n', encoding='utf-8')

    queue_path = outputs/'human_review_queue.csv'
    queue_md = '\n\n## Human review queue\n\n' + (md_table(_safe_read_csv(queue_path), max_rows=25) if queue_path.exists() else '_No queue generated._')
    (reports/'genai_telemetry_report.md').write_text('# GenAI Telemetry Report\n\n```json\n'+json.dumps(telemetry_metrics, indent=2)+'\n```\n'+queue_md+'\n', encoding='utf-8')

    failed_df = pd.DataFrame(gate_result.get('failed_checks', []))
    checks_df = pd.DataFrame(gate_result.get('checks', []))
    (reports/'launch_decision_memo.md').write_text(
        '# Launch Decision Memo\n\n'
        f"**Decision:** {gate_result['status']}\n\n"
        '## Failed checks\n\n'+md_table(failed_df, max_rows=80)+'\n\n'
        '## All gate checks\n\n'+md_table(checks_df, max_rows=120)+'\n',
        encoding='utf-8'
    )

    champion = all_metrics.get('model_quality', {})
    split_metrics = all_metrics.get('split_integrity', {})
    product_taxonomy = all_metrics.get('product_taxonomy', {})
    label_policy = all_metrics.get('label_policy', {})
    provenance = all_metrics.get('data_provenance', {})
    model_card = f"""# Model Card: Quality Sign-Off Champion

## Data provenance
```json
{json.dumps(provenance, indent=2)}
```

## Champion model
- **Model:** {champion.get('champion_model')}
- **Selection reason:** {champion.get('champion_reason')}
- **Selection macro-F1:** {champion.get('selection_macro_f1')}
- **Selection/test macro-F1 gap:** {champion.get('selection_test_macro_f1_gap')}

## Intended use
Offline portfolio-style model quality evaluation on public CFPB-style complaint data and synthetic telemetry. This is not a production financial decision system.

## Evaluation summary
- Final test macro-F1: {champion.get('best_macro_f1')}
- Macro-F1 95% bootstrap interval: [{champion.get('macro_f1_ci_low')}, {champion.get('macro_f1_ci_high')}]
- Final evaluated rows: {champion.get('evaluated_rows')}
- PR-AUC: {champion.get('best_pr_auc')}
- ECE: {champion.get('best_ece')}
- Brier: {champion.get('best_brier')}
- Log loss: {champion.get('best_log_loss')}
- Worst-slice F1: {champion.get('worst_slice_f1')}
- Unknown-target rate: {champion.get('unknown_target_rate')}

## Split integrity
```json
{json.dumps(split_metrics, indent=2)}
```

## Product taxonomy
```json
{json.dumps(product_taxonomy, indent=2)}
```

## Label policy
```json
{json.dumps(label_policy, indent=2)}
```

## Limitations
CFPB public complaints are not a representative sample of all consumer experiences, and synthetic telemetry is used only to simulate GenAI workflow quality signals. Launch decisions are local/offline quality gates, not production approvals.
"""
    (reports/'model_card.md').write_text(model_card, encoding='utf-8')

    data_card = f"""# Data Card

## Data sources
- Public CFPB-style complaint records when a CFPB CSV is supplied.
- Synthetic CFPB-style fallback data for local smoke tests.
- Synthetic payment-style GenAI telemetry for evidence/grounding and workflow-quality evaluation.

## Provenance
```json
{json.dumps(provenance, indent=2)}
```

## Product taxonomy normalization
```json
{json.dumps(product_taxonomy, indent=2)}
```

## Label consolidation policy
```json
{json.dumps(label_policy, indent=2)}
```

## Quality checks
{md_table(data_quality_df, max_rows=80)}

## Split checks
{md_table(split_df, max_rows=80)}

## Claim boundary
This project does not use Apple, private customer, private payment, or production monitoring data. Synthetic telemetry is not evidence of real production behavior.
"""
    (reports/'data_card.md').write_text(data_card, encoding='utf-8')

    executive = f"""# Executive Summary

## Project
End-to-End AI/ML Model Quality Sign-Off Framework

## Launch decision
**{gate_result['status']}**

## Scope
Local-runnable public/proxy/synthetic data framework for data validation, split-integrity checks, supervised model training, MLflow experiment tracking, evidence-quality evaluation, drift testing, synthetic GenAI telemetry replay, and PASS/REVIEW/BLOCK launch gates.

## Data provenance
```json
{json.dumps(provenance, indent=2)}
```

## Champion model
- **Model:** {champion.get('champion_model')}
- **Reason:** {champion.get('champion_reason')}

## Model selection leaderboard

{md_table(selection_leaderboard, max_rows=80)}

## Final test leaderboard

{md_table(model_leaderboard, max_rows=80)}

## Split integrity

{md_table(split_df, max_rows=80)}

## Key evidence-quality metrics

```json
{json.dumps(evidence_metrics, indent=2)}
```
"""
    (reports/'executive_summary.md').write_text(executive, encoding='utf-8')
