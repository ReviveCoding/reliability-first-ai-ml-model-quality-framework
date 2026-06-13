import json
from pathlib import Path

import pandas as pd

try:
    import streamlit as st
except Exception:
    raise SystemExit('Install streamlit to run the dashboard: pip install streamlit')

ROOT = Path(__file__).resolve().parents[1]
outputs = ROOT / 'outputs'
reports = ROOT / 'reports'

st.set_page_config(page_title='AI/ML Model Quality Sign-Off', layout='wide')
st.title('End-to-End AI/ML Model Quality Sign-Off Framework')

if not outputs.exists():
    st.warning('Run the pipeline first: python scripts/run_full_pipeline.py --use-synthetic --enable-lightgbm')
    st.stop()

cols = st.columns(5)
gate_path = outputs / 'launch_gate_result.json'
if gate_path.exists():
    gate = json.loads(gate_path.read_text())
    cols[0].metric('Launch Gate', gate.get('status','UNKNOWN'))
    cols[1].metric('Failed checks', len(gate.get('failed_checks', [])))
else:
    cols[0].metric('Launch Gate','UNKNOWN')
    cols[1].metric('Failed checks', '—')

leaderboard_path = outputs / 'model_leaderboard.csv'
if leaderboard_path.exists():
    lb = pd.read_csv(leaderboard_path)
    if not lb.empty and 'macro_f1' in lb:
        cols[2].metric('Best macro-F1', f"{pd.to_numeric(lb['macro_f1'], errors='coerce').max():.3f}")
    if not lb.empty and 'ece' in lb:
        cols[3].metric('Best ECE', f"{pd.to_numeric(lb['ece'], errors='coerce').min():.3f}")

telemetry_path = outputs / 'telemetry_metrics.csv'
if telemetry_path.exists():
    tm = pd.read_csv(telemetry_path)
    if not tm.empty and 'human_review_rate' in tm:
        cols[4].metric('Human review rate', f"{float(tm['human_review_rate'].iloc[0]):.3f}")

sections = [
    ('Model Selection Leaderboard', outputs/'model_selection_leaderboard.csv'),
    ('Final Test Leaderboard', outputs/'model_test_leaderboard.csv'),
    ('Data Quality', outputs/'data_quality_metrics.csv'),
    ('Split Integrity', outputs/'split_integrity_metrics.csv'),
    ('Calibration', outputs/'calibration_summary.csv'),
    ('Evidence Quality', outputs/'evidence_quality_metrics.csv'),
    ('Drift', outputs/'drift_alerts.csv'),
    ('Telemetry', outputs/'telemetry_metrics.csv'),
    ('Human Review Queue', outputs/'human_review_queue.csv'),
]

for name, path in sections:
    st.header(name)
    if path.exists():
        st.dataframe(pd.read_csv(path), use_container_width=True)
    else:
        st.info(f'Missing {path.name}')

st.header('Launch Decision Memo')
memo = reports / 'launch_decision_memo.md'
if memo.exists():
    st.markdown(memo.read_text(encoding='utf-8'))
else:
    st.info('Run the pipeline to generate the launch decision memo.')


st.header('Monte Carlo Validation')
mc_summary = ROOT / 'monte_carlo' / 'outputs' / 'monte_carlo_scenario_summary.csv'
mc_checks = ROOT / 'monte_carlo' / 'outputs' / 'monte_carlo_validation_checks.json'
if mc_summary.exists():
    mc = pd.read_csv(mc_summary)
    show_cols = [c for c in ['scenario','runs','pass_rate','review_rate','block_rate','macro_f1_mean','log_loss_mean','max_psi_mean','task_completion_rate_mean','latency_p95_ms_mean'] if c in mc.columns]
    st.dataframe(mc[show_cols], use_container_width=True)
    if mc_checks.exists():
        checks = json.loads(mc_checks.read_text(encoding='utf-8'))
        st.metric('Monte Carlo sensitivity checks', 'PASS' if checks.get('all_passed') else 'REVIEW')
else:
    st.info('Run: python scripts/run_monte_carlo.py --runs-per-scenario 6 --sample-size 800 --enable-lightgbm --jobs 3')
