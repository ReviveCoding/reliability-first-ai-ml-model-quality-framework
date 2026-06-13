import pandas as pd

from model_quality.evaluation.model_selection import select_champion_model


def test_select_champion_prefers_best_quality_model():
    lb = pd.DataFrame([
        {'model':'baseline','macro_f1':0.55,'pr_auc':0.60,'ece':0.02,'brier':0.20,'worst_slice_f1':0.50},
        {'model':'advanced','macro_f1':0.80,'pr_auc':0.78,'ece':0.05,'brier':0.12,'worst_slice_f1':0.70},
    ])
    champ = select_champion_model(lb)
    assert champ['champion_model'] == 'advanced'
    assert champ['macro_f1'] == 0.80
