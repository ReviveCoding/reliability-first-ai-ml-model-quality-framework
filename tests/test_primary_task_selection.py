import pandas as pd

from model_quality.evaluation.model_selection import select_champion_model


def test_champion_restricted_to_primary_target():
    df = pd.DataFrame([
        {'model': 'product_model', 'target_col': 'product', 'macro_f1': .70, 'pr_auc': .75, 'ece': .10, 'brier': .20, 'log_loss': .80, 'worst_slice_f1': .60},
        {'model': 'timely_model', 'target_col': 'timely_response', 'macro_f1': .99, 'pr_auc': .99, 'ece': .01, 'brier': .01, 'log_loss': .05, 'worst_slice_f1': .99},
    ])
    out = select_champion_model(df, primary_target='product')
    assert out['champion_model'] == 'product_model'
    assert out['target_col'] == 'product'
