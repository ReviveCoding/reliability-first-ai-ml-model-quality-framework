import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from model_quality.evaluation.uncertainty import bootstrap_macro_f1_interval


class FakeModel:
    def __init__(self):
        self.label_encoder = LabelEncoder().fit(['A', 'B'])
        self.target_col = 'product'

    def predict(self, df):
        return np.where(df['x'].to_numpy() > 0, 'B', 'A')


def test_bootstrap_macro_f1_interval_is_bounded_and_reproducible():
    df = pd.DataFrame({
        'product': ['A', 'A', 'B', 'B', 'B', 'A'],
        'x': [-1, -1, 1, 1, -1, -1],
    })
    first = bootstrap_macro_f1_interval(FakeModel(), df, 'product', n_resamples=100, random_state=3)
    second = bootstrap_macro_f1_interval(FakeModel(), df, 'product', n_resamples=100, random_state=3)
    assert first == second
    assert 0.0 <= first['macro_f1_ci_low'] <= first['macro_f1_ci_high'] <= 1.0
