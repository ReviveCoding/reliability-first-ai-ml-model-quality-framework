import pandas as pd

from model_quality.drift.psi import psi
from model_quality.drift.stress_injection import inject_feature_drift


def test_categorical_drift_changes_distribution():
    df = pd.DataFrame({'x': ['a'] * 80 + ['b'] * 15 + ['c'] * 5})
    shifted = inject_feature_drift(df, 'x', rate=0.5, random_state=3, target_value='c')
    assert shifted['x'].value_counts(normalize=True)['c'] > df['x'].value_counts(normalize=True)['c']
    assert psi(df['x'], shifted['x']) > 0.05
