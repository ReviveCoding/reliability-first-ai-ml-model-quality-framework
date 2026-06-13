import pandas as pd

from model_quality.drift.psi import psi
from model_quality.drift.stress_injection import simulate_training_serving_skew


def test_psi_nonnegative():
    assert psi(pd.Series(['a','a','b']), pd.Series(['a','b','b'])) >= 0


def test_training_serving_skew():
    ref = pd.DataFrame({'a':[1], 'b':[2]})
    cur = pd.DataFrame({'a':[1]})
    out = simulate_training_serving_skew(ref, cur)
    assert out['training_serving_skew'] == 0.5
