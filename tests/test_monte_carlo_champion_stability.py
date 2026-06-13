import pandas as pd

from scripts.run_monte_carlo import _champion_distribution


def test_champion_distribution_rates_sum_to_one_per_scenario():
    raw = pd.DataFrame(
        {
            'scenario': ['nominal', 'nominal', 'nominal', 'severe', 'severe'],
            'champion_model': ['a', 'a', 'b', 'a', 'b'],
        }
    )
    out = _champion_distribution(raw)
    sums = out.groupby('scenario')['rate'].sum().to_dict()
    assert sums == {'nominal': 1.0, 'severe': 1.0}
    nominal_a = out[(out['scenario'] == 'nominal') & (out['champion_model'] == 'a')].iloc[0]
    assert nominal_a['count'] == 2
    assert abs(nominal_a['rate'] - 2 / 3) < 1e-12
