import pandas as pd

from model_quality.features.label_policy import consolidate_rare_labels


def test_rare_labels_are_collapsed_and_logged():
    df = pd.DataFrame({'product': ['A'] * 5 + ['B'] * 2 + ['C']})
    out, meta = consolidate_rare_labels(df, min_count=3)
    assert out['product'].value_counts().to_dict() == {'A': 5, 'Other': 3}
    assert set(meta['collapsed_labels']) == {'B', 'C'}
    assert meta['collapsed_row_count'] == 3
