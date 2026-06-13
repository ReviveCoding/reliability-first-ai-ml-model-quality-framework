import pandas as pd

from model_quality.features.label_policy import apply_label_policy, fit_label_policy


def test_label_policy_is_fit_on_training_only_and_preserves_novel_labels():
    train = pd.DataFrame({'product': ['A'] * 5 + ['B']})
    future = pd.DataFrame({'product': ['A', 'B', 'C', 'C']})

    policy = fit_label_policy(train, min_count=2)
    train_out, train_stats = apply_label_policy(train, policy)
    future_out, future_stats = apply_label_policy(future, policy)

    assert policy['fit_scope'] == 'training_only'
    assert policy['collapsed_labels'] == ['B']
    assert train_out['product'].value_counts().to_dict() == {'A': 5, 'Other': 1}
    assert future_out['product'].tolist() == ['A', 'Other', 'C', 'C']
    assert future_stats['novel_labels'] == ['C']
    assert future_stats['novel_target_rate'] == 0.5
    assert train_stats['novel_target_rate'] == 0.0
