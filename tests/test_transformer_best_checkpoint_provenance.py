from pathlib import Path


def test_transformer_training_uses_best_checkpoint_settings():
    source = Path("src/model_quality/models/train_transformer.py").read_text(
        encoding="utf-8"
    )

    assert "save_strategy='epoch'" in source
    assert "'load_best_model_at_end': True" in source
    assert "'metric_for_best_model': 'macro_f1'" in source
    assert "'greater_is_better': True" in source
    assert "save_total_limit=2" in source


def test_transformer_training_writes_provenance_artifact():
    source = Path("src/model_quality/models/train_transformer.py").read_text(
        encoding="utf-8"
    )

    assert "transformer_training_provenance.json" in source
    assert "'checkpoint_selection_split': 'calibration'" in source
    assert "'framework_model_selection_split': 'selection'" in source
    assert "'final_evaluation_split': 'test'" in source
    assert "'test_used_for_checkpoint_selection': False" in source
    assert "'test_used_for_framework_selection': False" in source
    assert "'test_used_for_reselection': False" in source
