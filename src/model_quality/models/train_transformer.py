from __future__ import annotations

import inspect
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class TransformerTrainResult:
    enabled: bool
    model_name: str
    device: str
    metrics: dict
    selection_metrics: dict | None = None
    artifact_dir: str | None = None
    note: str = ''


def detect_device(device: str = 'auto') -> str:
    requested = str(device).lower()
    try:
        import torch
    except Exception:
        return 'cpu' if requested == 'auto' else requested
    if requested == 'auto':
        if torch.cuda.is_available():
            return 'cuda'
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
        return 'cpu'
    if requested.startswith('cuda'):
        return 'cuda' if torch.cuda.is_available() else 'unavailable-cuda'
    if requested == 'mps':
        available = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        return 'mps' if available else 'unavailable-mps'
    return 'cpu'


def _training_arguments(TrainingArguments, *, out_dir: str, epochs: int, batch_size: int, device: str, seed: int):
    params = inspect.signature(TrainingArguments.__init__).parameters
    kwargs = dict(
        output_dir=out_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        save_strategy='epoch',
        save_total_limit=2,
        logging_steps=50,
        report_to=[],
        seed=seed,
        data_seed=seed,
        dataloader_num_workers=0,
        fp16=(device == 'cuda'),
    )
    best_checkpoint_kwargs = {
        'load_best_model_at_end': True,
        'metric_for_best_model': 'macro_f1',
        'greater_is_better': True,
    }
    for key, value in best_checkpoint_kwargs.items():
        if key in params:
            kwargs[key] = value
    if 'eval_strategy' in params:
        kwargs['eval_strategy'] = 'epoch'
    else:
        kwargs['evaluation_strategy'] = 'epoch'
    if 'use_cpu' in params:
        kwargs['use_cpu'] = device == 'cpu'
    elif 'no_cuda' in params:
        kwargs['no_cuda'] = device == 'cpu'
    return TrainingArguments(**kwargs)


def train_transformer_classifier(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    selection_df: pd.DataFrame | None = None,
    test_df: pd.DataFrame | None = None,
    text_col: str = 'consumer_complaint_narrative',
    target_col: str = 'product',
    model_name: str = 'distilbert-base-uncased',
    out_dir: str = 'outputs/transformer_model',
    epochs: int = 1,
    batch_size: int = 16,
    device: str = 'auto',
    random_state: int = 7,
) -> TransformerTrainResult:
    """Train an optional GPU-capable closed-set text classifier.

    ``val_df`` is used only as the Trainer feedback window. Champion selection
    metrics are computed on ``selection_df`` and final metrics on ``test_df``.
    This mirrors the classical model path and avoids choosing a champion on the
    final test set.
    """
    try:
        import numpy as np
        from datasets import Dataset
        from sklearn.metrics import accuracy_score, average_precision_score, f1_score, log_loss, roc_auc_score
        from sklearn.preprocessing import LabelEncoder
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

        from model_quality.evaluation.calibration_metrics import expected_calibration_error
    except Exception as exc:
        return TransformerTrainResult(False, model_name, detect_device(device), {}, note=f'Skipped transformer training: {type(exc).__name__}: {exc}')

    used_device = detect_device(device)
    if used_device.startswith('unavailable-'):
        return TransformerTrainResult(False, model_name, used_device, {}, note=f'Requested device {device!r} is unavailable.')

    le = LabelEncoder()
    le.fit(train_df[target_col].astype(str))
    known = set(map(str, le.classes_))

    def prepare_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, float]:
        labels = frame[target_col].astype(str)
        mask = labels.isin(known)
        return frame.loc[mask].copy(), float(1.0 - mask.mean()) if len(frame) else 0.0

    val_known, val_unknown_rate = prepare_frame(val_df)
    selection_source = selection_df if selection_df is not None else val_df
    selection_known, selection_unknown_rate = prepare_frame(selection_source)
    final_source = test_df if test_df is not None else selection_source
    test_known, test_unknown_rate = prepare_frame(final_source)
    if val_known.empty or selection_known.empty or test_known.empty:
        return TransformerTrainResult(
            False,
            model_name,
            used_device,
            {},
            note='Skipped transformer training because an evaluation window contains no labels observed in training.',
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(le.classes_))

    def to_dataset(frame: pd.DataFrame):
        labels = le.transform(frame[target_col].astype(str))
        ds = Dataset.from_dict({'text': frame[text_col].fillna('').astype(str).tolist(), 'label': labels.tolist()})

        def tok(batch):
            return tokenizer(batch['text'], truncation=True, padding='max_length', max_length=128)

        ds = ds.map(tok, batched=True)
        ds.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
        return ds

    train_ds = to_dataset(train_df)
    val_ds = to_dataset(val_known)
    selection_ds = to_dataset(selection_known)
    test_ds = to_dataset(test_known)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            'accuracy': float(accuracy_score(labels, preds)),
            'macro_f1': float(f1_score(labels, preds, average='macro', zero_division=0)),
        }

    def prediction_metrics(prediction, source_df: pd.DataFrame, unknown_rate: float, total_rows: int) -> dict:
        logits = np.asarray(prediction.predictions)
        logits = logits - logits.max(axis=1, keepdims=True)
        proba = np.exp(logits)
        proba = proba / proba.sum(axis=1, keepdims=True)
        y_true = np.asarray(prediction.label_ids)
        y_pred = np.argmax(proba, axis=1)
        metrics = {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'macro_f1': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
            'unknown_target_rate': float(unknown_rate),
            'total_rows': int(total_rows),
            'evaluated_rows': int(len(source_df)),
            'evaluation_class_coverage': float(len(np.unique(y_true)) / max(1, len(le.classes_))),
        }
        try:
            if proba.shape[1] == 2:
                metrics['roc_auc'] = float(roc_auc_score(y_true, proba[:, 1]))
                metrics['pr_auc'] = float(average_precision_score(y_true, proba[:, 1]))
            else:
                roc_values = []
                ap_values = []
                for i in np.unique(y_true):
                    binary = (y_true == i).astype(int)
                    if binary.min() == binary.max():
                        continue
                    roc_values.append(roc_auc_score(binary, proba[:, i]))
                    ap_values.append(average_precision_score(binary, proba[:, i]))
                metrics['roc_auc'] = float(np.mean(roc_values)) if roc_values else None
                metrics['pr_auc'] = float(np.mean(ap_values)) if ap_values else None
        except ValueError:
            metrics['roc_auc'] = None
            metrics['pr_auc'] = None
        ece, _ = expected_calibration_error(y_true, proba)
        onehot = np.eye(proba.shape[1])[y_true]
        metrics['ece'] = float(ece)
        metrics['brier'] = float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))
        metrics['log_loss'] = float(log_loss(y_true, proba, labels=np.arange(proba.shape[1])))

        if 'state' in source_df.columns:
            slice_scores = []
            eval_with_pred = source_df.copy()
            eval_with_pred['_pred'] = le.inverse_transform(y_pred)
            for _, group in eval_with_pred.groupby('state', dropna=False):
                if len(group) >= 5:
                    slice_scores.append(f1_score(group[target_col].astype(str), group['_pred'], average='macro', zero_division=0))
            metrics['worst_slice_f1'] = float(min(slice_scores)) if slice_scores else None
        else:
            metrics['worst_slice_f1'] = None

        rng = np.random.default_rng(random_state + 97)
        bootstrap_scores = []
        if len(y_true):
            for _ in range(300):
                idx = rng.integers(0, len(y_true), size=len(y_true))
                bootstrap_scores.append(
                    f1_score(
                        y_true[idx], y_pred[idx], labels=np.arange(proba.shape[1]),
                        average='macro', zero_division=0,
                    )
                )
        if bootstrap_scores:
            metrics['macro_f1_ci_low'] = float(np.quantile(bootstrap_scores, 0.025))
            metrics['macro_f1_ci_high'] = float(np.quantile(bootstrap_scores, 0.975))
            metrics['bootstrap_resamples'] = int(len(bootstrap_scores))
        else:
            metrics['macro_f1_ci_low'] = None
            metrics['macro_f1_ci_high'] = None
            metrics['bootstrap_resamples'] = 0
        return metrics

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    args = _training_arguments(
        TrainingArguments,
        out_dir=out_dir,
        epochs=epochs,
        batch_size=batch_size,
        device=used_device,
        seed=random_state,
    )
    trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds, compute_metrics=compute_metrics)
    trainer.train()

    selection_metrics = prediction_metrics(
        trainer.predict(selection_ds), selection_known, selection_unknown_rate, len(selection_source)
    )
    test_metrics = prediction_metrics(
        trainer.predict(test_ds), test_known, test_unknown_rate, len(final_source)
    )
    test_metrics['validation_unknown_target_rate'] = val_unknown_rate

    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)
    (Path(out_dir) / 'label_classes.json').write_text(json.dumps(le.classes_.tolist(), indent=2), encoding='utf-8')

    def _json_safe(value):
        if value is None:
            return None
        try:
            import math

            numeric_value = float(value)
            if math.isnan(numeric_value) or math.isinf(numeric_value):
                return None
            return numeric_value
        except (TypeError, ValueError):
            return str(value)

    eval_strategy_value = getattr(
        args,
        'eval_strategy',
        getattr(args, 'evaluation_strategy', None),
    )
    save_strategy_value = getattr(args, 'save_strategy', None)

    transformer_training_provenance = {
        'schema_version': 1,
        'model': 'transformer_text_classifier',
        'base_model': model_name,
        'training_split': 'train',
        'checkpoint_selection_split': 'calibration',
        'framework_model_selection_split': 'selection',
        'final_evaluation_split': 'test',
        'trainer_eval_dataset': 'calibration',
        'checkpoint_metric': 'macro_f1',
        'metric_for_best_model': getattr(args, 'metric_for_best_model', None),
        'greater_is_better': bool(getattr(args, 'greater_is_better', True)),
        'load_best_model_at_end': bool(getattr(args, 'load_best_model_at_end', False)),
        'save_strategy': str(save_strategy_value),
        'evaluation_strategy': str(eval_strategy_value),
        'save_total_limit': getattr(args, 'save_total_limit', None),
        'best_model_checkpoint': getattr(trainer.state, 'best_model_checkpoint', None),
        'best_checkpoint_macro_f1': _json_safe(getattr(trainer.state, 'best_metric', None)),
        'framework_selection_macro_f1': _json_safe(selection_metrics.get('macro_f1')),
        'candidate_test_macro_f1': _json_safe(test_metrics.get('macro_f1')),
        'candidate_test_pr_auc': _json_safe(test_metrics.get('pr_auc')),
        'candidate_test_ece': _json_safe(test_metrics.get('ece')),
        'candidate_test_worst_slice_f1': _json_safe(test_metrics.get('worst_slice_f1')),
        'selection_rows_total': int(len(selection_source)),
        'selection_rows_evaluated': int(len(selection_known)),
        'test_rows_total': int(len(final_source)),
        'test_rows_evaluated': int(len(test_known)),
        'validation_unknown_target_rate': _json_safe(val_unknown_rate),
        'selection_unknown_target_rate': _json_safe(selection_unknown_rate),
        'test_unknown_target_rate': _json_safe(test_unknown_rate),
        'test_used_for_checkpoint_selection': False,
        'test_used_for_framework_selection': False,
        'test_used_for_reselection': False,
    }

    provenance_json = json.dumps(
        transformer_training_provenance,
        indent=2,
        allow_nan=False,
    )

    provenance_paths = [
        Path(out_dir) / 'transformer_training_provenance.json',
        Path(out_dir).parent / 'transformer_training_provenance.json',
    ]
    for provenance_path in provenance_paths:
        provenance_path.write_text(provenance_json + '\n', encoding='utf-8')

    note = (
        f'Selection metrics used {len(selection_known)}/{len(selection_source)} known-label rows; '
        f'final metrics used {len(test_known)}/{len(final_source)} rows on device={used_device}.'
    )
    return TransformerTrainResult(
        True,
        model_name,
        used_device,
        test_metrics,
        selection_metrics=selection_metrics,
        artifact_dir=out_dir,
        note=note,
    )
