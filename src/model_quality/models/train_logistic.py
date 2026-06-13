from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder


@dataclass
class TemperatureScaledTextPipeline:
    base_pipeline: object
    temperature: float

    def predict_proba(self, texts):
        base_proba = np.asarray(self.base_pipeline.predict_proba(texts), dtype=float)
        logits = np.log(np.clip(base_proba, 1e-12, 1.0)) / max(float(self.temperature), 1e-6)
        logits -= logits.max(axis=1, keepdims=True)
        scaled = np.exp(logits)
        return scaled / scaled.sum(axis=1, keepdims=True)

    def predict(self, texts):
        return np.argmax(self.predict_proba(texts), axis=1)


@dataclass
class TrainedSklearnModel:
    name: str
    pipeline: object
    label_encoder: LabelEncoder
    target_col: str
    text_col: str

    def predict(self, df: pd.DataFrame):
        return self.label_encoder.inverse_transform(self.pipeline.predict(df[self.text_col].fillna('').astype(str)))

    def predict_proba(self, df: pd.DataFrame):
        if hasattr(self.pipeline, 'predict_proba'):
            return self.pipeline.predict_proba(df[self.text_col].fillna('').astype(str))
        return None

    def save(self, path):
        joblib.dump(self, path)


def _make_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2, sublinear_tf=True)


def _make_base_logistic(random_state: int = 7) -> LogisticRegression:
    return LogisticRegression(
        max_iter=300,
        class_weight='balanced',
        random_state=random_state,
        solver='lbfgs',
    )


def train_tfidf_logistic(
    train_df: pd.DataFrame,
    text_col: str = 'consumer_complaint_narrative',
    target_col: str = 'product',
    random_state: int = 7,
) -> TrainedSklearnModel:
    le = LabelEncoder()
    y = le.fit_transform(train_df[target_col].astype(str))
    pipe = Pipeline([
        ('tfidf', _make_vectorizer()),
        ('clf', _make_base_logistic(random_state=random_state)),
    ])
    pipe.fit(train_df[text_col].fillna('').astype(str), y)
    return TrainedSklearnModel('tfidf_logistic', pipe, le, target_col, text_col)


def _fit_temperature(base_model: TrainedSklearnModel, calibration_df: pd.DataFrame) -> float:
    le = base_model.label_encoder
    known = set(map(str, le.classes_))
    cal = calibration_df[calibration_df[base_model.target_col].astype(str).isin(known)].copy()
    if cal.empty or cal[base_model.target_col].astype(str).nunique() < 2:
        raise ValueError('Held-out calibration requires at least two known target classes.')
    y_cal = le.transform(cal[base_model.target_col].astype(str))
    base_proba = np.asarray(base_model.predict_proba(cal), dtype=float)

    def objective(log_temperature: float) -> float:
        temperature = float(np.exp(log_temperature))
        logits = np.log(np.clip(base_proba, 1e-12, 1.0)) / temperature
        logits -= logits.max(axis=1, keepdims=True)
        proba = np.exp(logits)
        proba /= proba.sum(axis=1, keepdims=True)
        return float(log_loss(y_cal, proba, labels=np.arange(base_proba.shape[1])))

    result = minimize_scalar(objective, bounds=(np.log(0.2), np.log(5.0)), method='bounded')
    return float(np.exp(result.x)) if result.success else 1.0


def train_calibrated_tfidf_logistic(
    train_df: pd.DataFrame,
    text_col: str = 'consumer_complaint_narrative',
    target_col: str = 'product',
    random_state: int = 7,
    calibration_df: pd.DataFrame | None = None,
    base_model: TrainedSklearnModel | None = None,
) -> TrainedSklearnModel:
    """Create a calibrated challenger.

    With a validation set, scalar temperature scaling is used. It is fast,
    reuses the fitted base model, and remains valid when the holdout omits a
    rare training class. A compact CV sigmoid fallback is retained when no
    held-out calibration set is provided.
    """
    if calibration_df is not None:
        base_model = base_model or train_tfidf_logistic(
            train_df, text_col=text_col, target_col=target_col, random_state=random_state
        )
        temperature = _fit_temperature(base_model, calibration_df)
        scaled_pipeline = TemperatureScaledTextPipeline(base_model.pipeline, temperature)
        return TrainedSklearnModel(
            'tfidf_logistic_temperature_scaled', scaled_pipeline,
            base_model.label_encoder, target_col, text_col,
        )

    le = LabelEncoder()
    y = le.fit_transform(train_df[target_col].astype(str))
    counts = pd.Series(y).value_counts()
    cv = min(3, int(counts.min()))
    if cv < 2:
        raise ValueError('Not enough examples per class to calibrate logistic regression.')
    base = _make_base_logistic(random_state=random_state)
    pipe = Pipeline([
        ('tfidf', _make_vectorizer()),
        ('clf', CalibratedClassifierCV(base, method='sigmoid', cv=cv)),
    ])
    pipe.fit(train_df[text_col].fillna('').astype(str), y)
    return TrainedSklearnModel('tfidf_logistic_calibrated_cv', pipe, le, target_col, text_col)
