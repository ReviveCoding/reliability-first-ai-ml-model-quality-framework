from __future__ import annotations

import warnings
from dataclasses import dataclass

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

try:
    from lightgbm import LGBMClassifier
except Exception:
    LGBMClassifier = None

@dataclass
class TrainedTabularModel:
    name: str
    pipeline: object
    label_encoder: LabelEncoder
    target_col: str
    feature_cols: list[str]

    def predict(self, df: pd.DataFrame):
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='X does not have valid feature names.*')
            pred = self.pipeline.predict(df[self.feature_cols])
        return self.label_encoder.inverse_transform(pred)

    def predict_proba(self, df: pd.DataFrame):
        if hasattr(self.pipeline, 'predict_proba'):
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message='X does not have valid feature names.*')
                return self.pipeline.predict_proba(df[self.feature_cols])
        return None

    def save(self, path):
        joblib.dump(self, path)


def train_lightgbm_or_fallback(train_df: pd.DataFrame, target_col='timely_response', random_state: int = 7) -> TrainedTabularModel:
    feature_cols = [c for c in ['product','issue','company_response_to_consumer','state','submitted_via'] if c in train_df.columns]
    le = LabelEncoder()
    y = le.fit_transform(train_df[target_col].astype(str))
    pre = ColumnTransformer([
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('ohe', OneHotEncoder(handle_unknown='ignore'))]), feature_cols)
    ])
    if LGBMClassifier is not None:
        clf = LGBMClassifier(n_estimators=12, learning_rate=0.08, max_depth=4, num_leaves=15, n_jobs=1, random_state=random_state, class_weight='balanced', verbose=-1, force_col_wise=True)
        name = 'lightgbm'
    else:
        clf = HistGradientBoostingClassifier(max_iter=20, max_leaf_nodes=15, random_state=random_state)
        name = 'hist_gradient_boosting_fallback'
    pipe = Pipeline([('pre', pre), ('clf', clf)])
    pipe.fit(train_df[feature_cols], y)
    return TrainedTabularModel(name, pipe, le, target_col, feature_cols)
