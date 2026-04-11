from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class TrainResult:
    model: Pipeline
    train_rows: int


def train_classifier(df_train: pd.DataFrame, feature_cols: list[str], label_col: str) -> TrainResult:
    x_train = df_train[feature_cols].copy()
    y_train = df_train[label_col].astype(int)

    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1200, random_state=42)),
        ]
    )
    pipe.fit(x_train, y_train)
    return TrainResult(model=pipe, train_rows=len(df_train))
