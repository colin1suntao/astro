from __future__ import annotations

import pandas as pd

from .model import train_classifier


def walk_forward_predict(
    df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str,
    rolling_train_days: int,
    rolling_test_days: int,
) -> pd.DataFrame:
    df = df.sort_values("ts_utc").reset_index(drop=True)

    all_preds: list[pd.DataFrame] = []
    start = 0

    while True:
        train_end = start + rolling_train_days
        test_end = train_end + rolling_test_days
        if test_end > len(df):
            break

        df_train = df.iloc[start:train_end].copy()
        df_test = df.iloc[train_end:test_end].copy()

        result = train_classifier(df_train, feature_cols, label_col)
        probs = result.model.predict_proba(df_test[feature_cols])[:, 1]

        pred = df_test[["ts_utc", label_col, "y_ret_fwd_1d"]].copy()
        pred["prob_up"] = probs
        all_preds.append(pred)

        start += rolling_test_days

    if not all_preds:
        raise ValueError("数据不足，无法执行 walk-forward。")

    return pd.concat(all_preds, ignore_index=True)
