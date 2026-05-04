from __future__ import annotations

import numpy as np
import pandas as pd


REQUIRED_MARKET_COLUMNS = {"ts_utc", "close", "volume"}
REQUIRED_ASTRO_COLUMNS = {
    "ts_utc",
    "sun_longitude_deg",
    "moon_longitude_deg",
    "moon_age_days",
    "is_mercury_retrograde",
}


def _ensure_required_columns(df: pd.DataFrame, required: set[str], table: str) -> None:
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{table} 缺少字段: {missing}")


def load_market_csv(path: str, symbol: str = "BTCUSDT") -> pd.DataFrame:
    df = pd.read_csv(path)
    _ensure_required_columns(df, REQUIRED_MARKET_COLUMNS, "market")

    out = df.copy()
    out["ts_utc"] = pd.to_datetime(out["ts_utc"], utc=True)
    out = out.sort_values("ts_utc").reset_index(drop=True)
    out["symbol"] = out.get("symbol", symbol)
    return out


def load_astro_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    _ensure_required_columns(df, REQUIRED_ASTRO_COLUMNS, "astro")

    out = df.copy()
    out["ts_utc"] = pd.to_datetime(out["ts_utc"], utc=True)
    return out.sort_values("ts_utc").reset_index(drop=True)


def build_features_labels_from_raw(
    market_df: pd.DataFrame,
    astro_df: pd.DataFrame,
    horizon_days: int = 1,
) -> pd.DataFrame:
    df = pd.merge(market_df, astro_df, on="ts_utc", how="inner", validate="one_to_one")
    df = df.sort_values("ts_utc").reset_index(drop=True)

    close = df["close"].astype(float)
    df["ret_1d"] = np.log(close / close.shift(1)).fillna(0)
    df["ret_3d"] = np.log(close / close.shift(3)).fillna(0)
    df["ret_5d"] = np.log(close / close.shift(5)).fillna(0)
    df["vol_20d"] = df["ret_1d"].rolling(20).std().fillna(df["ret_1d"].std())
    ma5 = close.rolling(5).mean()
    ma20 = close.rolling(20).mean()
    df["ma_5_gap"] = ((close - ma5) / ma5).replace([np.inf, -np.inf], 0).fillna(0)
    df["ma_20_gap"] = ((close - ma20) / ma20).replace([np.inf, -np.inf], 0).fillna(0)

    vol = df["volume"].astype(float)
    vol_roll_mean = vol.rolling(20).mean()
    vol_roll_std = vol.rolling(20).std()
    df["volume_z_20d"] = ((vol - vol_roll_mean) / vol_roll_std).replace([np.inf, -np.inf], 0).fillna(0)

    # 占星因子
    df["sun_lon_sin"] = np.sin(np.deg2rad(df["sun_longitude_deg"].astype(float)))
    df["sun_lon_cos"] = np.cos(np.deg2rad(df["sun_longitude_deg"].astype(float)))
    df["moon_lon_sin"] = np.sin(np.deg2rad(df["moon_longitude_deg"].astype(float)))
    df["moon_lon_cos"] = np.cos(np.deg2rad(df["moon_longitude_deg"].astype(float)))
    moon_age = df["moon_age_days"].astype(float)
    df["is_new_moon"] = (moon_age < 1.0).astype(int)
    df["is_full_moon"] = (np.abs(moon_age - 14.76) < 1.0).astype(int)
    df["is_mercury_retrograde"] = df["is_mercury_retrograde"].astype(int)

    # 标签
    df["y_ret_fwd_1d"] = df["ret_1d"].shift(-horizon_days).fillna(0)
    df["y_up_fwd_1d"] = (df["y_ret_fwd_1d"] > 0).astype(int)

    return df
