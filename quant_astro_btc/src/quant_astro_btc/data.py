from __future__ import annotations

import numpy as np
import pandas as pd


def build_demo_dataset(n_days: int = 1800, seed: int = 42) -> pd.DataFrame:
    """生成可直接跑通的演示数据（市场 + 占星 + 标签）。"""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2019-01-01", periods=n_days, freq="D", tz="UTC")

    noise = rng.normal(0, 0.02, size=n_days)
    season = 0.0008 * np.sin(np.arange(n_days) * 2 * np.pi / 28)
    ret = noise + season

    df = pd.DataFrame({"ts_utc": dates})
    df["ret_1d"] = ret
    df["ret_3d"] = pd.Series(ret).rolling(3).sum().fillna(0).values
    df["ret_5d"] = pd.Series(ret).rolling(5).sum().fillna(0).values
    df["vol_20d"] = pd.Series(ret).rolling(20).std().fillna(pd.Series(ret).std()).values
    df["ma_5_gap"] = pd.Series(ret).rolling(5).mean().fillna(0).values
    df["ma_20_gap"] = pd.Series(ret).rolling(20).mean().fillna(0).values
    vol = rng.lognormal(mean=10, sigma=0.2, size=n_days)
    df["volume_z_20d"] = (pd.Series(vol) - pd.Series(vol).rolling(20).mean()) / pd.Series(vol).rolling(20).std()
    df["volume_z_20d"] = df["volume_z_20d"].replace([np.inf, -np.inf], 0).fillna(0)

    # 占星周期特征（示例化）
    sun_theta = np.arange(n_days) * (2 * np.pi / 365.25)
    moon_theta = np.arange(n_days) * (2 * np.pi / 29.53)
    df["sun_lon_sin"] = np.sin(sun_theta)
    df["sun_lon_cos"] = np.cos(sun_theta)
    df["moon_lon_sin"] = np.sin(moon_theta)
    df["moon_lon_cos"] = np.cos(moon_theta)

    moon_age = np.mod(np.arange(n_days), 29.53)
    df["is_new_moon"] = (moon_age < 1.0).astype(int)
    df["is_full_moon"] = (np.abs(moon_age - 14.76) < 1.0).astype(int)
    df["is_mercury_retrograde"] = ((np.arange(n_days) % 116) < 21).astype(int)

    # 标签：让部分信号与特征相关，便于模板可观测
    alpha = (
        0.18 * df["ret_3d"].values
        + 0.05 * df["moon_lon_sin"].values
        - 0.04 * df["is_mercury_retrograde"].values
    )
    y_ret = alpha + rng.normal(0, 0.015, size=n_days)
    df["y_ret_fwd_1d"] = pd.Series(y_ret).shift(-1).fillna(0)
    df["y_up_fwd_1d"] = (df["y_ret_fwd_1d"] > 0).astype(int)

    return df
