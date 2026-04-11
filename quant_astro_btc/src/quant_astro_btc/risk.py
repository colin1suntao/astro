from __future__ import annotations

import pandas as pd


def apply_risk_controls(
    df: pd.DataFrame,
    desired_position_col: str,
    ret_col: str,
    vol_lookback: int,
    target_vol_daily: float,
    max_daily_loss: float,
    max_drawdown: float,
) -> pd.DataFrame:
    """基于波动率目标与熔断规则，输出风险调整后仓位。"""
    out = df.copy()

    rolling_vol = out[ret_col].rolling(vol_lookback).std().fillna(out[ret_col].std())
    vol_scale = (target_vol_daily / rolling_vol).clip(lower=0.0, upper=1.5)
    out["risk_position"] = (out[desired_position_col] * vol_scale).clip(lower=0.0, upper=1.0)

    # 使用前一日仓位估算净值与风控触发
    out["risk_position_prev"] = out["risk_position"].shift(1).fillna(0)
    out["risk_gross_ret"] = out["risk_position_prev"] * out[ret_col]
    out["risk_equity"] = (1 + out["risk_gross_ret"]).cumprod()

    out["risk_running_max"] = out["risk_equity"].cummax()
    out["risk_drawdown"] = out["risk_equity"] / out["risk_running_max"] - 1.0

    # 日亏损超过阈值 -> 次日减半
    out["daily_loss_trigger"] = (out["risk_gross_ret"] < -abs(max_daily_loss)).astype(int)
    mask_next_day = out["daily_loss_trigger"].shift(1).fillna(0).astype(bool)
    out.loc[mask_next_day, "risk_position"] = out.loc[mask_next_day, "risk_position"] * 0.5

    # 回撤超过阈值 -> 保护模式（强制空仓）
    out["dd_trigger"] = (out["risk_drawdown"] < -abs(max_drawdown)).astype(int)
    out.loc[out["dd_trigger"] == 1, "risk_position"] = 0.0

    return out
