from __future__ import annotations

import numpy as np
import pandas as pd

from .risk import apply_risk_controls


def prob_to_position(prob: float, long_th: float, half_th: float) -> float:
    if prob > long_th:
        return 1.0
    if prob > half_th:
        return 0.5
    return 0.0


def run_backtest(
    df_pred: pd.DataFrame,
    return_col: str,
    long_th: float,
    half_th: float,
    transaction_cost_bps: float,
    slippage_bps: float,
    vol_lookback: int = 20,
    target_vol_daily: float = 0.02,
    max_daily_loss: float = 0.02,
    max_drawdown: float = 0.12,
) -> tuple[pd.DataFrame, dict[str, float]]:
    out = df_pred.copy()
    out["desired_position"] = out["prob_up"].apply(lambda p: prob_to_position(p, long_th, half_th))

    out = apply_risk_controls(
        out,
        desired_position_col="desired_position",
        ret_col=return_col,
        vol_lookback=vol_lookback,
        target_vol_daily=target_vol_daily,
        max_daily_loss=max_daily_loss,
        max_drawdown=max_drawdown,
    )

    out["position"] = out["risk_position"]
    out["position_prev"] = out["position"].shift(1).fillna(0)
    out["turnover"] = (out["position"] - out["position_prev"]).abs()

    cost = out["turnover"] * (transaction_cost_bps + slippage_bps) / 10000.0
    out["gross_ret"] = out["position_prev"] * out[return_col]
    out["net_ret"] = out["gross_ret"] - cost
    out["equity"] = (1 + out["net_ret"]).cumprod()

    mean_daily = out["net_ret"].mean()
    std_daily = out["net_ret"].std(ddof=0)
    sharpe = (mean_daily / std_daily) * np.sqrt(365) if std_daily > 1e-12 else 0.0

    running_max = out["equity"].cummax()
    drawdown = out["equity"] / running_max - 1.0
    max_dd = drawdown.min()

    metrics = {
        "total_return": float(out["equity"].iloc[-1] - 1),
        "sharpe": float(sharpe),
        "max_drawdown": float(max_dd),
        "avg_turnover": float(out["turnover"].mean()),
        "daily_loss_triggers": float(out["daily_loss_trigger"].sum()),
        "dd_triggers": float(out["dd_trigger"].sum()),
    }
    return out, metrics
