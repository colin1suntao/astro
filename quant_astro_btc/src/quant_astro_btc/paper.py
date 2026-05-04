from __future__ import annotations

import pandas as pd


def generate_paper_orders(
    df: pd.DataFrame,
    ts_col: str = "ts_utc",
    target_pos_col: str = "position",
    symbol_col: str = "symbol",
) -> pd.DataFrame:
    """将目标仓位变化转换为纸交易订单日志。"""
    out = df.copy()
    if symbol_col not in out.columns:
        out[symbol_col] = "BTCUSDT"

    out["prev_target"] = out[target_pos_col].shift(1).fillna(0)
    out["delta"] = out[target_pos_col] - out["prev_target"]

    orders = out.loc[out["delta"].abs() > 1e-10, [ts_col, symbol_col, "prev_target", target_pos_col, "delta"]].copy()
    orders["side"] = orders["delta"].apply(lambda x: "BUY" if x > 0 else "SELL")
    orders["order_type"] = "MKT"
    orders["status"] = "SIMULATED_FILLED"
    orders = orders.rename(
        columns={
            ts_col: "order_ts_utc",
            symbol_col: "symbol",
            target_pos_col: "target_position",
            "prev_target": "prev_position",
            "delta": "qty_fraction",
        }
    )
    return orders.reset_index(drop=True)
