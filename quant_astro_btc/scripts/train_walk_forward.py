from __future__ import annotations

import argparse
import json
import pathlib

from quant_astro_btc.backtest import run_backtest
from quant_astro_btc.data import build_demo_dataset
from quant_astro_btc.io import build_features_labels_from_raw, load_astro_csv, load_market_csv
from quant_astro_btc.paper import generate_paper_orders
from quant_astro_btc.walkforward import walk_forward_predict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train walk-forward astro+market BTC model")
    parser.add_argument("--config", default="config/default.json", help="Path to JSON config")
    parser.add_argument("--market-csv", default=None, help="Path to market CSV")
    parser.add_argument("--astro-csv", default=None, help="Path to astro CSV")
    parser.add_argument("--use-demo", action="store_true", help="Use synthetic demo data")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = pathlib.Path(__file__).resolve().parents[1]
    cfg = json.loads((root / args.config).read_text(encoding="utf-8"))

    if args.use_demo:
        df = build_demo_dataset(seed=cfg["seed"])
        df["symbol"] = cfg["symbol"]
    else:
        if not args.market_csv or not args.astro_csv:
            raise ValueError("非 demo 模式必须提供 --market-csv 和 --astro-csv")
        market = load_market_csv(args.market_csv, symbol=cfg["symbol"])
        astro = load_astro_csv(args.astro_csv)
        df = build_features_labels_from_raw(market, astro, horizon_days=cfg["horizon_days"])

    preds = walk_forward_predict(
        df=df,
        feature_cols=cfg["feature_cols"],
        label_col=cfg["label_col"],
        rolling_train_days=cfg["rolling_train_days"],
        rolling_test_days=cfg["rolling_test_days"],
    )

    preds = preds.merge(df[["ts_utc", "symbol"]], on="ts_utc", how="left")

    trades, metrics = run_backtest(
        df_pred=preds,
        return_col=cfg["return_col"],
        long_th=cfg["signal_long_threshold"],
        half_th=cfg["signal_half_threshold"],
        transaction_cost_bps=cfg["transaction_cost_bps"],
        slippage_bps=cfg["slippage_bps"],
        vol_lookback=cfg["risk"]["vol_lookback"],
        target_vol_daily=cfg["risk"]["target_vol_daily"],
        max_daily_loss=cfg["risk"]["max_daily_loss"],
        max_drawdown=cfg["risk"]["max_drawdown"],
    )
    orders = generate_paper_orders(trades, target_pos_col="position", symbol_col="symbol")

    out_dir = root / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    preds.to_csv(out_dir / "oof_predictions.csv", index=False)
    trades.to_csv(out_dir / "backtest_trades.csv", index=False)
    orders.to_csv(out_dir / "paper_orders.csv", index=False)

    print("=== Backtest Metrics ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.6f}")
    print(f"paper_orders: {len(orders)}")


if __name__ == "__main__":
    main()
