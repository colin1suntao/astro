# quant_astro_btc

用于“市场因子 + 占星因子”预测 BTC 方向的最小可行脚手架。

## 快速开始

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=src python scripts/train_walk_forward.py --use-demo
```

## 接入真实数据（第4步）

```bash
PYTHONPATH=src python scripts/train_walk_forward.py \
  --market-csv data/market_daily.csv \
  --astro-csv data/astro_daily.csv
```

### `market_daily.csv` 最低字段
- `ts_utc` (ISO 时间)
- `close`
- `volume`
- `symbol` (可选)

### `astro_daily.csv` 最低字段
- `ts_utc`
- `sun_longitude_deg`
- `moon_longitude_deg`
- `moon_age_days`
- `is_mercury_retrograde` (0/1)

## 风控与纸交易（第5步）

默认启用：
- 波动率目标仓位（`risk.target_vol_daily`）
- 单日亏损触发次日减仓（`risk.max_daily_loss`）
- 最大回撤保护模式（`risk.max_drawdown`）

并输出：
- `artifacts/oof_predictions.csv`
- `artifacts/backtest_trades.csv`
- `artifacts/paper_orders.csv`（纸交易订单日志）
