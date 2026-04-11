# 占星预测 BTC 量化模型：数据表结构（SQL + Parquet）

## 1) 目录与分层

建议数据湖按 `bronze/silver/gold` 三层组织：

- `bronze`: 原始交易所与天文数据（不做业务逻辑）
- `silver`: 清洗对齐后的标准化时序数据
- `gold`: 直接喂给模型与回测的特征/标签数据

Parquet 分区建议：`symbol=BTCUSDT/freq=1d/year=YYYY/month=MM`。

---

## 2) SQL DDL（推荐 PostgreSQL）

### 2.1 行情主表 `market_ohlcv`

```sql
CREATE TABLE IF NOT EXISTS market_ohlcv (
  ts_utc            TIMESTAMPTZ NOT NULL,
  symbol            TEXT        NOT NULL,
  freq              TEXT        NOT NULL, -- 1d / 4h
  open              NUMERIC(20, 8) NOT NULL,
  high              NUMERIC(20, 8) NOT NULL,
  low               NUMERIC(20, 8) NOT NULL,
  close             NUMERIC(20, 8) NOT NULL,
  volume            NUMERIC(28, 8) NOT NULL,
  vwap              NUMERIC(20, 8),
  trade_count       BIGINT,
  source_exchange   TEXT        NOT NULL,
  ingestion_ts_utc  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  PRIMARY KEY (ts_utc, symbol, freq, source_exchange)
);

CREATE INDEX IF NOT EXISTS idx_market_ohlcv_symbol_freq_ts
  ON market_ohlcv(symbol, freq, ts_utc);
```

### 2.2 衍生品数据 `market_derivatives`

```sql
CREATE TABLE IF NOT EXISTS market_derivatives (
  ts_utc            TIMESTAMPTZ NOT NULL,
  symbol            TEXT        NOT NULL,
  freq              TEXT        NOT NULL,
  funding_rate      NUMERIC(12, 8),
  open_interest     NUMERIC(28, 8),
  basis             NUMERIC(12, 8),
  long_short_ratio  NUMERIC(12, 8),
  source_exchange   TEXT        NOT NULL,
  ingestion_ts_utc  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  PRIMARY KEY (ts_utc, symbol, freq, source_exchange)
);
```

### 2.3 占星（天文）事件表 `astro_events`

```sql
CREATE TABLE IF NOT EXISTS astro_events (
  ts_utc              TIMESTAMPTZ NOT NULL,
  body_a              TEXT        NOT NULL, -- sun/moon/mercury...
  body_b              TEXT,                 -- 相位时可为空/或另一行星
  event_type          TEXT        NOT NULL, -- aspect/retrograde/moon_phase
  aspect_deg          INT,                  -- 0/60/90/120/180
  orb_deg             NUMERIC(8, 4),
  is_retrograde       BOOLEAN,
  moon_phase_name     TEXT,
  moon_age_days       NUMERIC(8, 4),
  ecliptic_longitude  NUMERIC(10, 6),
  source              TEXT        NOT NULL,
  ingestion_ts_utc    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  PRIMARY KEY (ts_utc, body_a, COALESCE(body_b, ''), event_type, source)
);
```

### 2.4 特征表 `features_daily`

```sql
CREATE TABLE IF NOT EXISTS features_daily (
  ts_utc                    TIMESTAMPTZ NOT NULL,
  symbol                    TEXT        NOT NULL,

  -- 市场因子
  ret_1d                    NUMERIC(12, 8),
  ret_3d                    NUMERIC(12, 8),
  ret_5d                    NUMERIC(12, 8),
  vol_20d                   NUMERIC(12, 8),
  atr_14                    NUMERIC(12, 8),
  ma_5_gap                  NUMERIC(12, 8),
  ma_20_gap                 NUMERIC(12, 8),
  volume_z_20d              NUMERIC(12, 8),

  -- 占星因子（示例）
  sun_lon_sin               NUMERIC(12, 8),
  sun_lon_cos               NUMERIC(12, 8),
  moon_lon_sin              NUMERIC(12, 8),
  moon_lon_cos              NUMERIC(12, 8),
  is_new_moon               BOOLEAN,
  is_full_moon              BOOLEAN,
  is_mercury_retrograde     BOOLEAN,
  mars_saturn_square_orb    NUMERIC(12, 8),

  created_ts_utc            TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  PRIMARY KEY (ts_utc, symbol)
);
```

### 2.5 标签表 `labels_daily`

```sql
CREATE TABLE IF NOT EXISTS labels_daily (
  ts_utc                 TIMESTAMPTZ NOT NULL,
  symbol                 TEXT        NOT NULL,
  y_ret_fwd_1d           NUMERIC(12, 8),
  y_up_fwd_1d            SMALLINT,          -- 0/1
  y_ret_fwd_3d           NUMERIC(12, 8),
  y_up_fwd_3d            SMALLINT,
  PRIMARY KEY (ts_utc, symbol)
);
```

---

## 3) Parquet 字段定义（用于本地/云端训练）

### 3.1 `gold/model_input_daily.parquet`

| 字段 | 类型 | 说明 |
|---|---|---|
| ts_utc | timestamp[us, UTC] | 对齐后时间 |
| symbol | string | BTCUSDT |
| ret_1d~volume_z_20d | float64 | 市场因子 |
| sun_lon_sin~mars_saturn_square_orb | float64/bool | 占星因子 |
| y_up_fwd_1d | int8 | 分类标签 |
| y_ret_fwd_1d | float64 | 回归标签 |

建议附加元数据：`feature_version`, `label_horizon`, `tz=UTC`, `generated_at`。

---

## 4) 数据质量检查（落地必须）

- 主键去重：`(ts_utc, symbol)` 唯一。
- 时间连续性：日频缺口需记录并可追溯。
- 泄漏检测：任何标签列不能出现在特征计算窗口中。
- 统计漂移：每月输出 PSI/KL 指标并记录。

