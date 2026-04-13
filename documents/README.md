# 占星预测比特币价格走势量化模型

## 项目简介

本项目旨在通过占星数据预测比特币价格走势，结合现代量化交易技术，构建一个完整的预测和交易系统。项目使用占星学中的行星位置、相位关系等数据，结合比特币的历史价格、交易量等市场数据，通过机器学习模型进行预测，并生成交易信号。

## 项目架构

项目采用模块化设计，主要包含以下模块：

1. **数据获取模块**：负责获取占星数据和比特币价格数据
2. **数据处理模块**：负责数据清洗、特征提取和预处理
3. **模型训练模块**：负责训练和评估预测模型
4. **交易策略模块**：负责生成交易信号和执行交易
5. **可视化模块**：负责数据可视化和结果分析

### 目录结构

```
astro-bitcoin/
├── data/                 # 数据目录
│   ├── raw/             # 原始数据
│   ├── processed/       # 处理后的数据
│   ├── models/          # 模型文件
│   └── visualizations/  # 可视化结果
├── src/                 # 源代码目录
│   ├── data/            # 数据获取和处理模块
│   │   ├── __init__.py
│   │   ├── astro_data.py    # 占星数据获取
│   │   ├── bitcoin_data.py   # 比特币数据获取
│   │   └── data_processor.py # 数据处理
│   ├── models/          # 模型训练和预测模块
│   │   ├── __init__.py
│   │   └── model_trainer.py  # 模型训练器
│   ├── strategy/        # 交易策略模块
│   │   ├── __init__.py
│   │   └── strategy.py       # 交易策略
│   ├── visualization/   # 数据可视化模块
│   │   ├── __init__.py
│   │   └── visualizer.py     # 可视化工具
│   ├── utils/           # 工具函数
│   ├── config.py        # 配置文件
│   └── main.py          # 主程序
├── tests/               # 测试文件
│   └── test_model.py    # 模型测试
├── documents/           # 文档目录
│   └── README.md        # 项目文档
├── requirements.txt     # 依赖包
├── .env.example         # 环境变量示例
└── README.md            # 项目说明
```

## 模块说明

### 1. 数据获取模块

#### 占星数据获取 (`src/data/astro_data.py`)
- 负责获取行星位置、星座角度、相位关系等占星数据
- 使用AstroPy库计算行星位置
- 支持获取指定日期范围内的占星数据

#### 比特币数据获取 (`src/data/bitcoin_data.py`)
- 负责获取比特币的历史价格、交易量、市值等数据
- 使用CoinGecko API获取市场数据
- 支持获取当前价格和历史价格数据

### 2. 数据处理模块 (`src/data/data_processor.py`)
- 负责合并占星数据和比特币价格数据
- 提取特征，包括技术指标和占星特征
- 预处理数据，包括标准化和分割

### 3. 模型训练模块 (`src/models/model_trainer.py`)
- 支持多种机器学习模型，包括随机森林和XGBoost
- 负责模型训练、预测和评估
- 支持模型保存和加载

### 4. 交易策略模块 (`src/strategy/strategy.py`)
- 基于预测结果生成交易信号
- 支持模拟交易和实盘交易
- 实现风险管理策略，包括仓位控制、止损和止盈

### 5. 可视化模块 (`src/visualization/visualizer.py`)
- 绘制价格趋势图、预测结果图、回测结果图和特征重要性图
- 支持保存可视化结果

## 安装和使用

### 1. 安装依赖包

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env 文件，设置相关配置
```

### 3. 运行主程序

```bash
python src/main.py
```

## 配置说明

### 环境变量配置 (` .env`)

```
# 数据获取配置
COINGECKO_API_KEY=your_coingecko_api_key
BLOCKCHAIN_API_KEY=your_blockchain_api_key

# 数据库配置
DATABASE_URL=sqlite:///data.db

# 交易配置
TRADING_API_KEY=your_trading_api_key
TRADING_SECRET=your_trading_secret
TRADING_MODE=paper # paper 或 live

# 模型配置
MODEL_TYPE=xgboost # xgboost, random_forest
TRAINING_PERIOD=365 # 训练周期（天）
PREDICTION_HORIZON=7 # 预测 horizon（天）

# 风险管理配置
MAX_POSITION_SIZE=0.1 # 最大仓位比例
STOP_LOSS=0.05 # 止损比例
TAKE_PROFIT=0.1 # 止盈比例

# 日志配置
LOG_LEVEL=INFO
```

## 常见问题解答

### 1. 占星数据获取失败怎么办？
- 检查网络连接
- 确保AstroPy库安装正确
- 尝试减少获取数据的时间范围

### 2. 比特币数据获取失败怎么办？
- 检查网络连接
- 确保CoinGecko API密钥正确
- 尝试减少获取数据的时间范围

### 3. 模型训练失败怎么办？
- 检查数据质量
- 尝试调整模型参数
- 尝试使用不同的模型类型

### 4. 回测结果不理想怎么办？
- 尝试调整交易策略参数
- 尝试使用不同的特征组合
- 尝试使用不同的模型类型

### 5. 如何部署到生产环境？
- 使用Docker容器化部署
- 设置定时任务自动运行
- 配置监控和告警

## 注意事项

- 本项目仅用于研究和学习目的，不构成投资建议
- 占星预测比特币价格的有效性尚未得到科学验证
- 投资有风险，入市需谨慎
- 请确保遵守相关法律法规

## 技术栈

- 数据处理：pandas, numpy, scikit-learn
- 机器学习：xgboost, random_forest
- 数据获取：requests, astropy, ccxt
- 数据存储：sqlite3, psycopg2
- 可视化：matplotlib, seaborn, plotly
- 工具：pytz, python-dotenv, loguru

## 贡献

欢迎贡献代码和提出建议！请提交Pull Request或创建Issue。

## 许可证

本项目采用MIT许可证。
