# 占星预测比特币价格走势量化模型

## 项目简介

本项目旨在通过占星数据预测比特币价格走势，结合现代量化交易技术，构建一个完整的预测和交易系统。

## 项目结构

```
astro-bitcoin/
├── data/
│   ├── raw/         # 原始数据
│   ├── processed/   # 处理后的数据
├── src/
│   ├── data/        # 数据获取和处理模块
│   ├── models/      # 模型训练和预测模块
│   ├── strategy/    # 交易策略模块
│   ├── visualization/ # 数据可视化模块
│   ├── utils/       # 工具函数
│   ├── config.py    # 配置文件
│   └── main.py      # 主程序
├── tests/           # 测试文件
├── requirements.txt # 依赖包
└── README.md        # 项目说明
```

## 功能模块

1. **数据获取模块**：获取占星数据和比特币价格数据
2. **数据处理模块**：数据清洗、特征提取和预处理
3. **模型训练模块**：训练和评估预测模型
4. **交易策略模块**：生成交易信号和执行交易
5. **可视化模块**：数据可视化和结果分析

## 技术栈

- 数据处理：pandas, numpy, scikit-learn
- 机器学习：xgboost, keras, tensorflow
- 数据获取：requests, astropy, ccxt
- 数据存储：sqlite3, psycopg2
- 可视化：matplotlib, seaborn, plotly
- 工具：pytz, python-dotenv, loguru

## 安装和使用

1. 安装依赖包：
   ```bash
   pip install -r requirements.txt
   ```

2. 配置环境变量：
   ```bash
   cp .env.example .env
   # 编辑 .env 文件，设置相关配置
   ```

3. 运行主程序：
   ```bash
   python src/main.py
   ```

## 注意事项

- 本项目仅用于研究和学习目的，不构成投资建议
- 占星预测比特币价格的有效性尚未得到科学验证
- 投资有风险，入市需谨慎
