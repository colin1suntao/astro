import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from loguru import logger
import os
from src.config import config
from src.data.astro_data import AstroData
from src.data.bitcoin_data import BitcoinData
from src.data.data_processor import DataProcessor
from src.models.model_trainer import ModelTrainer
from src.strategy.strategy import TradingStrategy
from src.visualization.visualizer import Visualizer

class AstroBitcoinModel:
    """占星预测比特币价格走势的量化模型"""
    
    def __init__(self):
        """初始化"""
        self.astro_data = AstroData()
        self.bitcoin_data = BitcoinData()
        self.data_processor = DataProcessor()
        self.model_trainer = ModelTrainer(model_type=config.MODEL_TYPE)
        self.strategy = TradingStrategy()
        self.visualizer = Visualizer()
    
    def run(self):
        """运行模型"""
        try:
            # 1. 获取数据
            logger.info("开始获取数据...")
            
            # 生成更长时间的模拟数据用于回测验证
            logger.info("使用长时间模拟数据进行回测验证...")
            # 生成5年的数据
            long_training_period = 1825  # 5年
            dates = pd.date_range(end=datetime.now(), periods=long_training_period)
            np.random.seed(42)
            
            # 生成更真实的比特币价格数据，包含周期性和趋势
            # 基础趋势（更长期的趋势）
            trend = np.linspace(0, 20000, long_training_period)
            # 季节性波动（更多周期）
            seasonality = 5000 * np.sin(np.linspace(0, 20 * np.pi, long_training_period))
            # 随机波动
            noise = np.cumsum(np.random.randn(long_training_period) * 500)
            # 综合
            prices = 30000 + trend + seasonality + noise
            
            # 生成交易量数据，与价格波动相关
            volumes = 1e10 + np.abs(prices * 1000) * np.random.rand(long_training_period)
            # 生成市值数据
            market_caps = prices * 21000000  # 假设流通量为2100万
            
            bitcoin_df = pd.DataFrame({
                'date': dates.strftime('%Y-%m-%d'),
                'price': prices,
                'volume': volumes,
                'market_cap': market_caps
            })
            bitcoin_df['price_change'] = bitcoin_df['price'].pct_change() * 100
            bitcoin_df['price_change'] = bitcoin_df['price_change'].fillna(0)
            bitcoin_df['ma7'] = bitcoin_df['price'].rolling(window=7).mean()
            bitcoin_df['ma30'] = bitcoin_df['price'].rolling(window=30).mean()
            
            # 生成模拟占星数据
            astro_df = pd.DataFrame({
                'date': dates.strftime('%Y-%m-%d'),
                'sun_ra': np.random.randn(long_training_period) * 10 + 180,
                'sun_dec': np.random.randn(long_training_period) * 23 + 0,
                'moon_ra': np.random.randn(long_training_period) * 10 + 90,
                'moon_dec': np.random.randn(long_training_period) * 28 + 0,
                'mercury_ra': np.random.randn(long_training_period) * 10 + 180,
                'mercury_dec': np.random.randn(long_training_period) * 7 + 0,
                'venus_ra': np.random.randn(long_training_period) * 10 + 180,
                'venus_dec': np.random.randn(long_training_period) * 3 + 0,
                'mars_ra': np.random.randn(long_training_period) * 10 + 180,
                'mars_dec': np.random.randn(long_training_period) * 2 + 0,
                'jupiter_ra': np.random.randn(long_training_period) * 10 + 180,
                'jupiter_dec': np.random.randn(long_training_period) * 1 + 0,
                'saturn_ra': np.random.randn(long_training_period) * 10 + 180,
                'saturn_dec': np.random.randn(long_training_period) * 1 + 0,
                'uranus_ra': np.random.randn(long_training_period) * 10 + 180,
                'uranus_dec': np.random.randn(long_training_period) * 1 + 0,
                'neptune_ra': np.random.randn(long_training_period) * 10 + 180,
                'neptune_dec': np.random.randn(long_training_period) * 1 + 0,
                'pluto_ra': np.random.randn(long_training_period) * 10 + 180,
                'pluto_dec': np.random.randn(long_training_period) * 1 + 0,
                'moon_phase': np.random.randn(long_training_period) * 0.5 + 0.5,
                'lunar_eclipse': np.random.randint(0, 2, long_training_period),
                'solar_eclipse': np.random.randint(0, 2, long_training_period),
                'retrograde_mercury': np.random.randint(0, 2, long_training_period),
                'retrograde_venus': np.random.randint(0, 2, long_training_period),
                'retrograde_mars': np.random.randint(0, 2, long_training_period),
                'retrograde_jupiter': np.random.randint(0, 2, long_training_period),
                'retrograde_saturn': np.random.randint(0, 2, long_training_period),
                'retrograde_uranus': np.random.randint(0, 2, long_training_period),
                'retrograde_neptune': np.random.randint(0, 2, long_training_period),
                'retrograde_pluto': np.random.randint(0, 2, long_training_period)
            })
            
            # 2. 数据处理
            logger.info("开始数据处理...")
            
            # 合并数据
            merged_df = self.data_processor.merge_data(astro_df, bitcoin_df)
            if merged_df.empty:
                logger.error("合并数据失败")
                return
            
            # 提取特征
            features_df = self.data_processor.extract_features(merged_df)
            if features_df.empty:
                logger.error("提取特征失败")
                return
            
            # 预处理数据
            X, y = self.data_processor.preprocess_data(features_df)
            if X is None or y is None:
                logger.error("预处理数据失败")
                return
            
            # 分割数据
            X_train, X_test, y_train, y_test = self.data_processor.split_data(X, y)
            if X_train is None:
                logger.error("分割数据失败")
                return
            
            # 3. 模型训练
            logger.info("开始模型训练...")
            model = self.model_trainer.train(X_train, y_train)
            if model is None:
                logger.error("训练模型失败")
                return
            
            # 4. 模型评估
            logger.info("开始模型评估...")
            y_pred = self.model_trainer.predict(X_test)
            if y_pred is None:
                logger.error("预测失败")
                return
            
            metrics = self.model_trainer.evaluate(y_test, y_pred)
            if not metrics:
                logger.error("评估模型失败")
                return
            
            # 5. 动态调整策略参数
            logger.info("开始调整策略参数...")
            # 构建市场特征字典
            latest_features = features_df.iloc[-1]
            market_features = {
                'market_sentiment': latest_features['market_sentiment'],
                'volatility_7d': latest_features['volatility_7d'],
                'rsi': latest_features['rsi'],
                'macd': latest_features['macd'],
                'bb_position': latest_features['bb_position'],
                'stoch_k': latest_features['stoch_k'],
                'adx': latest_features['adx'],
                'atr': latest_features.get('atr', 0),
                'obv': latest_features.get('obv', 0),
                'roc': latest_features.get('roc', 0),
                'cci': latest_features.get('cci', 0),
                'momentum_14': latest_features.get('momentum_14', 0),
                'volatility_change': latest_features.get('volatility_change', 0),
                'volume_ratio': latest_features.get('volume_ratio', 1),
                'fear_greed_index': latest_features.get('fear_greed_index', 50),
                'hash_rate_change': latest_features.get('hash_rate_change', 0),
                'price_change': latest_features['price_change']
            }
            # 调整参数
            self.strategy.adjust_parameters(market_features)
            
            # 6. 生成交易信号
            logger.info("开始生成交易信号...")
            signals = self.strategy.generate_signals(y_pred)
            if not signals:
                logger.error("生成交易信号失败")
                return
            
            # 6. 回测策略
            logger.info("开始回测策略...")
            # 提取对应时期的市场特征数据
            test_start_idx = len(features_df) - len(y_test)
            market_features_data = features_df.iloc[test_start_idx:]
            
            backtest_results = self.strategy.backtest(
                signals, 
                features_df['price'].iloc[-len(y_test):],
                market_features_data=market_features_data
            )
            if not backtest_results:
                logger.error("回测策略失败")
                return
            
            # 7. 可视化结果
            logger.info("开始可视化结果...")
            
            # 绘制价格趋势图
            self.visualizer.plot_price_trend(bitcoin_df, save=True)
            
            # 绘制预测结果图
            test_df = features_df.iloc[-len(y_test):].copy()
            self.visualizer.plot_prediction(test_df, y_pred, save=True)
            
            # 绘制回测结果图
            self.visualizer.plot_backtest_results(backtest_results, save=True)
            
            # 绘制策略参数变化图
            self.visualizer.plot_parameter_changes(backtest_results, save=True)
            
            # 绘制策略性能分析图
            self.visualizer.plot_strategy_performance(backtest_results, save=True)
            
            # 绘制市场情绪与参数关系图
            self.visualizer.plot_market_sentiment_vs_parameters(features_df, backtest_results, save=True)
            
            # 绘制特征重要性图
            if hasattr(model, 'feature_importances_') or hasattr(model, 'coef_'):
                feature_columns = [col for col in features_df.columns if col not in ['date', 'timestamp', 'price_change', 'price', 'market_cap', 'volume', 'ma7', 'ma30']]
                self.visualizer.plot_feature_importance(model, feature_columns, save=True)
            
            # 8. 保存模型
            logger.info("开始保存模型...")
            model_path = os.path.join(config.DATA_DIR, 'models', f'{config.MODEL_TYPE}_model.pkl')
            self.model_trainer.save_model(model_path)
            
            logger.info("模型运行完成")
        except Exception as e:
            logger.error(f"模型运行失败: {e}")

if __name__ == "__main__":
    # 初始化并运行模型
    model = AstroBitcoinModel()
    model.run()
