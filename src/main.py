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
            
            # 获取比特币历史数据
            bitcoin_df = self.bitcoin_data.get_historical_data(days=config.TRAINING_PERIOD)
            if bitcoin_df.empty:
                logger.warning("获取比特币数据失败，使用模拟数据")
                # 生成模拟比特币数据
                dates = pd.date_range(end=datetime.now(), periods=config.TRAINING_PERIOD)
                np.random.seed(42)
                prices = 40000 + np.cumsum(np.random.randn(config.TRAINING_PERIOD) * 1000)
                volumes = 1e10 + np.random.randn(config.TRAINING_PERIOD) * 1e9
                market_caps = 8e11 + np.cumsum(np.random.randn(config.TRAINING_PERIOD) * 1e10)
                
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
            
            # 获取占星数据
            start_date = (datetime.now() - timedelta(days=config.TRAINING_PERIOD)).strftime('%Y-%m-%d')
            end_date = datetime.now().strftime('%Y-%m-%d')
            astro_df = self.astro_data.get_astro_data(start_date, end_date)
            if astro_df.empty:
                logger.warning("获取占星数据失败，使用模拟数据")
                # 生成模拟占星数据
                astro_df = pd.DataFrame({
                    'date': dates.strftime('%Y-%m-%d'),
                    'sun_ra': np.random.randn(config.TRAINING_PERIOD) * 10 + 180,
                    'sun_dec': np.random.randn(config.TRAINING_PERIOD) * 23 + 0,
                    'moon_ra': np.random.randn(config.TRAINING_PERIOD) * 10 + 90,
                    'moon_dec': np.random.randn(config.TRAINING_PERIOD) * 28 + 0,
                    'mercury_ra': np.random.randn(config.TRAINING_PERIOD) * 10 + 180,
                    'mercury_dec': np.random.randn(config.TRAINING_PERIOD) * 7 + 0,
                    'venus_ra': np.random.randn(config.TRAINING_PERIOD) * 10 + 180,
                    'venus_dec': np.random.randn(config.TRAINING_PERIOD) * 3 + 0,
                    'mars_ra': np.random.randn(config.TRAINING_PERIOD) * 10 + 180,
                    'mars_dec': np.random.randn(config.TRAINING_PERIOD) * 2 + 0,
                    'jupiter_ra': np.random.randn(config.TRAINING_PERIOD) * 10 + 180,
                    'jupiter_dec': np.random.randn(config.TRAINING_PERIOD) * 1 + 0,
                    'saturn_ra': np.random.randn(config.TRAINING_PERIOD) * 10 + 180,
                    'saturn_dec': np.random.randn(config.TRAINING_PERIOD) * 1 + 0,
                    'uranus_ra': np.random.randn(config.TRAINING_PERIOD) * 10 + 180,
                    'uranus_dec': np.random.randn(config.TRAINING_PERIOD) * 1 + 0,
                    'neptune_ra': np.random.randn(config.TRAINING_PERIOD) * 10 + 180,
                    'neptune_dec': np.random.randn(config.TRAINING_PERIOD) * 1 + 0,
                    'pluto_ra': np.random.randn(config.TRAINING_PERIOD) * 10 + 180,
                    'pluto_dec': np.random.randn(config.TRAINING_PERIOD) * 1 + 0,
                    'moon_phase': np.random.randn(config.TRAINING_PERIOD) * 0.5 + 0.5,
                    'lunar_eclipse': np.random.randint(0, 2, config.TRAINING_PERIOD),
                    'solar_eclipse': np.random.randint(0, 2, config.TRAINING_PERIOD),
                    'retrograde_mercury': np.random.randint(0, 2, config.TRAINING_PERIOD),
                    'retrograde_venus': np.random.randint(0, 2, config.TRAINING_PERIOD),
                    'retrograde_mars': np.random.randint(0, 2, config.TRAINING_PERIOD),
                    'retrograde_jupiter': np.random.randint(0, 2, config.TRAINING_PERIOD),
                    'retrograde_saturn': np.random.randint(0, 2, config.TRAINING_PERIOD),
                    'retrograde_uranus': np.random.randint(0, 2, config.TRAINING_PERIOD),
                    'retrograde_neptune': np.random.randint(0, 2, config.TRAINING_PERIOD),
                    'retrograde_pluto': np.random.randint(0, 2, config.TRAINING_PERIOD)
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
            
            # 5. 生成交易信号
            logger.info("开始生成交易信号...")
            signals = self.strategy.generate_signals(y_pred)
            if not signals:
                logger.error("生成交易信号失败")
                return
            
            # 6. 回测策略
            logger.info("开始回测策略...")
            backtest_results = self.strategy.backtest(signals, features_df['price'].iloc[-len(y_test):])
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
