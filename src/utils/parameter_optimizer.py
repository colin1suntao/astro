import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from loguru import logger
import itertools
from src.config import config
from src.data.astro_data import AstroData
from src.data.bitcoin_data import BitcoinData
from src.data.data_processor import DataProcessor
from src.strategy.strategy import TradingStrategy

class ParameterOptimizer:
    """参数优化类"""
    
    def __init__(self):
        """初始化"""
        self.astro_data = AstroData()
        self.bitcoin_data = BitcoinData()
        self.data_processor = DataProcessor()
        self.strategy = TradingStrategy()
    
    def get_data(self):
        """
        获取数据
        
        Returns:
            X, y: 特征和目标数据
        """
        try:
            # 生成模拟比特币数据
            from datetime import datetime, timedelta
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
            
            # 合并数据
            merged_df = self.data_processor.merge_data(astro_df, bitcoin_df)
            
            # 提取特征
            features_df = self.data_processor.extract_features(merged_df)
            
            # 预处理数据
            X, y = self.data_processor.preprocess_data(features_df)
            
            return X, y, features_df['price']
        except Exception as e:
            logger.error(f"获取数据失败: {e}")
            return None, None, None
    
    def optimize_model_params(self, X, y):
        """
        优化模型参数
        
        Args:
            X: 特征数据
            y: 目标数据
            
        Returns:
            dict: 最佳模型参数
        """
        try:
            logger.info("开始优化模型参数...")
            
            # 定义参数网格
            param_grid = {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.05, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            }
            
            # 使用时间序列交叉验证
            tscv = TimeSeriesSplit(n_splits=5)
            
            # 初始化模型
            model = XGBRegressor(random_state=42)
            
            # 网格搜索
            grid_search = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                cv=tscv,
                scoring='neg_mean_squared_error',
                n_jobs=-1,
                verbose=1
            )
            
            # 拟合数据
            grid_search.fit(X, y)
            
            logger.info(f"最佳模型参数: {grid_search.best_params_}")
            logger.info(f"最佳模型分数: {abs(grid_search.best_score_):.4f}")
            
            return grid_search.best_params_
        except Exception as e:
            logger.error(f"优化模型参数失败: {e}")
            return None
    
    def optimize_strategy_params(self, X, y, price_data):
        """
        优化策略参数
        
        Args:
            X: 特征数据
            y: 目标数据
            price_data: 价格数据
            
        Returns:
            dict: 最佳策略参数
        """
        try:
            logger.info("开始优化策略参数...")
            
            # 定义参数网格
            thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
            max_position_sizes = [0.05, 0.1, 0.15, 0.2]
            
            best_return = -float('inf')
            best_params = {}
            
            # 训练基础模型
            model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            model.fit(X_train, y_train)
            
            # 预测
            y_pred = model.predict(X_test)
            
            # 遍历参数组合
            for threshold in thresholds:
                for max_position in max_position_sizes:
                    # 生成信号
                    signals = []
                    for pred in y_pred:
                        if pred > threshold:
                            signals.append(1)
                        elif pred < -threshold:
                            signals.append(-1)
                        else:
                            signals.append(0)
                    
                    # 回测
                    temp_strategy = TradingStrategy()
                    temp_strategy.max_position_size = max_position
                    
                    # 获取对应时期的价格数据
                    test_price_data = price_data.iloc[split_idx:]
                    
                    backtest_results = temp_strategy.backtest(signals, test_price_data)
                    
                    if backtest_results and backtest_results['total_return'] > best_return:
                        best_return = backtest_results['total_return']
                        best_params = {
                            'threshold': threshold,
                            'max_position_size': max_position
                        }
                        logger.info(f"找到更好的参数组合: {best_params}, 收益率: {best_return:.2f}%")
            
            logger.info(f"最佳策略参数: {best_params}")
            logger.info(f"最佳收益率: {best_return:.2f}%")
            
            return best_params
        except Exception as e:
            logger.error(f"优化策略参数失败: {e}")
            return None
    
    def run_optimization(self):
        """
        运行完整的参数优化
        
        Returns:
            dict: 最佳参数组合
        """
        try:
            # 获取数据
            X, y, price_data = self.get_data()
            if X is None or y is None:
                logger.error("无法获取数据，优化失败")
                return None
            
            # 优化模型参数
            best_model_params = self.optimize_model_params(X, y)
            
            # 优化策略参数
            best_strategy_params = self.optimize_strategy_params(X, y, price_data)
            
            # 综合最佳参数
            best_params = {
                'model_params': best_model_params,
                'strategy_params': best_strategy_params
            }
            
            logger.info("参数优化完成")
            logger.info(f"最佳参数组合: {best_params}")
            
            return best_params
        except Exception as e:
            logger.error(f"运行优化失败: {e}")
            return None

if __name__ == "__main__":
    # 运行参数优化
    optimizer = ParameterOptimizer()
    best_params = optimizer.run_optimization()
    
    if best_params:
        print("\n最佳参数组合:")
        print(f"模型参数: {best_params['model_params']}")
        print(f"策略参数: {best_params['strategy_params']}")
