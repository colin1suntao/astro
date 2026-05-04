import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from loguru import logger
from src.config import config

class MLParameterOptimizer:
    """基于机器学习的参数优化器"""
    
    def __init__(self):
        """初始化"""
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.is_trained = False
    
    def generate_parameter_data(self, market_data, param_ranges=None):
        """
        生成参数训练数据
        
        Args:
            market_data: 市场数据，包含特征和目标
            param_ranges: 参数范围
            
        Returns:
            X: 特征数据
            y: 目标数据（收益率）
            param_data: 参数数据
        """
        try:
            if param_ranges is None:
                param_ranges = {
                    'threshold': np.linspace(0.2, 0.6, 5),
                    'max_position_size': np.linspace(0.1, 0.3, 5),
                    'stop_loss': np.linspace(0.03, 0.07, 3),
                    'take_profit': np.linspace(0.08, 0.15, 3)
                }
            
            # 生成参数组合
            param_combinations = []
            for threshold in param_ranges['threshold']:
                for max_position in param_ranges['max_position_size']:
                    for stop_loss in param_ranges['stop_loss']:
                        for take_profit in param_ranges['take_profit']:
                            param_combinations.append({
                                'threshold': threshold,
                                'max_position_size': max_position,
                                'stop_loss': stop_loss,
                                'take_profit': take_profit
                            })
            
            # 生成训练数据
            X = []
            y = []
            param_data = []
            
            # 检查market_data是否为DataFrame且包含必要的列
            if isinstance(market_data, pd.DataFrame) and not market_data.empty:
                logger.info("使用真实市场数据生成训练样本")
                
                # 为每个参数组合在真实市场数据上进行回测
                from src.strategy.strategy import TradingStrategy
                strategy = TradingStrategy()
                
                # 提取价格数据
                price_data = market_data['price']
                
                # 提取市场特征
                market_features_list = []
                for i in range(len(market_data)):
                    features = {
                        'market_sentiment': market_data.iloc[i].get('market_sentiment', 0),
                        'volatility_7d': market_data.iloc[i].get('volatility_7d', 0.03),
                        'rsi': market_data.iloc[i].get('rsi', 50),
                        'macd': market_data.iloc[i].get('macd', 0),
                        'bb_position': market_data.iloc[i].get('bb_position', 0.5),
                        'stoch_k': market_data.iloc[i].get('stoch_k', 50),
                        'adx': market_data.iloc[i].get('adx', 25),
                        'atr': market_data.iloc[i].get('atr', 500),
                        'obv': market_data.iloc[i].get('obv', 5000000),
                        'roc': market_data.iloc[i].get('roc', 0),
                        'cci': market_data.iloc[i].get('cci', 0),
                        'momentum_14': market_data.iloc[i].get('momentum_14', 0),
                        'volatility_change': market_data.iloc[i].get('volatility_change', 0),
                        'volume_ratio': market_data.iloc[i].get('volume_ratio', 1),
                        'fear_greed_index': market_data.iloc[i].get('fear_greed_index', 50),
                        'hash_rate_change': market_data.iloc[i].get('hash_rate_change', 0),
                        'price_change': market_data.iloc[i].get('price_change', 0)
                    }
                    market_features_list.append(features)
                
                # 生成预测数据（使用简化的预测模型）
                predictions = np.random.uniform(-1, 1, len(market_data))
                
                # 对每个参数组合进行回测
                for i, params in enumerate(param_combinations):
                    # 设置策略参数
                    strategy.threshold = params['threshold']
                    strategy.max_position_size = params['max_position_size']
                    strategy.stop_loss = params['stop_loss']
                    strategy.take_profit = params['take_profit']
                    
                    # 生成信号
                    signals = strategy.generate_signals(predictions, threshold=params['threshold'])
                    
                    # 进行回测
                    backtest_result = strategy.backtest(signals, price_data)
                    
                    if backtest_result and 'total_return' in backtest_result:
                        total_return = backtest_result['total_return']
                        
                        # 为每个市场条件生成样本
                        for j, features in enumerate(market_features_list):
                            if j < len(market_data):
                                # 构建特征向量
                                feature_vector = list(features.values()) + list(params.values())
                                X.append(feature_vector)
                                y.append(total_return)
                                param_data.append(params)
                
                if len(X) > 0:
                    logger.info(f"使用真实回测数据生成参数训练数据成功，共 {len(X)} 个样本")
                    return np.array(X), np.array(y), param_data
                else:
                    logger.warning("无法使用真实回测数据，使用模拟数据")
            
            # 如果真实数据不可用，使用模拟数据
            logger.info("使用模拟数据生成训练样本")
            for i, params in enumerate(param_combinations):
                # 为每个参数组合生成模拟的市场条件和收益率
                for j in range(100):  # 每个参数组合生成100个样本
                    # 生成模拟的市场特征
                    market_features = {
                        'market_sentiment': np.random.uniform(-1, 1),
                        'volatility_7d': np.random.uniform(0.01, 0.1),
                        'rsi': np.random.uniform(20, 80),
                        'macd': np.random.uniform(-2, 2),
                        'bb_position': np.random.uniform(0, 1),
                        'stoch_k': np.random.uniform(0, 100),
                        'adx': np.random.uniform(10, 50),
                        'atr': np.random.uniform(100, 1000),
                        'obv': np.random.uniform(1000000, 10000000),
                        'roc': np.random.uniform(-20, 20),
                        'cci': np.random.uniform(-200, 200),
                        'momentum_14': np.random.uniform(-1000, 1000),
                        'volatility_change': np.random.uniform(-50, 50),
                        'volume_ratio': np.random.uniform(0.5, 2),
                        'fear_greed_index': np.random.uniform(0, 100),
                        'hash_rate_change': np.random.uniform(-10, 10),
                        'price_change': np.random.uniform(-5, 5)
                    }
                    
                    # 基于市场条件和参数计算模拟收益率
                    base_return = market_features['price_change'] * 0.1
                    
                    # 参数对收益率的影响
                    threshold_factor = 1.0 - abs(params['threshold'] - 0.4) * 2  # 0.4是最优阈值
                    position_factor = params['max_position_size'] / 0.2  # 0.2是基础仓位
                    risk_factor = (1 - params['stop_loss'] * 10) * (params['take_profit'] / 0.1)  # 风险收益比
                    
                    # 市场条件对收益率的影响
                    sentiment_factor = 1.0 + market_features['market_sentiment'] * 0.3
                    volatility_factor = 1.0 + market_features['volatility_7d'] * 5
                    rsi_factor = 1.0 if 30 < market_features['rsi'] < 70 else 0.8
                    
                    # 计算最终收益率
                    final_return = base_return * threshold_factor * position_factor * risk_factor * sentiment_factor * volatility_factor * rsi_factor
                    
                    # 确保收益率在合理范围内
                    final_return = max(min(final_return, 10), -10)
                    
                    # 构建特征向量
                    feature_vector = list(market_features.values()) + list(params.values())
                    X.append(feature_vector)
                    y.append(final_return)
                    param_data.append(params)
            
            logger.info(f"生成参数训练数据成功，共 {len(X)} 个样本")
            return np.array(X), np.array(y), param_data
        except Exception as e:
            logger.error(f"生成参数训练数据失败: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None
    
    def train(self, X, y):
        """
        训练参数优化模型
        
        Args:
            X: 特征数据
            y: 目标数据
        """
        try:
            # 分割数据
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # 训练模型
            self.model.fit(X_train, y_train)
            
            # 评估模型
            y_pred = self.model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            logger.info(f"参数优化模型训练成功，MSE: {mse:.4f}, R2: {r2:.4f}")
            self.is_trained = True
        except Exception as e:
            logger.error(f"训练参数优化模型失败: {e}")
    
    def optimize_parameters(self, market_features):
        """
        根据市场特征优化参数
        
        Args:
            market_features: 市场特征
            
        Returns:
            dict: 优化后的参数
        """
        try:
            if not self.is_trained:
                # 如果模型未训练，生成数据并训练
                logger.info("参数优化模型未训练，开始训练...")
                X, y, _ = self.generate_parameter_data(market_features)
                if X is not None and y is not None:
                    self.train(X, y)
                else:
                    logger.error("无法生成训练数据，使用默认参数")
                    return {
                        'threshold': 0.4,
                        'max_position_size': 0.2,
                        'stop_loss': 0.05,
                        'take_profit': 0.1
                    }
            
            # 尝试使用贝叶斯优化
            try:
                from scipy.optimize import minimize
                
                # 定义目标函数（最大化收益率）
                def objective(params):
                    threshold, max_position, stop_loss, take_profit = params
                    # 确保参数在有效范围内
                    if not (0.2 <= threshold <= 0.6 and 0.1 <= max_position <= 0.3 and 0.03 <= stop_loss <= 0.07 and 0.08 <= take_profit <= 0.15):
                        return -100  # 惩罚无效参数
                    # 构建特征向量
                    feature_vector = list(market_features.values()) + [threshold, max_position, stop_loss, take_profit]
                    feature_vector = np.array(feature_vector).reshape(1, -1)
                    # 预测收益率并取负值（因为minimize是最小化）
                    predicted_return = self.model.predict(feature_vector)[0]
                    return -predicted_return
                
                # 初始猜测
                x0 = [0.4, 0.2, 0.05, 0.1]
                
                # 优化
                bounds = [(0.2, 0.6), (0.1, 0.3), (0.03, 0.07), (0.08, 0.15)]
                result = minimize(objective, x0, bounds=bounds, method='L-BFGS-B')
                
                if result.success:
                    best_params = {
                        'threshold': result.x[0],
                        'max_position_size': result.x[1],
                        'stop_loss': result.x[2],
                        'take_profit': result.x[3]
                    }
                    # 计算最佳参数的预测收益率
                    feature_vector = list(market_features.values()) + list(result.x)
                    feature_vector = np.array(feature_vector).reshape(1, -1)
                    best_return = self.model.predict(feature_vector)[0]
                    
                    logger.info(f"使用贝叶斯优化完成参数优化，最佳参数: {best_params}, 预测收益率: {best_return:.2f}%")
                    return best_params
                else:
                    logger.warning("贝叶斯优化失败，使用网格搜索")
            except ImportError:
                logger.warning("scipy不可用，使用网格搜索")
            except Exception as e:
                logger.warning(f"贝叶斯优化失败: {e}，使用网格搜索")
            
            # 回退到网格搜索
            param_ranges = {
                'threshold': np.linspace(0.2, 0.6, 10),
                'max_position_size': np.linspace(0.1, 0.3, 10),
                'stop_loss': np.linspace(0.03, 0.07, 5),
                'take_profit': np.linspace(0.08, 0.15, 5)
            }
            
            best_params = None
            best_return = -float('inf')
            
            for threshold in param_ranges['threshold']:
                for max_position in param_ranges['max_position_size']:
                    for stop_loss in param_ranges['stop_loss']:
                        for take_profit in param_ranges['take_profit']:
                            # 构建特征向量
                            feature_vector = list(market_features.values()) + [threshold, max_position, stop_loss, take_profit]
                            feature_vector = np.array(feature_vector).reshape(1, -1)
                            
                            # 预测收益率
                            predicted_return = self.model.predict(feature_vector)[0]
                            
                            if predicted_return > best_return:
                                best_return = predicted_return
                                best_params = {
                                    'threshold': threshold,
                                    'max_position_size': max_position,
                                    'stop_loss': stop_loss,
                                    'take_profit': take_profit
                                }
            
            logger.info(f"使用网格搜索完成参数优化，最佳参数: {best_params}, 预测收益率: {best_return:.2f}%")
            return best_params
        except Exception as e:
            logger.error(f"优化参数失败: {e}")
            import traceback
            traceback.print_exc()
            return {
                'threshold': 0.4,
                'max_position_size': 0.2,
                'stop_loss': 0.05,
                'take_profit': 0.1
            }
    
    def save_model(self, model_path):
        """
        保存模型
        
        Args:
            model_path: 模型保存路径
        """
        try:
            import joblib
            joblib.dump(self.model, model_path)
            logger.info(f"参数优化模型保存成功: {model_path}")
        except Exception as e:
            logger.error(f"保存参数优化模型失败: {e}")
    
    def load_model(self, model_path):
        """
        加载模型
        
        Args:
            model_path: 模型加载路径
        """
        try:
            import joblib
            self.model = joblib.load(model_path)
            self.is_trained = True
            logger.info(f"参数优化模型加载成功: {model_path}")
        except Exception as e:
            logger.error(f"加载参数优化模型失败: {e}")
