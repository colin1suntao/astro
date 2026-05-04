import pandas as pd
import numpy as np
from loguru import logger
import ccxt
from src.config import config
from src.utils.ml_parameter_optimizer import MLParameterOptimizer

class TradingStrategy:
    """交易策略类"""
    
    def __init__(self):
        """初始化"""
        self.base_max_position_size = config.MAX_POSITION_SIZE
        self.base_stop_loss = config.STOP_LOSS
        self.base_take_profit = config.TAKE_PROFIT
        self.base_threshold = 0.4  # 优化后的基础阈值
        
        # 动态参数
        self.max_position_size = self.base_max_position_size
        self.stop_loss = self.base_stop_loss
        self.take_profit = self.base_take_profit
        self.threshold = self.base_threshold
        
        # 初始化机器学习参数优化器
        self.ml_optimizer = MLParameterOptimizer()
        
        self.trading_mode = config.TRADING_MODE
        self.exchange = None
        
        # 初始化交易所连接
        if self.trading_mode == 'live' and config.TRADING_API_KEY and config.TRADING_SECRET:
            try:
                self.exchange = ccxt.binance({
                    'apiKey': config.TRADING_API_KEY,
                    'secret': config.TRADING_SECRET,
                    'enableRateLimit': True,
                })
                logger.info("交易所连接成功")
            except Exception as e:
                logger.error(f"交易所连接失败: {e}")
                self.exchange = None
    
    def adjust_parameters(self, market_features):
        """
        根据市场情况调整策略参数
        
        Args:
            market_features: 市场特征字典，包含情绪、波动率、技术指标等
        """
        try:
            # 使用机器学习优化器优化参数
            optimized_params = self.ml_optimizer.optimize_parameters(market_features)
            
            # 更新参数
            self.max_position_size = optimized_params['max_position_size']
            self.threshold = optimized_params['threshold']
            self.stop_loss = optimized_params['stop_loss']
            self.take_profit = optimized_params['take_profit']
            
            logger.info(f"基于机器学习的动态调整参数: 最大仓位={self.max_position_size:.2f}, 阈值={self.threshold:.2f}, 止盈={self.take_profit:.2f}, 止损={self.stop_loss:.2f}")
        except Exception as e:
            logger.error(f"调整参数失败: {e}")
            # 失败时使用基于规则的调整
            try:
                market_sentiment = market_features.get('market_sentiment', 0)
                volatility = market_features.get('volatility_7d', 0.03)
                
                # 根据市场情绪调整参数
                if market_sentiment > 0.5:  # 乐观情绪
                    # 增加仓位，降低阈值以捕捉更多机会
                    self.max_position_size = min(self.base_max_position_size * 1.2, 0.3)  # 最大不超过30%
                    self.threshold = max(self.base_threshold * 0.8, 0.2)  # 最低阈值0.2
                    self.take_profit = min(self.base_take_profit * 1.1, 0.15)  # 提高止盈
                    self.stop_loss = max(self.base_stop_loss * 0.9, 0.03)  # 降低止损
                elif market_sentiment < -0.5:  # 悲观情绪
                    # 减少仓位，提高阈值以过滤噪音
                    self.max_position_size = max(self.base_max_position_size * 0.8, 0.1)  # 最小不低于10%
                    self.threshold = min(self.base_threshold * 1.2, 0.6)  # 最高阈值0.6
                    self.take_profit = max(self.base_take_profit * 0.9, 0.08)  # 降低止盈
                    self.stop_loss = min(self.base_stop_loss * 1.1, 0.07)  # 提高止损
                else:  # 中性情绪
                    # 恢复基础参数
                    self.max_position_size = self.base_max_position_size
                    self.threshold = self.base_threshold
                    self.take_profit = self.base_take_profit
                    self.stop_loss = self.base_stop_loss
                
                # 根据波动率调整参数
                if volatility > 0.05:  # 高波动率
                    # 减少仓位，提高止损
                    self.max_position_size = max(self.max_position_size * 0.9, 0.1)
                    self.stop_loss = min(self.stop_loss * 1.1, 0.07)
                elif volatility < 0.02:  # 低波动率
                    # 增加仓位，降低止损
                    self.max_position_size = min(self.max_position_size * 1.1, 0.3)
                    self.stop_loss = max(self.stop_loss * 0.9, 0.03)
                
                logger.info(f"基于规则的动态调整参数: 最大仓位={self.max_position_size:.2f}, 阈值={self.threshold:.2f}, 止盈={self.take_profit:.2f}, 止损={self.stop_loss:.2f}")
            except Exception as e2:
                logger.error(f"基于规则的参数调整也失败: {e2}")
    
    def generate_signals(self, predictions, threshold=None):
        """
        生成交易信号
        
        Args:
            predictions: 预测结果
            threshold: 信号阈值（None时使用动态调整的阈值）
            
        Returns:
            list: 交易信号列表，1表示买入，-1表示卖出，0表示持有
        """
        try:
            # 使用动态调整的阈值或传入的阈值
            current_threshold = threshold if threshold is not None else self.threshold
            
            signals = []
            for pred in predictions:
                if pred > current_threshold:
                    signals.append(1)  # 买入
                elif pred < -current_threshold:
                    signals.append(-1)  # 卖出
                else:
                    signals.append(0)  # 持有
            
            logger.info("交易信号生成成功")
            return signals
        except Exception as e:
            logger.error(f"生成交易信号失败: {e}")
            return []
    
    def calculate_position_size(self, account_balance, current_price):
        """
        计算仓位大小
        
        Args:
            account_balance: 账户余额
            current_price: 当前价格
            
        Returns:
            float: 仓位大小（比特币数量）
        """
        try:
            # 计算最大可购买数量
            max_amount = (account_balance * self.max_position_size) / current_price
            return max_amount
        except Exception as e:
            logger.error(f"计算仓位大小失败: {e}")
            return 0
    
    def execute_trade(self, signal, current_price, account_balance):
        """
        执行交易
        
        Args:
            signal: 交易信号
            current_price: 当前价格
            account_balance: 账户余额
            
        Returns:
            dict: 交易结果
        """
        try:
            if self.trading_mode == 'paper':
                # 模拟交易
                position_size = self.calculate_position_size(account_balance, current_price)
                if signal == 1:
                    # 买入
                    cost = position_size * current_price
                    logger.info(f"模拟买入 {position_size:.6f} BTC，成本: ${cost:.2f}")
                    return {
                        'status': 'success',
                        'action': 'buy',
                        'amount': position_size,
                        'price': current_price,
                        'cost': cost
                    }
                elif signal == -1:
                    # 卖出
                    proceeds = position_size * current_price
                    logger.info(f"模拟卖出 {position_size:.6f} BTC，收益: ${proceeds:.2f}")
                    return {
                        'status': 'success',
                        'action': 'sell',
                        'amount': position_size,
                        'price': current_price,
                        'proceeds': proceeds
                    }
                else:
                    logger.info("持有，不执行交易")
                    return {
                        'status': 'success',
                        'action': 'hold',
                        'amount': 0,
                        'price': current_price
                    }
            else:
                # 实盘交易
                if self.exchange is None:
                    logger.error("交易所连接失败，无法执行实盘交易")
                    return {'status': 'error', 'message': '交易所连接失败'}
                
                position_size = self.calculate_position_size(account_balance, current_price)
                if signal == 1:
                    # 买入
                    order = self.exchange.create_market_buy_order('BTC/USDT', position_size)
                    logger.info(f"实盘买入 {position_size:.6f} BTC，订单ID: {order['id']}")
                    return {
                        'status': 'success',
                        'action': 'buy',
                        'amount': position_size,
                        'price': current_price,
                        'order_id': order['id']
                    }
                elif signal == -1:
                    # 卖出
                    order = self.exchange.create_market_sell_order('BTC/USDT', position_size)
                    logger.info(f"实盘卖出 {position_size:.6f} BTC，订单ID: {order['id']}")
                    return {
                        'status': 'success',
                        'action': 'sell',
                        'amount': position_size,
                        'price': current_price,
                        'order_id': order['id']
                    }
                else:
                    logger.info("持有，不执行交易")
                    return {
                        'status': 'success',
                        'action': 'hold',
                        'amount': 0,
                        'price': current_price
                    }
        except Exception as e:
            logger.error(f"执行交易失败: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def backtest(self, signals, price_data, initial_balance=10000, market_features_data=None):
        """
        回测策略

        Args:
            signals: 交易信号
            price_data: 价格数据
            initial_balance: 初始余额
            market_features_data: 市场特征数据

        Returns:
            dict: 回测结果
        """
        try:
            balance = initial_balance
            position = 0
            trade_history = []

            for i, signal in enumerate(signals):
                current_price = price_data.iloc[i] if hasattr(price_data, 'iloc') else price_data[i]
                
                # 动态调整参数（如果有市场数据）
                if market_features_data is not None:
                    if i < len(market_features_data):
                        # 构建市场特征字典
                        market_features = {
                            'market_sentiment': market_features_data.iloc[i]['market_sentiment'] if hasattr(market_features_data, 'iloc') else market_features_data[i]['market_sentiment'],
                            'volatility_7d': market_features_data.iloc[i]['volatility_7d'] if hasattr(market_features_data, 'iloc') else market_features_data[i]['volatility_7d'],
                            'rsi': market_features_data.iloc[i]['rsi'] if hasattr(market_features_data, 'iloc') else market_features_data[i]['rsi'],
                            'macd': market_features_data.iloc[i]['macd'] if hasattr(market_features_data, 'iloc') else market_features_data[i]['macd'],
                            'bb_position': market_features_data.iloc[i]['bb_position'] if hasattr(market_features_data, 'iloc') else market_features_data[i]['bb_position'],
                            'stoch_k': market_features_data.iloc[i]['stoch_k'] if hasattr(market_features_data, 'iloc') else market_features_data[i]['stoch_k'],
                            'adx': market_features_data.iloc[i]['adx'] if hasattr(market_features_data, 'iloc') else market_features_data[i]['adx'],
                            'atr': market_features_data.iloc[i].get('atr', 0) if hasattr(market_features_data, 'iloc') else market_features_data[i].get('atr', 0),
                            'obv': market_features_data.iloc[i].get('obv', 0) if hasattr(market_features_data, 'iloc') else market_features_data[i].get('obv', 0),
                            'roc': market_features_data.iloc[i].get('roc', 0) if hasattr(market_features_data, 'iloc') else market_features_data[i].get('roc', 0),
                            'cci': market_features_data.iloc[i].get('cci', 0) if hasattr(market_features_data, 'iloc') else market_features_data[i].get('cci', 0),
                            'momentum_14': market_features_data.iloc[i].get('momentum_14', 0) if hasattr(market_features_data, 'iloc') else market_features_data[i].get('momentum_14', 0),
                            'volatility_change': market_features_data.iloc[i].get('volatility_change', 0) if hasattr(market_features_data, 'iloc') else market_features_data[i].get('volatility_change', 0),
                            'volume_ratio': market_features_data.iloc[i].get('volume_ratio', 1) if hasattr(market_features_data, 'iloc') else market_features_data[i].get('volume_ratio', 1),
                            'fear_greed_index': market_features_data.iloc[i].get('fear_greed_index', 50) if hasattr(market_features_data, 'iloc') else market_features_data[i].get('fear_greed_index', 50),
                            'hash_rate_change': market_features_data.iloc[i].get('hash_rate_change', 0) if hasattr(market_features_data, 'iloc') else market_features_data[i].get('hash_rate_change', 0),
                            'price_change': market_features_data.iloc[i]['price_change'] if hasattr(market_features_data, 'iloc') else market_features_data[i]['price_change']
                        }
                        self.adjust_parameters(market_features)

                if signal == 1 and position == 0:
                    # 买入
                    position_size = (balance * self.max_position_size) / current_price
                    cost = position_size * current_price
                    balance -= cost
                    position = position_size
                    # 尝试获取日期，如果没有索引则使用序号
                    try:
                        trade_date = price_data.index[i] if hasattr(price_data, 'index') else f'Day {i}'
                    except:
                        trade_date = f'Day {i}'
                    trade_history.append({
                        'date': trade_date,
                        'action': 'buy',
                        'price': current_price,
                        'amount': position_size,
                        'balance': balance,
                        'max_position_size': self.max_position_size,
                        'threshold': self.threshold,
                        'stop_loss': self.stop_loss,
                        'take_profit': self.take_profit
                    })
                elif signal == -1 and position > 0:
                    # 卖出
                    proceeds = position * current_price
                    balance += proceeds
                    # 尝试获取日期，如果没有索引则使用序号
                    try:
                        trade_date = price_data.index[i] if hasattr(price_data, 'index') else f'Day {i}'
                    except:
                        trade_date = f'Day {i}'
                    trade_history.append({
                        'date': trade_date,
                        'action': 'sell',
                        'price': current_price,
                        'amount': position,
                        'balance': balance,
                        'max_position_size': self.max_position_size,
                        'threshold': self.threshold,
                        'stop_loss': self.stop_loss,
                        'take_profit': self.take_profit
                    })
                    position = 0

            # 计算最终余额
            final_price = price_data.iloc[-1] if hasattr(price_data, 'iloc') else price_data[-1]
            final_balance = balance + (position * final_price)
            total_return = (final_balance - initial_balance) / initial_balance * 100

            logger.info(f"回测完成，初始余额: ${initial_balance:.2f}，最终余额: ${final_balance:.2f}，总收益率: {total_return:.2f}%")

            return {
                'initial_balance': initial_balance,
                'final_balance': final_balance,
                'total_return': total_return,
                'trade_history': trade_history
            }
        except Exception as e:
            logger.error(f"回测策略失败: {e}")
            import traceback
            traceback.print_exc()
            return {}
