import pandas as pd
import numpy as np
from loguru import logger
import ccxt
from src.config import config

class TradingStrategy:
    """交易策略类"""
    
    def __init__(self):
        """初始化"""
        self.max_position_size = config.MAX_POSITION_SIZE
        self.stop_loss = config.STOP_LOSS
        self.take_profit = config.TAKE_PROFIT
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
    
    def generate_signals(self, predictions, threshold=0.5):
        """
        生成交易信号
        
        Args:
            predictions: 预测结果
            threshold: 信号阈值
            
        Returns:
            list: 交易信号列表，1表示买入，-1表示卖出，0表示持有
        """
        try:
            signals = []
            for pred in predictions:
                if pred > threshold:
                    signals.append(1)  # 买入
                elif pred < -threshold:
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
    
    def backtest(self, signals, price_data, initial_balance=10000):
        """
        回测策略

        Args:
            signals: 交易信号
            price_data: 价格数据
            initial_balance: 初始余额

        Returns:
            dict: 回测结果
        """
        try:
            balance = initial_balance
            position = 0
            trade_history = []

            for i, signal in enumerate(signals):
                current_price = price_data.iloc[i] if hasattr(price_data, 'iloc') else price_data[i]

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
                        'balance': balance
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
                        'balance': balance
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
