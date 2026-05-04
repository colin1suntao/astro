import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from loguru import logger
import os
from src.config import config

class Visualizer:
    """数据可视化类"""
    
    def __init__(self):
        """初始化"""
        # 设置绘图风格
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette('husl')
        
        # 创建输出目录
        self.output_dir = os.path.join(config.DATA_DIR, 'visualizations')
        os.makedirs(self.output_dir, exist_ok=True)
    
    def plot_price_trend(self, df, save=False):
        """
        绘制价格趋势图
        
        Args:
            df: 包含价格数据的DataFrame
            save: 是否保存图片
        """
        try:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # 绘制价格
            ax.plot(df['date'], df['price'], label='Price', linewidth=2)
            
            # 绘制移动平均线
            if 'ma7' in df.columns:
                ax.plot(df['date'], df['ma7'], label='7-day MA', linewidth=1.5, linestyle='--')
            if 'ma30' in df.columns:
                ax.plot(df['date'], df['ma30'], label='30-day MA', linewidth=1.5, linestyle='--')
            
            # 设置标题和标签
            ax.set_title('Bitcoin Price Trend', fontsize=16)
            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel('Price (USD)', fontsize=12)
            
            # 添加图例
            ax.legend()
            
            # 旋转日期标签
            plt.xticks(rotation=45)
            
            # 调整布局
            plt.tight_layout()
            
            if save:
                output_path = os.path.join(self.output_dir, 'price_trend.png')
                plt.savefig(output_path)
                logger.info(f"价格趋势图保存成功: {output_path}")
            else:
                plt.show()
        except Exception as e:
            logger.error(f"绘制价格趋势图失败: {e}")
    
    def plot_prediction(self, df, predictions, save=False):
        """
        绘制预测结果图
        
        Args:
            df: 包含实际价格数据的DataFrame
            predictions: 预测结果
            save: 是否保存图片
        """
        try:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # 绘制实际价格变化
            ax.plot(df['date'], df['price_change'], label='Actual', linewidth=2)
            
            # 绘制预测价格变化
            ax.plot(df['date'], predictions, label='Predicted', linewidth=2, linestyle='--')
            
            # 设置标题和标签
            ax.set_title('Bitcoin Price Change Prediction', fontsize=16)
            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel('Price Change (%)', fontsize=12)
            
            # 添加图例
            ax.legend()
            
            # 旋转日期标签
            plt.xticks(rotation=45)
            
            # 调整布局
            plt.tight_layout()
            
            if save:
                output_path = os.path.join(self.output_dir, 'prediction.png')
                plt.savefig(output_path)
                logger.info(f"预测结果图保存成功: {output_path}")
            else:
                plt.show()
        except Exception as e:
            logger.error(f"绘制预测结果图失败: {e}")
    
    def plot_backtest_results(self, backtest_results, save=False):
        """
        绘制回测结果图
        
        Args:
            backtest_results: 回测结果
            save: 是否保存图片
        """
        try:
            # 提取交易历史
            trade_history = backtest_results['trade_history']
            if not trade_history:
                logger.warning("回测结果中没有交易历史")
                return
            
            # 转换为DataFrame
            trade_df = pd.DataFrame(trade_history)
            
            # 计算累计收益
            initial_balance = backtest_results['initial_balance']
            trade_df['cumulative_balance'] = trade_df['balance']
            
            # 添加最终余额
            final_balance = backtest_results['final_balance']
            final_row = pd.DataFrame([{
                'date': trade_df['date'].iloc[-1],
                'action': 'final',
                'price': 0,
                'amount': 0,
                'balance': final_balance,
                'cumulative_balance': final_balance
            }])
            trade_df = pd.concat([trade_df, final_row], ignore_index=True)
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # 绘制累计余额
            ax.plot(trade_df['date'], trade_df['cumulative_balance'], label='Cumulative Balance', linewidth=2)
            
            # 标记买入和卖出点
            buy_points = trade_df[trade_df['action'] == 'buy']
            sell_points = trade_df[trade_df['action'] == 'sell']
            
            ax.scatter(buy_points['date'], buy_points['cumulative_balance'], color='green', marker='^', s=100, label='Buy')
            ax.scatter(sell_points['date'], sell_points['cumulative_balance'], color='red', marker='v', s=100, label='Sell')
            
            # 设置标题和标签
            ax.set_title('Backtest Results', fontsize=16)
            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel('Balance (USD)', fontsize=12)
            
            # 添加图例
            ax.legend()
            
            # 旋转日期标签
            plt.xticks(rotation=45)
            
            # 调整布局
            plt.tight_layout()
            
            if save:
                output_path = os.path.join(self.output_dir, 'backtest_results.png')
                plt.savefig(output_path)
                logger.info(f"回测结果图保存成功: {output_path}")
            else:
                plt.show()
        except Exception as e:
            logger.error(f"绘制回测结果图失败: {e}")
    
    def plot_feature_importance(self, model, feature_names, save=False):
        """
        绘制特征重要性图
        
        Args:
            model: 训练好的模型
            feature_names: 特征名称列表
            save: 是否保存图片
        """
        try:
            # 获取特征重要性
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importances = np.abs(model.coef_)
            else:
                logger.warning("模型不支持特征重要性分析")
                return
            
            # 排序特征重要性
            indices = np.argsort(importances)[::-1]
            sorted_importances = importances[indices]
            sorted_features = [feature_names[i] for i in indices]
            
            # 只显示前20个最重要的特征
            top_n = min(20, len(sorted_features))
            sorted_importances = sorted_importances[:top_n]
            sorted_features = sorted_features[:top_n]
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # 绘制特征重要性
            ax.barh(range(top_n), sorted_importances, align='center')
            ax.set_yticks(range(top_n))
            ax.set_yticklabels(sorted_features)
            ax.invert_yaxis()  # 最重要的特征在顶部
            
            # 设置标题和标签
            ax.set_title('Feature Importance', fontsize=16)
            ax.set_xlabel('Importance', fontsize=12)
            ax.set_ylabel('Feature', fontsize=12)
            
            # 调整布局
            plt.tight_layout()
            
            if save:
                output_path = os.path.join(self.output_dir, 'feature_importance.png')
                plt.savefig(output_path)
                logger.info(f"特征重要性图保存成功: {output_path}")
            else:
                plt.show()
        except Exception as e:
            logger.error(f"绘制特征重要性图失败: {e}")
    
    def plot_parameter_changes(self, backtest_results, save=False):
        """
        绘制策略参数变化图
        
        Args:
            backtest_results: 回测结果
            save: 是否保存图片
        """
        try:
            # 提取交易历史
            trade_history = backtest_results['trade_history']
            if not trade_history:
                logger.warning("回测结果中没有交易历史")
                return
            
            # 转换为DataFrame
            trade_df = pd.DataFrame(trade_history)
            
            # 检查是否包含参数列
            param_columns = ['max_position_size', 'threshold', 'stop_loss', 'take_profit']
            if not all(col in trade_df.columns for col in param_columns):
                logger.warning("回测结果中没有参数数据")
                return
            
            fig, axs = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('Strategy Parameter Changes', fontsize=18)
            
            # 绘制最大仓位变化
            axs[0, 0].plot(trade_df['date'], trade_df['max_position_size'])
            axs[0, 0].set_title('Max Position Size')
            axs[0, 0].set_xlabel('Date')
            axs[0, 0].set_ylabel('Position Size')
            axs[0, 0].tick_params(axis='x', rotation=45)
            
            # 绘制阈值变化
            axs[0, 1].plot(trade_df['date'], trade_df['threshold'])
            axs[0, 1].set_title('Signal Threshold')
            axs[0, 1].set_xlabel('Date')
            axs[0, 1].set_ylabel('Threshold')
            axs[0, 1].tick_params(axis='x', rotation=45)
            
            # 绘制止损变化
            axs[1, 0].plot(trade_df['date'], trade_df['stop_loss'])
            axs[1, 0].set_title('Stop Loss')
            axs[1, 0].set_xlabel('Date')
            axs[1, 0].set_ylabel('Stop Loss')
            axs[1, 0].tick_params(axis='x', rotation=45)
            
            # 绘制止盈变化
            axs[1, 1].plot(trade_df['date'], trade_df['take_profit'])
            axs[1, 1].set_title('Take Profit')
            axs[1, 1].set_xlabel('Date')
            axs[1, 1].set_ylabel('Take Profit')
            axs[1, 1].tick_params(axis='x', rotation=45)
            
            # 调整布局
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            
            if save:
                output_path = os.path.join(self.output_dir, 'parameter_changes.png')
                plt.savefig(output_path)
                logger.info(f"策略参数变化图保存成功: {output_path}")
            else:
                plt.show()
        except Exception as e:
            logger.error(f"绘制策略参数变化图失败: {e}")
    
    def plot_strategy_performance(self, backtest_results, save=False):
        """
        绘制策略性能分析图
        
        Args:
            backtest_results: 回测结果
            save: 是否保存图片
        """
        try:
            # 提取交易历史
            trade_history = backtest_results['trade_history']
            if not trade_history:
                logger.warning("回测结果中没有交易历史")
                return
            
            # 转换为DataFrame
            trade_df = pd.DataFrame(trade_history)
            
            # 计算交易统计
            total_trades = len(trade_df)
            buy_trades = len(trade_df[trade_df['action'] == 'buy'])
            sell_trades = len(trade_df[trade_df['action'] == 'sell'])
            
            # 计算胜率
            if sell_trades > 0:
                winning_trades = 0
                for i in range(1, len(trade_df)):
                    if trade_df.iloc[i]['action'] == 'sell':
                        buy_price = trade_df.iloc[i-1]['price']
                        sell_price = trade_df.iloc[i]['price']
                        if sell_price > buy_price:
                            winning_trades += 1
                win_rate = (winning_trades / sell_trades) * 100
            else:
                win_rate = 0
            
            # 计算收益
            initial_balance = backtest_results['initial_balance']
            final_balance = backtest_results['final_balance']
            total_return = backtest_results['total_return']
            
            # 准备数据
            performance_data = {
                'Metric': ['Total Trades', 'Buy Trades', 'Sell Trades', 'Win Rate'],
                'Value': [str(total_trades), str(buy_trades), str(sell_trades), f"{win_rate:.2f}%"]
            }
            
            performance_df = pd.DataFrame(performance_data)
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            # 绘制性能指标
            ax1.bar(performance_df['Metric'], performance_df['Value'], color='skyblue')
            ax1.set_title('Trading Performance')
            ax1.set_ylabel('Value')
            ax1.tick_params(axis='x', rotation=45)
            
            # 绘制余额变化
            ax2.plot(trade_df['date'], trade_df['balance'])
            ax2.set_title('Account Balance Over Time')
            ax2.set_xlabel('Date')
            ax2.set_ylabel('Balance (USD)')
            ax2.tick_params(axis='x', rotation=45)
            
            # 调整布局
            plt.tight_layout()
            
            if save:
                output_path = os.path.join(self.output_dir, 'strategy_performance.png')
                plt.savefig(output_path)
                logger.info(f"策略性能分析图保存成功: {output_path}")
            else:
                plt.show()
        except Exception as e:
            logger.error(f"绘制策略性能分析图失败: {e}")
    
    def plot_market_sentiment_vs_parameters(self, features_df, backtest_results, save=False):
        """
        绘制市场情绪与参数调整的关系图
        
        Args:
            features_df: 特征数据
            backtest_results: 回测结果
            save: 是否保存图片
        """
        try:
            # 提取交易历史
            trade_history = backtest_results['trade_history']
            if not trade_history:
                logger.warning("回测结果中没有交易历史")
                return
            
            # 转换为DataFrame
            trade_df = pd.DataFrame(trade_history)
            
            # 检查是否包含市场情绪数据
            if 'market_sentiment' not in features_df.columns:
                logger.warning("特征数据中没有市场情绪数据")
                return
            
            # 检查是否包含参数列
            param_columns = ['max_position_size', 'threshold', 'stop_loss', 'take_profit']
            if not all(col in trade_df.columns for col in param_columns):
                logger.warning("回测结果中没有参数数据")
                return
            
            # 合并数据
            # 确保日期类型一致
            features_df['date'] = features_df['date'].astype(str)
            trade_df['date'] = trade_df['date'].astype(str)
            merged_df = pd.merge(features_df, trade_df, on='date', how='inner')
            
            if merged_df.empty:
                logger.warning("合并数据为空")
                return
            
            fig, axs = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('Market Sentiment vs Strategy Parameters', fontsize=18)
            
            # 绘制市场情绪与最大仓位的关系
            axs[0, 0].scatter(merged_df['market_sentiment'], merged_df['max_position_size'])
            axs[0, 0].set_title('Market Sentiment vs Max Position Size')
            axs[0, 0].set_xlabel('Market Sentiment')
            axs[0, 0].set_ylabel('Max Position Size')
            
            # 绘制市场情绪与阈值的关系
            axs[0, 1].scatter(merged_df['market_sentiment'], merged_df['threshold'])
            axs[0, 1].set_title('Market Sentiment vs Threshold')
            axs[0, 1].set_xlabel('Market Sentiment')
            axs[0, 1].set_ylabel('Threshold')
            
            # 绘制市场情绪与止损的关系
            axs[1, 0].scatter(merged_df['market_sentiment'], merged_df['stop_loss'])
            axs[1, 0].set_title('Market Sentiment vs Stop Loss')
            axs[1, 0].set_xlabel('Market Sentiment')
            axs[1, 0].set_ylabel('Stop Loss')
            
            # 绘制市场情绪与止盈的关系
            axs[1, 1].scatter(merged_df['market_sentiment'], merged_df['take_profit'])
            axs[1, 1].set_title('Market Sentiment vs Take Profit')
            axs[1, 1].set_xlabel('Market Sentiment')
            axs[1, 1].set_ylabel('Take Profit')
            
            # 调整布局
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            
            if save:
                output_path = os.path.join(self.output_dir, 'sentiment_vs_parameters.png')
                plt.savefig(output_path)
                logger.info(f"市场情绪与参数关系图保存成功: {output_path}")
            else:
                plt.show()
        except Exception as e:
            logger.error(f"绘制市场情绪与参数关系图失败: {e}")
