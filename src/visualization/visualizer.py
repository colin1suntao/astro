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
