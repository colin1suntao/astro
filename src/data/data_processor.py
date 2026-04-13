import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from loguru import logger

class DataProcessor:
    """数据处理类"""
    
    def __init__(self):
        """初始化"""
        self.scaler = StandardScaler()
    
    def merge_data(self, astro_df, bitcoin_df):
        """
        合并占星数据和比特币价格数据
        
        Args:
            astro_df: 占星数据
            bitcoin_df: 比特币价格数据
            
        Returns:
            DataFrame: 合并后的数据
        """
        try:
            # 按日期合并数据
            merged_df = pd.merge(bitcoin_df, astro_df, on='date', how='inner')
            logger.info(f"合并数据成功，共 {len(merged_df)} 条记录")
            return merged_df
        except Exception as e:
            logger.error(f"合并数据失败: {e}")
            return pd.DataFrame()
    
    def extract_features(self, df):
        """
        提取特征
        
        Args:
            df: 原始数据
            
        Returns:
            DataFrame: 提取特征后的数据
        """
        try:
            # 复制数据
            features_df = df.copy()
            
            # 计算技术指标
            features_df['price_change_abs'] = abs(features_df['price_change'])
            features_df['price_change_volatility'] = features_df['price_change'].rolling(window=7).std()
            features_df['volume_change'] = features_df['volume'].pct_change() * 100
            features_df['volume_change'] = features_df['volume_change'].fillna(0)
            
            # 计算占星特征
            # 行星位置的变化率
            planets = ['sun', 'moon', 'mercury', 'venus', 'mars', 'jupiter', 'saturn', 'uranus', 'neptune', 'pluto']
            for planet in planets:
                features_df[f"{planet}_ra_change"] = features_df[f"{planet}_ra"].pct_change() * 100
                features_df[f"{planet}_dec_change"] = features_df[f"{planet}_dec"].pct_change() * 100
                features_df[f"{planet}_ra_change"] = features_df[f"{planet}_ra_change"].fillna(0)
                features_df[f"{planet}_dec_change"] = features_df[f"{planet}_dec_change"].fillna(0)
            
            # 相位角度的标准化
            aspect_columns = [col for col in features_df.columns if '_' in col and not col.startswith('price') and not col.startswith('volume') and not col.startswith('market') and not col.startswith('ma')]
            for col in aspect_columns:
                if col not in ['date', 'timestamp']:
                    # 标准化相位角度（0-180度）
                    features_df[col] = features_df[col] / 180.0
            
            # 填充缺失值
            features_df = features_df.fillna(0)
            
            logger.info("特征提取成功")
            return features_df
        except Exception as e:
            logger.error(f"提取特征失败: {e}")
            return pd.DataFrame()
    
    def preprocess_data(self, df, target_col='price_change'):
        """
        预处理数据
        
        Args:
            df: 原始数据
            target_col: 目标列
            
        Returns:
            X: 特征数据
            y: 目标数据
        """
        try:
            # 选择特征列
            feature_columns = [col for col in df.columns if col not in ['date', 'timestamp', target_col, 'price', 'market_cap', 'volume', 'ma7', 'ma30']]
            
            # 提取特征和目标
            X = df[feature_columns].values
            y = df[target_col].values
            
            # 标准化特征
            X = self.scaler.fit_transform(X)
            
            logger.info(f"预处理数据成功，特征维度: {X.shape}")
            return X, y
        except Exception as e:
            logger.error(f"预处理数据失败: {e}")
            return None, None
    
    def split_data(self, X, y, test_size=0.2):
        """
        分割数据
        
        Args:
            X: 特征数据
            y: 目标数据
            test_size: 测试集比例
            
        Returns:
            X_train, X_test, y_train, y_test: 分割后的数据
        """
        try:
            # 计算分割点
            split_idx = int(len(X) * (1 - test_size))
            
            # 分割数据
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            logger.info(f"分割数据成功，训练集: {len(X_train)}，测试集: {len(X_test)}")
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logger.error(f"分割数据失败: {e}")
            return None, None, None, None
