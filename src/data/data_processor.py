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
            
            # 计算技术指标 - TradingView常用指标
            
            # 1. RSI (Relative Strength Index)
            def calculate_rsi(series, period=14):
                delta = series.diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                return rsi
            
            features_df['rsi'] = calculate_rsi(features_df['price'])
            
            # 2. MACD (Moving Average Convergence Divergence)
            exp1 = features_df['price'].ewm(span=12, adjust=False).mean()
            exp2 = features_df['price'].ewm(span=26, adjust=False).mean()
            features_df['macd'] = exp1 - exp2
            features_df['macd_signal'] = features_df['macd'].ewm(span=9, adjust=False).mean()
            features_df['macd_hist'] = features_df['macd'] - features_df['macd_signal']
            
            # 3. Bollinger Bands
            ma20 = features_df['price'].rolling(window=20).mean()
            std20 = features_df['price'].rolling(window=20).std()
            features_df['bb_upper'] = ma20 + (std20 * 2)
            features_df['bb_lower'] = ma20 - (std20 * 2)
            features_df['bb_width'] = (features_df['bb_upper'] - features_df['bb_lower']) / ma20
            features_df['bb_position'] = (features_df['price'] - features_df['bb_lower']) / (features_df['bb_upper'] - features_df['bb_lower'])
            
            # 4. Stochastic Oscillator
            low14 = features_df['price'].rolling(window=14).min()
            high14 = features_df['price'].rolling(window=14).max()
            features_df['stoch_k'] = ((features_df['price'] - low14) / (high14 - low14)) * 100
            features_df['stoch_d'] = features_df['stoch_k'].rolling(window=3).mean()
            
            # 5. ADX (Average Directional Index)
            def calculate_adx(high, low, close, period=14):
                plus_dm = high.diff()
                minus_dm = low.diff()
                plus_dm[plus_dm < 0] = 0
                minus_dm[minus_dm > 0] = 0
                minus_dm = abs(minus_dm)
                
                tr1 = abs(high - low)
                tr2 = abs(high - close.shift())
                tr3 = abs(low - close.shift())
                tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                
                atr = tr.rolling(window=period).mean()
                plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
                minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
                dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
                adx = dx.rolling(window=period).mean()
                return adx
            
            features_df['adx'] = calculate_adx(features_df['price'], features_df['price'], features_df['price'])
            
            # 6. 其他技术指标
            features_df['price_change_abs'] = abs(features_df['price_change'])
            features_df['price_change_volatility'] = features_df['price_change'].rolling(window=7).std()
            features_df['volume_change'] = features_df['volume'].pct_change() * 100
            features_df['volume_change'] = features_df['volume_change'].fillna(0)
            features_df['volume_ma7'] = features_df['volume'].rolling(window=7).mean()
            features_df['volume_ma30'] = features_df['volume'].rolling(window=30).mean()
            features_df['price_ma_diff'] = features_df['ma7'] - features_df['ma30']
            
            # 7. 市场情绪指数（模拟）
            # 基于价格波动和交易量的情绪指标
            features_df['market_sentiment'] = np.where(
                (features_df['price_change'] > 2) & (features_df['volume'] > features_df['volume_ma7']),
                1,  # 极度乐观
                np.where(
                    (features_df['price_change'] > 0.5) & (features_df['volume'] > features_df['volume_ma7']),
                    0.5,  # 乐观
                    np.where(
                        (features_df['price_change'] < -2) & (features_df['volume'] > features_df['volume_ma7']),
                        -1,  # 极度悲观
                        np.where(
                            (features_df['price_change'] < -0.5) & (features_df['volume'] > features_df['volume_ma7']),
                            -0.5,  # 悲观
                            0  # 中性
                        )
                    )
                )
            )
            
            # 8. 波动率指标
            features_df['volatility_7d'] = features_df['price'].rolling(window=7).std() / features_df['price'].rolling(window=7).mean()
            features_df['volatility_30d'] = features_df['price'].rolling(window=30).std() / features_df['price'].rolling(window=30).mean()
            
            # 计算占星特征
            # 行星位置的变化率
            planets = ['sun', 'moon', 'mercury', 'venus', 'mars', 'jupiter', 'saturn', 'uranus', 'neptune', 'pluto']
            for planet in planets:
                features_df[f"{planet}_ra_change"] = features_df[f"{planet}_ra"].pct_change() * 100
                features_df[f"{planet}_dec_change"] = features_df[f"{planet}_dec"].pct_change() * 100
                features_df[f"{planet}_ra_change"] = features_df[f"{planet}_ra_change"].fillna(0)
                features_df[f"{planet}_dec_change"] = features_df[f"{planet}_dec_change"].fillna(0)
            
            # 相位角度的标准化
            aspect_columns = [col for col in features_df.columns if '_' in col and not col.startswith('price') and not col.startswith('volume') and not col.startswith('market') and not col.startswith('ma') and not col in ['rsi', 'macd', 'macd_signal', 'macd_hist', 'bb_upper', 'bb_lower', 'bb_width', 'bb_position', 'stoch_k', 'stoch_d', 'adx', 'market_sentiment', 'volatility_7d', 'volatility_30d']]
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
            import traceback
            traceback.print_exc()
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
