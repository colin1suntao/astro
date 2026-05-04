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
            
            # 6. 高级技术指标
            
            # ATR (Average True Range)
            def calculate_atr(high, low, close, period=14):
                tr1 = abs(high - low)
                tr2 = abs(high - close.shift())
                tr3 = abs(low - close.shift())
                tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                atr = tr.rolling(window=period).mean()
                return atr
            
            features_df['atr'] = calculate_atr(features_df['price'], features_df['price'], features_df['price'])
            
            # OBV (On-Balance Volume)
            def calculate_obv(close, volume):
                obv = np.zeros(len(close))
                obv[0] = volume.iloc[0]
                for i in range(1, len(close)):
                    if close.iloc[i] > close.iloc[i-1]:
                        obv[i] = obv[i-1] + volume.iloc[i]
                    elif close.iloc[i] < close.iloc[i-1]:
                        obv[i] = obv[i-1] - volume.iloc[i]
                    else:
                        obv[i] = obv[i-1]
                return obv
            
            features_df['obv'] = calculate_obv(features_df['price'], features_df['volume'])
            features_df['obv_ma7'] = features_df['obv'].rolling(window=7).mean()
            
            # ROC (Rate of Change)
            def calculate_roc(series, period=12):
                return ((series / series.shift(period)) - 1) * 100
            
            features_df['roc'] = calculate_roc(features_df['price'])
            
            # CCI (Commodity Channel Index)
            def calculate_cci(high, low, close, period=20):
                tp = (high + low + close) / 3
                ma = tp.rolling(window=period).mean()
                mad = tp.rolling(window=period).apply(lambda x: np.mean(np.abs(x - x.mean())))
                cci = (tp - ma) / (0.015 * mad)
                return cci
            
            features_df['cci'] = calculate_cci(features_df['price'], features_df['price'], features_df['price'])
            
            # 7. 其他技术指标
            features_df['price_change_abs'] = abs(features_df['price_change'])
            features_df['price_change_volatility'] = features_df['price_change'].rolling(window=7).std()
            features_df['volume_change'] = features_df['volume'].pct_change() * 100
            features_df['volume_change'] = features_df['volume_change'].fillna(0)
            features_df['volume_ma7'] = features_df['volume'].rolling(window=7).mean()
            features_df['volume_ma30'] = features_df['volume'].rolling(window=30).mean()
            features_df['price_ma_diff'] = features_df['ma7'] - features_df['ma30']
            
            # 7. 市场情绪指数（模拟）
            # 基于价格波动、交易量和技术指标的情绪指标
            def calculate_market_sentiment(df):
                sentiment = np.zeros(len(df))
                
                for i in range(len(df)):
                    # 基础情绪得分
                    base_score = 0
                    
                    # 价格变化
                    price_change = df.iloc[i]['price_change']
                    if price_change > 2:
                        base_score += 0.5
                    elif price_change > 0.5:
                        base_score += 0.25
                    elif price_change < -2:
                        base_score -= 0.5
                    elif price_change < -0.5:
                        base_score -= 0.25
                    
                    # 交易量
                    volume = df.iloc[i]['volume']
                    volume_ma7 = df.iloc[i]['volume_ma7']
                    if volume > volume_ma7 * 1.2:
                        base_score *= 1.2  # 放大情绪
                    elif volume < volume_ma7 * 0.8:
                        base_score *= 0.8  # 减弱情绪
                    
                    # RSI指标
                    rsi = df.iloc[i]['rsi']
                    if rsi > 70:
                        base_score += 0.1  # 超买，增强乐观
                    elif rsi < 30:
                        base_score -= 0.1  # 超卖，增强悲观
                    
                    # MACD指标
                    macd = df.iloc[i]['macd']
                    if macd > 0:
                        base_score += 0.1
                    else:
                        base_score -= 0.1
                    
                    # CCI指标
                    cci = df.iloc[i]['cci']
                    if cci > 100:
                        base_score += 0.1
                    elif cci < -100:
                        base_score -= 0.1
                    
                    # 标准化情绪得分
                    sentiment[i] = max(min(base_score, 1), -1)
                
                return sentiment
            
            features_df['market_sentiment'] = calculate_market_sentiment(features_df)
            
            # 8. 波动率指标
            features_df['volatility_7d'] = features_df['price'].rolling(window=7).std() / features_df['price'].rolling(window=7).mean()
            features_df['volatility_30d'] = features_df['price'].rolling(window=30).std() / features_df['price'].rolling(window=30).mean()
            
            # 9. 市场动量指标
            features_df['momentum_14'] = features_df['price'] - features_df['price'].shift(14)
            features_df['momentum_30'] = features_df['price'] - features_df['price'].shift(30)
            
            # 10. 波动率变化率
            features_df['volatility_change'] = features_df['volatility_7d'].pct_change() * 100
            features_df['volatility_change'] = features_df['volatility_change'].fillna(0)
            
            # 11. 交易量指标
            features_df['volume_volatility'] = features_df['volume'].rolling(window=7).std() / features_df['volume'].rolling(window=7).mean()
            features_df['volume_ratio'] = features_df['volume'] / features_df['volume_ma7']
            
            # 12. 价格模式指标
            features_df['price_range'] = features_df['price'].rolling(window=7).max() - features_df['price'].rolling(window=7).min()
            features_df['price_range_ratio'] = features_df['price_range'] / features_df['price']
            
            # 13. 宏观经济指标（模拟）
            # 模拟利率变化
            features_df['interest_rate_change'] = np.random.normal(0, 0.1, len(features_df))
            # 模拟通货膨胀率
            features_df['inflation_rate'] = np.random.uniform(1, 5, len(features_df))
            # 模拟市场恐慌指数（类似VIX）
            features_df['fear_greed_index'] = np.random.uniform(0, 100, len(features_df))
            
            # 14. 加密货币特定指标
            # 模拟网络哈希率变化
            features_df['hash_rate_change'] = np.random.normal(0, 5, len(features_df))
            # 模拟活跃地址数变化
            features_df['active_addresses_change'] = np.random.normal(0, 10, len(features_df))
            
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
