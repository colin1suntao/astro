import pandas as pd
import requests
from datetime import datetime, timedelta
from loguru import logger
from src.config import config

class BitcoinData:
    """比特币数据获取类"""
    
    def __init__(self):
        """初始化"""
        self.base_url = "https://api.coingecko.com/api/v3"
        self.api_key = config.COINGECKO_API_KEY
        self.headers = {}
        if self.api_key:
            self.headers['x-cg-api-key'] = self.api_key
    
    def get_historical_data(self, days=365):
        """
        获取比特币历史价格数据
        
        Args:
            days: 获取多少天的数据
            
        Returns:
            DataFrame: 比特币历史价格数据
        """
        try:
            # 计算结束时间和开始时间
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # 构建API URL
            url = f"{self.base_url}/coins/bitcoin/market_chart/range"
            params = {
                'vs_currency': 'usd',
                'from': int(start_date.timestamp()),
                'to': int(end_date.timestamp())
            }
            
            # 发送请求
            response = requests.get(url, params=params, headers=self.headers)
            response.raise_for_status()
            
            # 解析响应
            data = response.json()
            
            # 构建DataFrame
            prices = data['prices']
            market_caps = data['market_caps']
            total_volumes = data['total_volumes']
            
            # 转换为DataFrame
            df_prices = pd.DataFrame(prices, columns=['timestamp', 'price'])
            df_market_caps = pd.DataFrame(market_caps, columns=['timestamp', 'market_cap'])
            df_volumes = pd.DataFrame(total_volumes, columns=['timestamp', 'volume'])
            
            # 合并数据
            df = df_prices.merge(df_market_caps, on='timestamp').merge(df_volumes, on='timestamp')
            
            # 转换时间戳
            df['date'] = pd.to_datetime(df['timestamp'], unit='ms').dt.date
            df['date'] = df['date'].astype(str)
            
            # 按日期分组，取每天的最后一个数据点
            df = df.groupby('date').last().reset_index()
            
            # 计算价格变化
            df['price_change'] = df['price'].pct_change() * 100
            df['price_change'] = df['price_change'].fillna(0)
            
            # 计算移动平均线
            df['ma7'] = df['price'].rolling(window=7).mean()
            df['ma30'] = df['price'].rolling(window=30).mean()
            
            return df
        except Exception as e:
            logger.error(f"获取比特币历史数据失败: {e}")
            return pd.DataFrame()
    
    def get_current_data(self):
        """
        获取比特币当前数据
        
        Returns:
            dict: 比特币当前数据
        """
        try:
            # 构建API URL
            url = f"{self.base_url}/simple/price"
            params = {
                'ids': 'bitcoin',
                'vs_currencies': 'usd',
                'include_market_cap': 'true',
                'include_24hr_vol': 'true',
                'include_24hr_change': 'true',
                'include_last_updated_at': 'true'
            }
            
            # 发送请求
            response = requests.get(url, params=params, headers=self.headers)
            response.raise_for_status()
            
            # 解析响应
            data = response.json()
            return data['bitcoin']
        except Exception as e:
            logger.error(f"获取比特币当前数据失败: {e}")
            return {}
    
    def get_blockchain_data(self):
        """
        获取比特币链上数据
        
        Returns:
            DataFrame: 比特币链上数据
        """
        try:
            # 构建API URL
            url = f"https://api.blockchain.com/v3/exchange/tickers/BTC-USD"
            
            # 发送请求
            response = requests.get(url)
            response.raise_for_status()
            
            # 解析响应
            data = response.json()
            
            # 构建DataFrame
            df = pd.DataFrame({
                'date': [datetime.now().strftime('%Y-%m-%d')],
                'last_price': [data['last_trade_price']],
                'volume_24h': [data['volume_24h']],
                'price_24h_change': [data['price_24h_change']]
            })
            
            return df
        except Exception as e:
            logger.error(f"获取比特币链上数据失败: {e}")
            return pd.DataFrame()
