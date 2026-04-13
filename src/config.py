import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

class Config:
    """项目配置类"""
    # 数据获取配置
    COINGECKO_API_KEY = os.getenv('COINGECKO_API_KEY', '')
    BLOCKCHAIN_API_KEY = os.getenv('BLOCKCHAIN_API_KEY', '')
    
    # 数据库配置
    DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///data.db')
    
    # 交易配置
    TRADING_API_KEY = os.getenv('TRADING_API_KEY', '')
    TRADING_SECRET = os.getenv('TRADING_SECRET', '')
    TRADING_MODE = os.getenv('TRADING_MODE', 'paper')
    
    # 模型配置
    MODEL_TYPE = os.getenv('MODEL_TYPE', 'xgboost')
    TRAINING_PERIOD = int(os.getenv('TRAINING_PERIOD', '365'))
    PREDICTION_HORIZON = int(os.getenv('PREDICTION_HORIZON', '7'))
    
    # 风险管理配置
    MAX_POSITION_SIZE = float(os.getenv('MAX_POSITION_SIZE', '0.1'))
    STOP_LOSS = float(os.getenv('STOP_LOSS', '0.05'))
    TAKE_PROFIT = float(os.getenv('TAKE_PROFIT', '0.1'))
    
    # 日志配置
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    
    # 数据目录
    DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
    PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
    
    # 确保目录存在
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

# 创建设置实例
config = Config()
