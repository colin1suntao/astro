import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from loguru import logger
import joblib
import os
from src.config import config

class ModelTrainer:
    """模型训练类"""
    
    def __init__(self, model_type='xgboost'):
        """
        初始化
        
        Args:
            model_type: 模型类型，可选值：xgboost, random_forest
        """
        self.model_type = model_type
        self.model = None
    
    def train(self, X_train, y_train):
        """
        训练模型
        
        Args:
            X_train: 训练特征数据
            y_train: 训练目标数据
            
        Returns:
            model: 训练好的模型
        """
        try:
            if self.model_type == 'random_forest':
                # 随机森林模型
                self.model = RandomForestRegressor(n_estimators=100, random_state=42)
                self.model.fit(X_train, y_train)
            elif self.model_type == 'xgboost':
                # XGBoost模型 - 使用优化后的参数
                self.model = XGBRegressor(
                    n_estimators=100,
                    learning_rate=0.05,
                    max_depth=5,
                    subsample=1.0,
                    colsample_bytree=0.9,
                    random_state=42
                )
                self.model.fit(X_train, y_train)
            else:
                raise ValueError(f"不支持的模型类型: {self.model_type}")
            
            logger.info(f"{self.model_type} 模型训练成功")
            return self.model
        except Exception as e:
            logger.error(f"训练模型失败: {e}")
            return None
    
    def predict(self, X_test):
        """
        预测
        
        Args:
            X_test: 测试特征数据
            
        Returns:
            y_pred: 预测结果
        """
        try:
            if self.model is None:
                raise ValueError("模型未训练")
            
            y_pred = self.model.predict(X_test)
            
            logger.info("预测成功")
            return y_pred
        except Exception as e:
            logger.error(f"预测失败: {e}")
            return None
    
    def evaluate(self, y_true, y_pred):
        """
        评估模型
        
        Args:
            y_true: 真实值
            y_pred: 预测值
            
        Returns:
            dict: 评估指标
        """
        try:
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_true, y_pred)
            
            metrics = {
                'mse': mse,
                'rmse': rmse,
                'r2': r2
            }
            
            logger.info(f"模型评估成功: MSE={mse:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}")
            return metrics
        except Exception as e:
            logger.error(f"评估模型失败: {e}")
            return {}
    
    def save_model(self, model_path):
        """
        保存模型
        
        Args:
            model_path: 模型保存路径
        """
        try:
            if self.model is None:
                raise ValueError("模型未训练")
            
            # 确保目录存在
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            # 保存模型
            joblib.dump(self.model, model_path)
            
            logger.info(f"模型保存成功: {model_path}")
        except Exception as e:
            logger.error(f"保存模型失败: {e}")
    
    def load_model(self, model_path):
        """
        加载模型
        
        Args:
            model_path: 模型加载路径
        """
        try:
            # 加载模型
            self.model = joblib.load(model_path)
            
            logger.info(f"模型加载成功: {model_path}")
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
