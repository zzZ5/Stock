"""
机器学习因子模块
提供特征工程、模型训练、预测评分等功能
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import os
from core.logger import get_logger

logger = get_logger(__name__)

# 尝试导入XGBoost和LightGBM（可选）
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    logger.warning("XGBoost未安装，将使用sklearn模型")

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    logger.warning("LightGBM未安装，将使用sklearn模型")


@dataclass
class PredictionResult:
    """预测结果"""
    ts_code: str
    predicted_return: float
    prediction_score: float  # 预测分数 (0-100)
    feature_importance: Dict[str, float]
    confidence: float  # 预测置信度


class FeatureEngineer:
    """特征工程类"""

    def __init__(self):
        """初始化特征工程"""
        self.scaler: StandardScaler = None
        self.feature_names: List[str] = []

    def create_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        创建技术面特征

        参数:
            df: 价格数据

        返回:
            特征DataFrame
        """
        features = pd.DataFrame(index=df.index)
        close = df['close'].values

        # 价格相关特征
        features['price_change_1d'] = df['close'].pct_change(1)
        features['price_change_5d'] = df['close'].pct_change(5)
        features['price_change_20d'] = df['close'].pct_change(20)

        # 均线相关
        features['ma_ratio_5'] = df['close'] / df['close'].rolling(5).mean()
        features['ma_ratio_20'] = df['close'] / df['close'].rolling(20).mean()
        features['ma_ratio_60'] = df['close'] / df['close'].rolling(60).mean()

        # 波动率特征
        features['volatility_5d'] = df['close'].pct_change(1).rolling(5).std()
        features['volatility_20d'] = df['close'].pct_change(1).rolling(20).std()

        # 最高最低价
        features['high_low_ratio'] = df['high'] / df['low']
        features['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])

        # 成交量特征
        if 'volume' in df.columns:
            features['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
            features['volume_volatility'] = df['volume'].pct_change(1).rolling(5).std()

        return features

    def create_lag_features(self, series: pd.Series, lags: List[int] = [1, 2, 3, 5, 10]) -> pd.DataFrame:
        """
        创建滞后特征

        参数:
            series: 时间序列
            lags: 滞后期数列表

        返回:
            滞后特征DataFrame
        """
        features = pd.DataFrame(index=series.index)

        for lag in lags:
            features[f'lag_{lag}'] = series.shift(lag)

        return features

    def create_rolling_features(self, series: pd.Series,
                               windows: List[int] = [5, 10, 20, 60],
                               functions: List[str] = ['mean', 'std', 'min', 'max']) -> pd.DataFrame:
        """
        创建滚动特征

        参数:
            series: 时间序列
            windows: 窗口大小列表
            functions: 统计函数列表

        返回:
            滚动特征DataFrame
        """
        features = pd.DataFrame(index=series.index)

        for window in windows:
            for func in functions:
                if func == 'mean':
                    features[f'{func}_{window}d'] = series.rolling(window).mean()
                elif func == 'std':
                    features[f'{func}_{window}d'] = series.rolling(window).std()
                elif func == 'min':
                    features[f'{func}_{window}d'] = series.rolling(window).min()
                elif func == 'max':
                    features[f'{func}_{window}d'] = series.rolling(window).max()

        return features

    def combine_factors(self, factor_dict: Dict[str, float],
                      feature_names: List[str] = None) -> np.ndarray:
        """
        组合因子为特征向量

        参数:
            factor_dict: 因子字典
            feature_names: 特征名称列表（用于排序）

        返回:
            特征向量
        """
        if feature_names is None:
            feature_names = sorted(factor_dict.keys())

        features = []
        for name in feature_names:
            value = factor_dict.get(name, np.nan)
            features.append(value if not pd.isna(value) else 0)

        return np.array(features)

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        拟合并转换数据

        参数:
            X: 特征DataFrame

        返回:
            标准化后的特征DataFrame
        """
        self.feature_names = X.columns.tolist()
        self.scaler = StandardScaler()

        scaled = self.scaler.fit_transform(X)
        scaled_df = pd.DataFrame(scaled, columns=X.columns, index=X.index)

        return scaled_df

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        转换数据

        参数:
            X: 特征DataFrame

        返回:
            标准化后的特征DataFrame
        """
        if self.scaler is None:
            raise ValueError("Scaler未拟合，请先调用fit_transform")

        # 确保列顺序一致
        X_aligned = X[self.feature_names]

        scaled = self.scaler.transform(X_aligned)
        scaled_df = pd.DataFrame(scaled, columns=self.feature_names, index=X.index)

        return scaled_df


class MLPredictor:
    """机器学习预测器"""

    def __init__(self, model_type: str = 'random_forest'):
        """
        初始化预测器

        参数:
            model_type: 模型类型 ('random_forest', 'gradient_boosting', 'xgboost', 'lightgbm')
        """
        self.model_type = model_type
        self.model = None
        self.feature_engineer = FeatureEngineer()
        self.is_trained = False

        # 创建模型
        self._create_model()

    def _create_model(self):
        """创建模型实例"""
        if self.model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == 'gradient_boosting':
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
        elif self.model_type == 'xgboost' and HAS_XGBOOST:
            self.model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == 'lightgbm' and HAS_LIGHTGBM:
            self.model = lgb.LGBMRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1
            )
        else:
            logger.warning(f"不支持的模型类型: {self.model_type}, 使用Random Forest")
            self.model_type = 'random_forest'
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )

    def train(self, X: pd.DataFrame, y: pd.Series,
              validation_split: float = 0.2,
              use_time_series_cv: bool = True) -> Dict[str, float]:
        """
        训练模型

        参数:
            X: 特征DataFrame
            y: 目标变量（未来收益率）
            validation_split: 验证集比例
            use_time_series_cv: 是否使用时间序列交叉验证

        返回:
            训练指标字典
        """
        # 移除缺失值
        X_clean = X.dropna()
        y_clean = y.loc[X_clean.index]

        if len(X_clean) < 50:
            logger.warning("训练数据不足，跳过训练")
            return {}

        # 特征标准化
        X_scaled = self.feature_engineer.fit_transform(X_clean)

        if use_time_series_cv:
            # 时间序列交叉验证
            tscv = TimeSeriesSplit(n_splits=5)

            cv_scores = cross_val_score(
                self.model, X_scaled, y_clean,
                cv=tscv, scoring='r2', n_jobs=-1
            )

            logger.info(f"交叉验证R2分数: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

        # 训练模型
        self.model.fit(X_scaled, y_clean)
        self.is_trained = True

        # 计算训练指标
        y_pred = self.model.predict(X_scaled)

        metrics = {
            'mse': mean_squared_error(y_clean, y_pred),
            'mae': mean_absolute_error(y_clean, y_pred),
            'r2': r2_score(y_clean, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_clean, y_pred))
        }

        logger.info(f"模型训练完成 - R2: {metrics['r2']:.4f}, RMSE: {metrics['rmse']:.4f}")

        return metrics

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        预测

        参数:
            X: 特征DataFrame

        返回:
            预测结果Series
        """
        if not self.is_trained:
            raise ValueError("模型未训练，请先调用train方法")

        X_scaled = self.feature_engineer.transform(X)
        predictions = self.model.predict(X_scaled)

        return pd.Series(predictions, index=X.index)

    def predict_single(self, factors: Dict[str, float]) -> PredictionResult:
        """
        预测单个股票

        参数:
            factors: 因子字典

        返回:
            预测结果
        """
        if not self.is_trained:
            raise ValueError("模型未训练，请先调用train方法")

        # 创建DataFrame
        factor_df = pd.DataFrame([factors])

        # 只使用训练时的特征
        feature_names = self.feature_engineer.feature_names
        available_features = [f for f in feature_names if f in factor_df.columns]

        if not available_features:
            logger.warning("没有可用的特征")
            return PredictionResult(
                ts_code='unknown',
                predicted_return=0.0,
                prediction_score=50.0,
                feature_importance={},
                confidence=0.0
            )

        factor_df = factor_df[available_features]

        # 预测
        prediction = self.predict(factor_df)

        # 获取特征重要性
        feature_importance = self.get_feature_importance()

        # 计算置信度
        confidence = self._calculate_confidence(factors)

        # 转换为分数 (0-100)
        prediction_score = self._return_to_score(prediction.iloc[0])

        return PredictionResult(
            ts_code='unknown',  # 需要外部传入
            predicted_return=prediction.iloc[0],
            prediction_score=prediction_score,
            feature_importance=feature_importance,
            confidence=confidence
        )

    def _return_to_score(self, predicted_return: float) -> float:
        """
        将预测收益率转换为分数

        参数:
            predicted_return: 预测收益率

        返回:
            分数 (0-100)
        """
        # 假设收益率范围在[-10%, 10%]之间
        score = 50 + predicted_return * 500
        return np.clip(score, 0, 100)

    def _calculate_confidence(self, factors: Dict[str, float]) -> float:
        """
        计算预测置信度

        参数:
            factors: 因子字典

        返回:
            置信度 (0-1)
        """
        # 简单实现：基于特征完整性
        available_ratio = len([v for v in factors.values() if not pd.isna(v)]) / len(factors)
        return available_ratio

    def get_feature_importance(self) -> Dict[str, float]:
        """
        获取特征重要性

        返回:
            特征重要性字典
        """
        if not self.is_trained:
            return {}

        feature_names = self.feature_engineer.feature_names

        if hasattr(self.model, 'feature_importances_'):
            importance_dict = dict(zip(feature_names, self.model.feature_importances_))
        elif hasattr(self.model, 'coef_'):
            importance_dict = dict(zip(feature_names, np.abs(self.model.coef_)))
        else:
            importance_dict = {}

        return importance_dict

    def save_model(self, filepath: str):
        """
        保存模型

        参数:
            filepath: 文件路径
        """
        model_data = {
            'model': self.model,
            'scaler': self.feature_engineer.scaler,
            'feature_names': self.feature_engineer.feature_names,
            'model_type': self.model_type,
            'is_trained': self.is_trained
        }

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(model_data, filepath)

        logger.info(f"模型已保存到: {filepath}")

    def load_model(self, filepath: str):
        """
        加载模型

        参数:
            filepath: 文件路径
        """
        model_data = joblib.load(filepath)

        self.model = model_data['model']
        self.feature_engineer.scaler = model_data['scaler']
        self.feature_engineer.feature_names = model_data['feature_names']
        self.model_type = model_data['model_type']
        self.is_trained = model_data['is_trained']

        logger.info(f"模型已从 {filepath} 加载")


class EnsemblePredictor:
    """集成预测器"""

    def __init__(self, predictors: List[MLPredictor] = None):
        """
        初始化集成预测器

        参数:
            predictors: 预测器列表
        """
        self.predictors = predictors if predictors else []
        self.weights = [1.0 / len(self.predictors)] * len(self.predictors) if self.predictors else []

    def add_predictor(self, predictor: MLPredictor, weight: float = None):
        """
        添加预测器

        参数:
            predictor: 预测器实例
            weight: 权重（None表示平均分配）
        """
        self.predictors.append(predictor)

        if weight is None:
            self.weights = [1.0 / len(self.predictors)] * len(self.predictors)
        else:
            self.weights.append(weight)
            # 归一化权重
            total = sum(self.weights)
            self.weights = [w / total for w in self.weights]

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        集成预测

        参数:
            X: 特征DataFrame

        返回:
            集成预测结果
        """
        if not self.predictors:
            raise ValueError("没有可用的预测器")

        predictions = []
        for predictor in self.predictors:
            if predictor.is_trained:
                predictions.append(predictor.predict(X))

        if not predictions:
            raise ValueError("没有已训练的预测器")

        # 加权平均
        weighted_predictions = np.zeros(len(predictions[0]))
        for pred, weight in zip(predictions, self.weights):
            weighted_predictions += pred * weight

        return pd.Series(weighted_predictions, index=X.index)

    def get_feature_importance(self) -> Dict[str, float]:
        """
        获取集成特征重要性

        返回:
            特征重要性字典
        """
        if not self.predictors:
            return {}

        # 平均特征重要性
        all_importance = {}

        for predictor, weight in zip(self.predictors, self.weights):
            importance = predictor.get_feature_importance()
            for feature, value in importance.items():
                all_importance[feature] = all_importance.get(feature, 0) + value * weight

        return all_importance
