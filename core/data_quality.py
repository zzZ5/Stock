"""
数据质量提升模块
提供异常值检测、缺失值处理、数据验证等功能
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy import stats
from core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class DataQualityReport:
    """数据质量报告"""
    total_rows: int
    total_columns: int
    missing_values: Dict[str, int]
    missing_percentage: Dict[str, float]
    outliers: Dict[str, int]
    outlier_percentage: Dict[str, float]
    duplicates: int
    data_types: Dict[str, str]
    summary: str


class DataCleaner:
    """数据清洗类"""

    def __init__(self, outlier_method: str = 'iqr', missing_strategy: str = 'forward_fill'):
        """
        初始化数据清洗器

        参数:
            outlier_method: 异常值检测方法 ('iqr', 'zscore', 'isolation_forest')
            missing_strategy: 缺失值处理策略 ('forward_fill', 'backward_fill',
                                             'mean', 'median', 'interpolate', 'drop')
        """
        self.outlier_method = outlier_method
        self.missing_strategy = missing_strategy

    def detect_outliers(self, df: pd.DataFrame, columns: List[str] = None,
                       threshold: float = 3.0) -> Dict[str, np.ndarray]:
        """
        检测异常值

        参数:
            df: 数据DataFrame
            columns: 要检测的列列表，None表示检测所有数值列
            threshold: 检测阈值

        返回:
            字典，键为列名，值为异常值的布尔数组
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        outliers = {}

        for col in columns:
            if col not in df.columns or df[col].isna().all():
                continue

            data = df[col].dropna()

            if len(data) < 10:
                continue

            if self.outlier_method == 'iqr':
                # 四分位距法
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers[col] = (df[col] < lower_bound) | (df[col] > upper_bound)

            elif self.outlier_method == 'zscore':
                # Z-score方法
                z_scores = np.abs(stats.zscore(data))
                outlier_mask = z_scores > threshold
                outliers[col] = df[col].isin(data[outlier_mask])

            else:
                logger.warning(f"未知的异常值检测方法: {self.outlier_method}")

        return outliers

    def handle_missing_values(self, df: pd.DataFrame, columns: List[str] = None) -> pd.DataFrame:
        """
        处理缺失值

        参数:
            df: 数据DataFrame
            columns: 要处理的列列表

        返回:
            处理后的DataFrame
        """
        df = df.copy()

        if columns is None:
            columns = df.columns.tolist()

        for col in columns:
            if col not in df.columns:
                continue

            if df[col].isna().any():
                if self.missing_strategy == 'forward_fill':
                    df[col] = df[col].ffill()
                    df[col] = df[col].bfill()  # 补充前向填充的剩余NA

                elif self.missing_strategy == 'backward_fill':
                    df[col] = df[col].bfill()
                    df[col] = df[col].ffill()

                elif self.missing_strategy == 'mean':
                    if pd.api.types.is_numeric_dtype(df[col]):
                        df[col] = df[col].fillna(df[col].mean())

                elif self.missing_strategy == 'median':
                    if pd.api.types.is_numeric_dtype(df[col]):
                        df[col] = df[col].fillna(df[col].median())

                elif self.missing_strategy == 'interpolate':
                    if pd.api.types.is_numeric_dtype(df[col]):
                        df[col] = df[col].interpolate(method='linear')

                elif self.missing_strategy == 'drop':
                    df = df.dropna(subset=[col])

        return df

    def remove_outliers(self, df: pd.DataFrame, columns: List[str] = None,
                       replace_method: str = 'clip') -> pd.DataFrame:
        """
        处理异常值

        参数:
            df: 数据DataFrame
            columns: 要处理的列列表
            replace_method: 替换方法 ('clip', 'median', 'mean', 'drop')

        返回:
            处理后的DataFrame
        """
        df = df.copy()
        outliers = self.detect_outliers(df, columns)

        for col, mask in outliers.items():
            if not mask.any():
                continue

            if replace_method == 'clip':
                # 使用IQR边界截断
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df.loc[mask, col] = df.loc[mask, col].clip(lower=lower_bound, upper=upper_bound)

            elif replace_method == 'median':
                median_val = df[col].median()
                df.loc[mask, col] = median_val

            elif replace_method == 'mean':
                mean_val = df[col].mean()
                df.loc[mask, col] = mean_val

            elif replace_method == 'drop':
                df = df[~mask]

        return df

    def clean_price_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        清洗价格数据（特殊处理OHLC数据）

        参数:
            df: 包含OHLC列的DataFrame

        返回:
            清洗后的DataFrame
        """
        df = df.copy()

        # 确保OHLC列存在
        price_cols = ['open', 'high', 'low', 'close']
        existing_cols = [col for col in price_cols if col in df.columns]

        if len(existing_cols) < 2:
            return df

        # 验证OHLC关系
        if 'high' in df.columns and 'low' in df.columns:
            # High不能低于Low
            invalid_hl = df['high'] < df['low']
            if invalid_hl.any():
                logger.warning(f"发现 {invalid_hl.sum()} 条High<Low的无效数据")
                df.loc[invalid_hl, 'high'] = df.loc[invalid_hl, 'low']

        # 验证Close在High和Low之间
        if 'close' in df.columns and 'high' in df.columns and 'low' in df.columns:
            invalid_close_high = df['close'] > df['high']
            invalid_close_low = df['close'] < df['low']

            if invalid_close_high.any():
                df.loc[invalid_close_high, 'close'] = df.loc[invalid_close_high, 'high']
            if invalid_close_low.any():
                df.loc[invalid_close_low, 'close'] = df.loc[invalid_close_low, 'low']

        # 处理负值和零值
        for col in existing_cols:
            if pd.api.types.is_numeric_dtype(df[col]):
                df.loc[df[col] <= 0, col] = np.nan

        # 前向填充价格数据
        df[existing_cols] = df[existing_cols].ffill()

        return df

    def check_data_consistency(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        检查数据一致性

        参数:
            df: 数据DataFrame

        返回:
            一致性检查结果
        """
        results = {}

        # 检查价格一致性
        if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            # High >= Low
            results['high_ge_low'] = (df['high'] >= df['low']).all()

            # Open/Close在High和Low之间
            results['close_in_range'] = ((df['close'] <= df['high']) &
                                        (df['close'] >= df['low'])).all()
            results['open_in_range'] = ((df['open'] <= df['high']) &
                                       (df['open'] >= df['low'])).all()

        # 检查成交量非负
        if 'volume' in df.columns:
            results['volume_non_negative'] = (df['volume'] >= 0).all()

        # 检查成交额非负
        if 'amount' in df.columns:
            results['amount_non_negative'] = (df['amount'] >= 0).all()

        return results


class DataValidator:
    """数据验证类"""

    @staticmethod
    def generate_quality_report(df: pd.DataFrame,
                               outlier_method: str = 'iqr') -> DataQualityReport:
        """
        生成数据质量报告

        参数:
            df: 数据DataFrame
            outlier_method: 异常值检测方法

        返回:
            数据质量报告
        """
        # 缺失值统计
        missing_values = df.isna().sum().to_dict()
        missing_percentage = (df.isna().sum() / len(df) * 100).to_dict()

        # 异常值检测
        cleaner = DataCleaner(outlier_method=outlier_method)
        outliers = cleaner.detect_outliers(df)
        outlier_counts = {col: mask.sum() for col, mask in outliers.items()}
        outlier_percentage = {col: mask.sum() / df[col].notna().sum() * 100
                            for col, mask in outliers.items() if df[col].notna().any()}

        # 重复行统计
        duplicates = df.duplicated().sum()

        # 数据类型
        data_types = df.dtypes.astype(str).to_dict()

        # 生成摘要
        total_missing = sum(missing_values.values())
        total_outliers = sum(outlier_counts.values())
        summary = (f"数据质量报告:\n"
                  f"  - 总行数: {len(df)}, 总列数: {len(df.columns)}\n"
                  f"  - 缺失值: {total_missing} ({total_missing/len(df)/len(df.columns)*100:.2f}%)\n"
                  f"  - 异常值: {total_outliers}\n"
                  f"  - 重复行: {duplicates}\n"
                  f"  - 数据完整性: {'良好' if total_missing == 0 else '存在问题'}")

        return DataQualityReport(
            total_rows=len(df),
            total_columns=len(df.columns),
            missing_values=missing_values,
            missing_percentage=missing_percentage,
            outliers=outlier_counts,
            outlier_percentage=outlier_percentage,
            duplicates=duplicates,
            data_types=data_types,
            summary=summary
        )

    @staticmethod
    def validate_stock_data(df: pd.DataFrame, min_length: int = 10) -> Tuple[bool, List[str]]:
        """
        验证股票数据有效性

        参数:
            df: 股票数据DataFrame
            min_length: 最小数据长度

        返回:
            (是否有效, 错误信息列表)
        """
        errors = []

        if df is None or len(df) < min_length:
            errors.append(f"数据长度不足: {len(df) if df is not None else 0} < {min_length}")
            return False, errors

        # 检查必要列
        required_cols = ['open', 'high', 'low', 'close']
        for col in required_cols:
            if col not in df.columns:
                errors.append(f"缺少必要列: {col}")

        # 检查数据类型
        for col in required_cols:
            if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
                errors.append(f"列 {col} 不是数值类型")

        # 检查缺失值
        for col in required_cols:
            if col in df.columns:
                missing_ratio = df[col].isna().sum() / len(df)
                if missing_ratio > 0.5:
                    errors.append(f"列 {col} 缺失值过多: {missing_ratio*100:.1f}%")

        # 检查价格有效性
        if all(col in df.columns for col in ['high', 'low', 'close']):
            if (df['high'] < df['low']).any():
                errors.append("存在high < low的无效数据")

            invalid_close = (df['close'] > df['high']) | (df['close'] < df['low'])
            if invalid_close.any():
                errors.append(f"存在 {invalid_close.sum()} 条收盘价超出高低点的无效数据")

        return len(errors) == 0, errors


class DataPipeline:
    """数据处理流水线"""

    def __init__(self,
                 outlier_method: str = 'iqr',
                 missing_strategy: str = 'forward_fill',
                 outlier_replace: str = 'clip'):
        """
        初始化数据处理流水线

        参数:
            outlier_method: 异常值检测方法
            missing_strategy: 缺失值处理策略
            outlier_replace: 异常值替换方法
        """
        self.cleaner = DataCleaner(outlier_method=outlier_method,
                                   missing_strategy=missing_strategy)
        self.outlier_replace = outlier_replace

    def process(self, df: pd.DataFrame,
                generate_report: bool = False) -> pd.DataFrame:
        """
        执行完整的数据处理流程

        参数:
            df: 原始数据DataFrame
            generate_report: 是否生成数据质量报告

        返回:
            处理后的DataFrame
        """
        logger.info("开始数据处理流程")

        # 1. 清洗价格数据
        df = self.cleaner.clean_price_data(df)

        # 2. 处理缺失值
        missing_before = df.isna().sum().sum()
        df = self.cleaner.handle_missing_values(df)
        missing_after = df.isna().sum().sum()
        logger.info(f"缺失值处理: {missing_before} -> {missing_after}")

        # 3. 处理异常值
        outliers = self.cleaner.detect_outliers(df)
        outlier_count = sum(mask.sum() for mask in outliers.values())
        df = self.cleaner.remove_outliers(df, replace_method=self.outlier_replace)
        logger.info(f"异常值处理: {outlier_count} 个异常值")

        # 4. 数据一致性检查
        consistency = self.cleaner.check_data_consistency(df)
        inconsistent = [k for k, v in consistency.items() if not v]
        if inconsistent:
            logger.warning(f"数据一致性问题: {', '.join(inconsistent)}")

        # 5. 生成质量报告
        if generate_report:
            report = DataValidator.generate_quality_report(df)
            logger.info(report.summary)

        logger.info("数据处理流程完成")
        return df

    def batch_process(self, dfs: Dict[str, pd.DataFrame],
                     progress_callback=None) -> Dict[str, pd.DataFrame]:
        """
        批量处理多个DataFrame

        参数:
            dfs: 字典 {股票代码: DataFrame}
            progress_callback: 进度回调函数

        返回:
            处理后的DataFrame字典
        """
        results = {}

        for i, (code, df) in enumerate(dfs.items()):
            results[code] = self.process(df)

            if progress_callback:
                progress_callback(i + 1, len(dfs))

        return results


def fill_missing_with_linear_trend(df: pd.DataFrame, columns: List[str] = None) -> pd.DataFrame:
    """
    使用线性趋势填充缺失值（适用于时间序列数据）

    参数:
        df: 数据DataFrame
        columns: 要处理的列列表

    返回:
        填充后的DataFrame
    """
    df = df.copy()

    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    for col in columns:
        if col not in df.columns or not df[col].isna().any():
            continue

        # 获取有效数据的索引
        valid_idx = df[col].notna()
        if valid_idx.sum() < 2:
            continue

        # 线性插值
        df[col] = df[col].interpolate(method='linear', limit_direction='both')

    return df


def detect_price_anomalies(df: pd.DataFrame,
                          price_col: str = 'close',
                          window: int = 20,
                          threshold: float = 3.0) -> pd.Series:
    """
    检测价格异常（基于滚动统计）

    参数:
        df: 数据DataFrame
        price_col: 价格列名
        window: 滚动窗口大小
        threshold: 异常阈值（标准差倍数）

    返回:
        异常值的布尔Series
    """
    if price_col not in df.columns:
        return pd.Series([False] * len(df), index=df.index)

    rolling_mean = df[price_col].rolling(window=window, min_periods=1).mean()
    rolling_std = df[price_col].rolling(window=window, min_periods=1).std()

    z_score = np.abs((df[price_col] - rolling_mean) / rolling_std)
    anomalies = z_score > threshold

    return anomalies.fillna(False)
