"""
趋势雷达选股系统 - 数据验证模块
提供数据验证、类型检查、参数校验等功能
"""
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Any, Optional, List, Dict, Union

from core.logger import get_validators_logger


class ValidationError(Exception):
    """数据验证错误"""
    pass


class DataFrameValidator:
    """DataFrame验证器"""

    @staticmethod
    def validate_dataframe(df: pd.DataFrame, required_columns: List[str],
                          df_name: str = "DataFrame") -> pd.DataFrame:
        """
        验证DataFrame是否存在必要的列

        参数:
            df: 待验证的DataFrame
            required_columns: 必需的列名列表
            df_name: DataFrame名称(用于错误提示)

        返回:
            验证通过的DataFrame

        抛出:
            ValidationError: 验证失败
        """
        logger = get_validators_logger()

        if df is None:
            logger.error(f"{df_name} is None")
            raise ValidationError(f"{df_name} 不能为None")

        if not isinstance(df, pd.DataFrame):
            logger.error(f"{df_name} is not a DataFrame")
            raise ValidationError(f"{df_name} 必须是pandas.DataFrame类型")

        if df.empty:
            logger.warning(f"{df_name} is empty")

        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            logger.error(f"{df_name} 缺少必需列: {missing_cols}")
            raise ValidationError(f"{df_name} 缺少必需列: {missing_cols}")

        logger.debug(f"{df_name} 验证通过，包含列: {list(df.columns)}")
        return df

    @staticmethod
    def validate_numeric_columns(df: pd.DataFrame, numeric_columns: List[str],
                                  df_name: str = "DataFrame") -> pd.DataFrame:
        """
        验证DataFrame中的数值列是否为数值类型

        参数:
            df: 待验证的DataFrame
            numeric_columns: 必须为数值类型的列名列表
            df_name: DataFrame名称

        返回:
            验证通过的DataFrame

        抛出:
            ValidationError: 验证失败
        """
        logger = get_validators_logger()

        for col in numeric_columns:
            if col not in df.columns:
                continue

            if not pd.api.types.is_numeric_dtype(df[col]):
                logger.error(f"{df_name}.{col} 不是数值类型: {df[col].dtype}")
                raise ValidationError(f"{df_name}.{col} 必须是数值类型")

            # 检查是否包含无限值
            if np.isinf(df[col]).any():
                logger.warning(f"{df_name}.{col} 包含无限值，将被替换为NaN")
                df[col] = df[col].replace([np.inf, -np.inf], np.nan)

        logger.debug(f"数值列验证通过: {numeric_columns}")
        return df

    @staticmethod
    def validate_min_rows(df: pd.DataFrame, min_rows: int,
                         df_name: str = "DataFrame") -> pd.DataFrame:
        """
        验证DataFrame是否包含最小行数

        参数:
            df: 待验证的DataFrame
            min_rows: 最小行数
            df_name: DataFrame名称

        返回:
            验证通过的DataFrame

        抛出:
            ValidationError: 行数不足
        """
        logger = get_validators_logger()

        if len(df) < min_rows:
            logger.error(f"{df_name} 行数不足: {len(df)} < {min_rows}")
            raise ValidationError(f"{df_name} 至少需要 {min_rows} 行数据，当前只有 {len(df)} 行")

        logger.debug(f"{df_name} 行数验证通过: {len(df)} >= {min_rows}")
        return df

    @staticmethod
    def validate_no_duplicates(df: pd.DataFrame, columns: Optional[List[str]] = None,
                               df_name: str = "DataFrame") -> pd.DataFrame:
        """
        验证DataFrame是否存在重复行

        参数:
            df: 待验证的DataFrame
            columns: 用于检查重复的列，None则检查所有列
            df_name: DataFrame名称

        返回:
            去重后的DataFrame

        抛出:
            ValidationError: 发现重复行
        """
        logger = get_validators_logger()

        if columns:
            duplicates = df.duplicated(subset=columns)
        else:
            duplicates = df.duplicated()

        if duplicates.any():
            dup_count = duplicates.sum()
            logger.warning(f"{df_name} 发现 {dup_count} 行重复数据")

        return df.drop_duplicates(subset=columns)


class PriceValidator:
    """价格数据验证器"""

    @staticmethod
    def validate_ohlc(df: pd.DataFrame) -> pd.DataFrame:
        """
        验证OHLC数据的有效性

        参数:
            df: 包含open, high, low, close列的DataFrame

        返回:
            清理后的DataFrame

        抛出:
            ValidationError: 价格数据无效
        """
        logger = get_validators_logger()

        required_cols = ['open', 'high', 'low', 'close']
        df = DataFrameValidator.validate_dataframe(df, required_cols, "OHLC数据")

        # 检查负数价格（优先检查，不能修正负数）
        for col in ['open', 'high', 'low', 'close']:
            negative = df[col] < 0
            if negative.any():
                logger.error(f"发现 {negative.sum()} 行数据 {col} 为负数")
                raise ValidationError(f"{col} 不能为负数")

        # 验证high >= low
        invalid_high_low = df['high'] < df['low']
        if invalid_high_low.any():
            logger.warning(f"发现 {invalid_high_low.sum()} 行数据 high < low")
            # 修正: 交换high和low
            df.loc[invalid_high_low, ['high', 'low']] = df.loc[invalid_high_low, ['low', 'high']].values

        # 验证close在high和low之间
        invalid_close = (df['close'] > df['high']) | (df['close'] < df['low'])
        if invalid_close.any():
            logger.warning(f"发现 {invalid_close.sum()} 行数据 close不在high和low之间")
            # 修正: 限制close在high和low之间
            df.loc[invalid_close, 'close'] = df.loc[invalid_close, 'close'].clip(
                df.loc[invalid_close, 'low'],
                df.loc[invalid_close, 'high']
            )

        # 验证open在high和low之间
        invalid_open = (df['open'] > df['high']) | (df['open'] < df['low'])
        if invalid_open.any():
            logger.warning(f"发现 {invalid_open.sum()} 行数据 open不在high和low之间")
            # 修正: 限制open在high和low之间
            df.loc[invalid_open, 'open'] = df.loc[invalid_open, 'open'].clip(
                df.loc[invalid_open, 'low'],
                df.loc[invalid_open, 'high']
            )

        logger.debug("OHLC数据验证通过")
        return df

    @staticmethod
    def validate_positive_values(series: pd.Series, series_name: str = "Series",
                                  allow_zero: bool = True) -> pd.Series:
        """
        验证序列值是否为正数(或非负数)

        参数:
            series: 待验证的序列
            series_name: 序列名称
            allow_zero: 是否允许零值

        返回:
            验证通过的序列

        抛出:
            ValidationError: 包含负值
        """
        logger = get_validators_logger()

        if not allow_zero:
            invalid = series <= 0
            if invalid.any():
                logger.error(f"{series_name} 包含非正值: {(series <= 0).sum()}")
                raise ValidationError(f"{series_name} 必须为正数")
        else:
            invalid = series < 0
            if invalid.any():
                logger.error(f"{series_name} 包含负值: {(series < 0).sum()}")
                raise ValidationError(f"{series_name} 不能为负数")

        return series


class DateValidator:
    """日期验证器"""

    @staticmethod
    def validate_date_format(date_str: str, date_formats: Optional[List[str]] = None,
                            date_name: str = "日期") -> bool:
        """
        验证日期字符串格式

        参数:
            date_str: 日期字符串
            date_formats: 支持的日期格式列表，默认为["%Y%m%d"]
            date_name: 日期名称(用于错误提示)

        返回:
            是否有效

        抛出:
            ValidationError: 日期格式无效
        """
        logger = get_validators_logger()

        if date_formats is None:
            date_formats = ["%Y%m%d"]

        for fmt in date_formats:
            try:
                datetime.strptime(date_str, fmt)
                logger.debug(f"{date_name} {date_str} 格式验证通过: {fmt}")
                return True
            except ValueError:
                continue

        logger.error(f"{date_name} {date_str} 格式无效，支持的格式: {date_formats}")
        raise ValidationError(f"{date_name} {date_str} 格式无效，支持的格式: {date_formats}")

    @staticmethod
    def validate_date_range(start_date: str, end_date: str,
                           date_format: str = "%Y%m%d") -> bool:
        """
        验证日期范围是否有效

        参数:
            start_date: 开始日期
            end_date: 结束日期
            date_format: 日期格式

        返回:
            是否有效

        抛出:
            ValidationError: 日期范围无效
        """
        logger = get_validators_logger()

        try:
            start = datetime.strptime(start_date, date_format)
            end = datetime.strptime(end_date, date_format)

            if start > end:
                logger.error(f"开始日期 {start_date} 晚于结束日期 {end_date}")
                raise ValidationError(f"开始日期 {start_date} 不能晚于结束日期 {end_date}")

            logger.debug(f"日期范围验证通过: {start_date} ~ {end_date}")
            return True

        except ValueError as e:
            logger.error(f"日期格式错误: {e}")
            raise ValidationError(f"日期格式错误: {e}")

    @staticmethod
    def validate_trade_dates(dates: List[str], date_format: str = "%Y%m%d") -> List[str]:
        """
        验证交易日列表

        参数:
            dates: 交易日列表
            date_format: 日期格式

        返回:
            排序后的唯一交易日列表

        抛出:
            ValidationError: 日期列表无效
        """
        logger = get_validators_logger()

        if not dates:
            logger.warning("交易日列表为空")
            return []

        # 验证每个日期格式
        for date in dates:
            DateValidator.validate_date_format(date, [date_format], "交易日")

        # 去重并排序
        unique_dates = sorted(set(dates))

        logger.debug(f"交易日列表验证通过: {len(unique_dates)} 个交易日")
        return unique_dates


class ParameterValidator:
    """参数验证器"""

    @staticmethod
    def validate_positive_number(value: Any, param_name: str,
                                  allow_zero: bool = False) -> float:
        """
        验证参数是否为正数

        参数:
            value: 待验证的值
            param_name: 参数名称
            allow_zero: 是否允许零值

        返回:
            转换为float后的值

        抛出:
            ValidationError: 参数无效
        """
        logger = get_validators_logger()

        try:
            num_value = float(value)
        except (TypeError, ValueError):
            logger.error(f"{param_name} 无法转换为数值: {value}")
            raise ValidationError(f"{param_name} 必须是数值类型")

        if not allow_zero and num_value <= 0:
            logger.error(f"{param_name} 必须为正数: {value}")
            raise ValidationError(f"{param_name} 必须为正数")

        if allow_zero and num_value < 0:
            logger.error(f"{param_name} 不能为负数: {value}")
            raise ValidationError(f"{param_name} 不能为负数")

        logger.debug(f"{param_name} 验证通过: {num_value}")
        return num_value

    @staticmethod
    def validate_percentage(value: Any, param_name: str,
                           min_val: float = 0.0, max_val: float = 1.0) -> float:
        """
        验证参数是否为有效的百分比

        参数:
            value: 待验证的值
            param_name: 参数名称
            min_val: 最小值(默认0)
            max_val: 最大值(默认1，即100%)

        返回:
            转换为float后的值

        抛出:
            ValidationError: 参数无效
        """
        logger = get_validators_logger()

        try:
            num_value = float(value)
        except (TypeError, ValueError):
            logger.error(f"{param_name} 无法转换为数值: {value}")
            raise ValidationError(f"{param_name} 必须是数值类型")

        if not (min_val <= num_value <= max_val):
            logger.error(f"{param_name} 超出范围: {value} (应在 [{min_val}, {max_val}] 之间)")
            raise ValidationError(f"{param_name} 必须在 {min_val*100:.0f}% 到 {max_val*100:.0f}% 之间")

        logger.debug(f"{param_name} 验证通过: {num_value}")
        return num_value

    @staticmethod
    def validate_integer(value: Any, param_name: str,
                        min_val: Optional[int] = None,
                        max_val: Optional[int] = None) -> int:
        """
        验证参数是否为整数

        参数:
            value: 待验证的值
            param_name: 参数名称
            min_val: 最小值(可选)
            max_val: 最大值(可选)

        返回:
            转换为int后的值

        抛出:
            ValidationError: 参数无效
        """
        logger = get_validators_logger()

        try:
            int_value = int(float(value))  # 允许字符串形式的数字
        except (TypeError, ValueError):
            logger.error(f"{param_name} 无法转换为整数: {value}")
            raise ValidationError(f"{param_name} 必须是整数")

        if min_val is not None and int_value < min_val:
            logger.error(f"{param_name} 小于最小值: {value} < {min_val}")
            raise ValidationError(f"{param_name} 不能小于 {min_val}")

        if max_val is not None and int_value > max_val:
            logger.error(f"{param_name} 大于最大值: {value} > {max_val}")
            raise ValidationError(f"{param_name} 不能大于 {max_val}")

        logger.debug(f"{param_name} 验证通过: {int_value}")
        return int_value

    @staticmethod
    def validate_period(period: int, param_name: str = "周期",
                       min_period: int = 1, max_period: int = 1000) -> int:
        """
        验证技术指标周期参数

        参数:
            period: 周期值
            param_name: 参数名称
            min_period: 最小周期
            max_period: 最大周期

        返回:
            验证通过的周期值

        抛出:
            ValidationError: 周期无效
        """
        logger = get_validators_logger()

        period = ParameterValidator.validate_integer(period, param_name)

        if period < min_period:
            logger.error(f"{param_name} 小于最小值: {period} < {min_period}")
            raise ValidationError(f"{param_name} 不能小于 {min_period}")

        if period > max_period:
            logger.error(f"{param_name} 大于最大值: {period} > {max_period}")
            raise ValidationError(f"{param_name} 不能大于 {max_period}")

        logger.debug(f"{param_name} 验证通过: {period}")
        return period


class ConfigValidator:
    """配置验证器"""

    @staticmethod
    def validate_backtest_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """
        验证回测配置参数

        参数:
            config: 配置字典

        返回:
            验证后的配置

        抛出:
            ValidationError: 配置无效
        """
        logger = get_validators_logger()

        validated = {}

        # 验证初始资金
        validated['initial_capital'] = ParameterValidator.validate_positive_number(
            config.get('initial_capital', 100000),
            'initial_capital'
        )

        # 验证最大持仓数量
        validated['max_positions'] = ParameterValidator.validate_integer(
            config.get('max_positions', 5),
            'max_positions',
            min_val=1,
            max_val=50
        )

        # 验证单只股票仓位比例
        validated['position_size'] = ParameterValidator.validate_percentage(
            config.get('position_size', 0.2),
            'position_size',
            min_val=0.01,
            max_val=1.0
        )

        # 验证滑点
        validated['slippage'] = ParameterValidator.validate_percentage(
            config.get('slippage', 0.001),
            'slippage',
            max_val=0.05
        )

        # 验证手续费率
        validated['commission'] = ParameterValidator.validate_percentage(
            config.get('commission', 0.0003),
            'commission',
            max_val=0.01
        )

        # 验证止损比例(负数)
        stop_loss = config.get('stop_loss', -0.08)
        validated['stop_loss'] = ParameterValidator.validate_percentage(
            abs(stop_loss),
            'stop_loss',
            max_val=0.5
        )
        validated['stop_loss'] = -validated['stop_loss']

        # 验证止盈比例
        validated['take_profit'] = ParameterValidator.validate_percentage(
            config.get('take_profit', 0.2),
            'take_profit',
            max_val=2.0
        )

        # 验证最大持仓天数
        validated['max_holding_days'] = ParameterValidator.validate_integer(
            config.get('max_holding_days', 30),
            'max_holding_days',
            min_val=1,
            max_val=365
        )

        # 验证重新选股间隔
        validated['rebalance_days'] = ParameterValidator.validate_integer(
            config.get('rebalance_days', 5),
            'rebalance_days',
            min_val=1,
            max_val=100
        )

        # 验证日期范围
        start_date = config.get('start_date')
        end_date = config.get('end_date')
        if start_date and end_date:
            DateValidator.validate_date_range(start_date, end_date)
            validated['start_date'] = start_date
            validated['end_date'] = end_date

        logger.info(f"回测配置验证通过")
        return validated


class SafeCalculator:
    """安全计算器，防止除零等错误"""

    @staticmethod
    def safe_divide(numerator: Union[float, pd.Series],
                    denominator: Union[float, pd.Series],
                    default: float = np.nan) -> Union[float, pd.Series]:
        """
        安全除法，防止除零错误

        参数:
            numerator: 分子
            denominator: 分母
            default: 除零时的默认返回值

        返回:
            除法结果
        """
        logger = get_validators_logger()

        if isinstance(denominator, pd.Series):
            result = np.where(denominator != 0, numerator / denominator, default)
            zero_count = (denominator == 0).sum()
            if zero_count > 0:
                logger.warning(f"除零警告: 发现 {zero_count} 个零值")
            return pd.Series(result, index=numerator.index)
        else:
            if denominator == 0:
                logger.warning("除零警告: 分母为零")
                return default
            return numerator / denominator

    @staticmethod
    def safe_percentage_change(old_value: Union[float, pd.Series],
                                new_value: Union[float, pd.Series]) -> Union[float, pd.Series]:
        """
        安全计算百分比变化，防止除零

        参数:
            old_value: 原值
            new_value: 新值

        返回:
            百分比变化
        """
        logger = get_validators_logger()

        if isinstance(old_value, pd.Series):
            zero_count = (old_value == 0).sum()
            if zero_count > 0:
                logger.warning(f"计算百分比变化时发现 {zero_count} 个零值")

        return SafeCalculator.safe_divide(
            new_value - old_value,
            old_value,
            default=np.nan
        )

    @staticmethod
    def clip_value(value: Union[float, pd.Series],
                   min_val: float = None,
                   max_val: float = None) -> Union[float, pd.Series]:
        """
        限制数值在指定范围内

        参数:
            value: 待限制的值
            min_val: 最小值
            max_val: 最大值

        返回:
            限制后的值
        """
        if isinstance(value, pd.Series):
            result = value.copy()
            if min_val is not None:
                result = result.clip(lower=min_val)
            if max_val is not None:
                result = result.clip(upper=max_val)
            return result
        else:
            if min_val is not None:
                value = max(value, min_val)
            if max_val is not None:
                value = min(value, max_val)
            return value
