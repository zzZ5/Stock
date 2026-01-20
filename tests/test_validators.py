"""
趋势雷达选股系统 - 数据验证模块测试
测试validators.py中的各个验证器
"""
import pytest
import pandas as pd
import numpy as np

from core.validators import (
    ValidationError,
    DataFrameValidator,
    PriceValidator,
    DateValidator,
    ParameterValidator,
    ConfigValidator,
    SafeCalculator
)


class TestDataFrameValidator:
    """测试DataFrame验证器"""

    def test_validate_dataframe_success(self):
        """测试成功的DataFrame验证"""
        df = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6]
        })
        result = DataFrameValidator.validate_dataframe(df, ['a', 'b'], "测试数据")
        assert result is not None
        assert len(result) == 3

    def test_validate_dataframe_missing_columns(self):
        """测试缺少列的情况"""
        df = pd.DataFrame({'a': [1, 2, 3]})
        with pytest.raises(ValidationError, match="缺少必需列"):
            DataFrameValidator.validate_dataframe(df, ['a', 'b'], "测试数据")

    def test_validate_dataframe_none(self):
        """测试None输入"""
        with pytest.raises(ValidationError, match="不能为None"):
            DataFrameValidator.validate_dataframe(None, ['a'], "测试数据")

    def test_validate_dataframe_not_dataframe(self):
        """测试非DataFrame输入"""
        with pytest.raises(ValidationError, match="必须是pandas.DataFrame类型"):
            DataFrameValidator.validate_dataframe([1, 2, 3], ['a'], "测试数据")

    def test_validate_numeric_columns(self):
        """测试数值列验证"""
        df = pd.DataFrame({
            'a': [1.0, 2.0, 3.0],
            'b': [4, 5, 6]
        })
        result = DataFrameValidator.validate_numeric_columns(df, ['a', 'b'], "测试数据")
        assert result is not None

    def test_validate_min_rows(self):
        """测试最小行数验证"""
        df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        result = DataFrameValidator.validate_min_rows(df, 5, "测试数据")
        assert len(result) == 5

    def test_validate_min_rows_fail(self):
        """测试行数不足"""
        df = pd.DataFrame({'a': [1, 2, 3]})
        with pytest.raises(ValidationError, match="至少需要"):
            DataFrameValidator.validate_min_rows(df, 5, "测试数据")


class TestPriceValidator:
    """测试价格验证器"""

    def test_validate_ohlc_success(self):
        """测试有效的OHLC数据"""
        df = pd.DataFrame({
            'open': [10, 11, 12],
            'high': [11, 12, 13],
            'low': [9, 10, 11],
            'close': [10.5, 11.5, 12.5]
        })
        result = PriceValidator.validate_ohlc(df)
        assert result is not None
        assert len(result) == 3

    def test_validate_ohlc_invalid_high_low(self):
        """测试high<low的情况(应被修正)"""
        df = pd.DataFrame({
            'open': [10],
            'high': [11],
            'low': [12],  # high < low
            'close': [11.5]
        })
        result = PriceValidator.validate_ohlc(df)
        assert result['high'].iloc[0] >= result['low'].iloc[0]

    def test_validate_ohlc_negative_price(self):
        """测试负数价格"""
        df = pd.DataFrame({
            'open': [-10],
            'high': [11],
            'low': [9],
            'close': [10.5]
        })
        with pytest.raises(ValidationError, match="不能为负数"):
            PriceValidator.validate_ohlc(df)

    def test_validate_positive_values(self):
        """测试正数值验证"""
        s = pd.Series([1.0, 2.0, 3.0])
        result = PriceValidator.validate_positive_values(s, "测试序列")
        assert result is not None

    def test_validate_positive_values_negative(self):
        """测试负值"""
        s = pd.Series([1.0, -2.0, 3.0])
        with pytest.raises(ValidationError, match="不能为负数"):
            PriceValidator.validate_positive_values(s, "测试序列", allow_zero=True)

    def test_validate_positive_values_zero_allowed(self):
        """测试允许零值"""
        s = pd.Series([0, 1.0, 2.0])
        result = PriceValidator.validate_positive_values(s, "测试序列", allow_zero=True)
        assert result is not None


class TestDateValidator:
    """测试日期验证器"""

    def test_validate_date_format_success(self):
        """测试有效的日期格式"""
        assert DateValidator.validate_date_format("20240120") == True

    def test_validate_date_format_fail(self):
        """测试无效的日期格式"""
        with pytest.raises(ValidationError, match="格式无效"):
            DateValidator.validate_date_format("2024-01-20", ["%Y%m%d"])

    def test_validate_date_range_success(self):
        """测试有效的日期范围"""
        assert DateValidator.validate_date_range("20240101", "20240120") == True

    def test_validate_date_range_invalid(self):
        """测试无效的日期范围(开始晚于结束)"""
        with pytest.raises(ValidationError, match="不能晚于"):
            DateValidator.validate_date_range("20240120", "20240101")

    def test_validate_trade_dates(self):
        """测试交易日列表验证"""
        dates = ["20240101", "20240102", "20240103"]
        result = DateValidator.validate_trade_dates(dates)
        assert len(result) == 3
        assert result == sorted(result)

    def test_validate_trade_dates_empty(self):
        """测试空列表"""
        result = DateValidator.validate_trade_dates([])
        assert result == []


class TestParameterValidator:
    """测试参数验证器"""

    def test_validate_positive_number_success(self):
        """测试有效的正数"""
        result = ParameterValidator.validate_positive_number(100, "测试参数")
        assert result == 100.0

    def test_validate_positive_number_negative(self):
        """测试负数"""
        with pytest.raises(ValidationError, match="必须为正数"):
            ParameterValidator.validate_positive_number(-10, "测试参数")

    def test_validate_positive_number_zero(self):
        """测试零值(默认不允许)"""
        with pytest.raises(ValidationError, match="必须为正数"):
            ParameterValidator.validate_positive_number(0, "测试参数")

    def test_validate_positive_number_zero_allowed(self):
        """测试允许零值"""
        result = ParameterValidator.validate_positive_number(0, "测试参数", allow_zero=True)
        assert result == 0.0

    def test_validate_percentage_success(self):
        """测试有效的百分比"""
        result = ParameterValidator.validate_percentage(0.5, "测试参数")
        assert result == 0.5

    def test_validate_percentage_out_of_range(self):
        """测试超出范围的百分比"""
        with pytest.raises(ValidationError, match="必须在"):
            ParameterValidator.validate_percentage(1.5, "测试参数")

    def test_validate_integer_success(self):
        """测试有效的整数"""
        result = ParameterValidator.validate_integer(10, "测试参数")
        assert result == 10

    def test_validate_integer_with_range(self):
        """测试带范围限制的整数"""
        result = ParameterValidator.validate_integer(5, "测试参数", min_val=1, max_val=10)
        assert result == 5

    def test_validate_integer_out_of_range(self):
        """测试超出范围的整数"""
        with pytest.raises(ValidationError, match="不能大于"):
            ParameterValidator.validate_integer(15, "测试参数", max_val=10)

    def test_validate_period_success(self):
        """测试有效的周期参数"""
        result = ParameterValidator.validate_period(20, "测试周期")
        assert result == 20

    def test_validate_period_too_small(self):
        """测试过小的周期"""
        with pytest.raises(ValidationError, match="不能小于"):
            ParameterValidator.validate_period(0, "测试周期")

    def test_validate_period_too_large(self):
        """测试过大的周期"""
        with pytest.raises(ValidationError, match="不能大于"):
            ParameterValidator.validate_period(1500, "测试周期")


class TestConfigValidator:
    """测试配置验证器"""

    def test_validate_backtest_config_success(self):
        """测试有效的回测配置"""
        config = {
            'initial_capital': 100000,
            'max_positions': 5,
            'position_size': 0.2,
            'slippage': 0.001,
            'commission': 0.0003,
            'stop_loss': -0.08,
            'take_profit': 0.2,
            'max_holding_days': 30,
            'rebalance_days': 5,
            'start_date': '20240101',
            'end_date': '20240120'
        }
        result = ConfigValidator.validate_backtest_config(config)
        assert 'initial_capital' in result
        assert result['initial_capital'] == 100000.0
        assert result['stop_loss'] < 0

    def test_validate_backtest_config_invalid_stop_loss(self):
        """测试无效的止损比例"""
        config = {
            'initial_capital': 100000,
            'max_positions': 5,
            'position_size': 0.2,
            'slippage': 0.001,
            'commission': 0.0003,
            'stop_loss': -0.8,  # 太大
            'take_profit': 0.2,
            'max_holding_days': 30,
            'rebalance_days': 5
        }
        with pytest.raises(ValidationError, match="不能大于"):
            ConfigValidator.validate_backtest_config(config)


class TestSafeCalculator:
    """测试安全计算器"""

    def test_safe_divide_success(self):
        """测试正常除法"""
        result = SafeCalculator.safe_divide(10, 2)
        assert result == 5.0

    def test_safe_divide_by_zero(self):
        """测试除零"""
        result = SafeCalculator.safe_divide(10, 0)
        assert pd.isna(result)

    def test_safe_divide_series_with_zero(self):
        """测试序列除法(包含零)"""
        numerator = pd.Series([10, 20, 30])
        denominator = pd.Series([2, 0, 3])
        result = SafeCalculator.safe_divide(numerator, denominator)
        assert result.iloc[0] == 5.0
        assert pd.isna(result.iloc[1])
        assert result.iloc[2] == 10.0

    def test_safe_divide_with_default(self):
        """测试带默认值的除法"""
        result = SafeCalculator.safe_divide(10, 0, default=999)
        assert result == 999

    def test_safe_percentage_change(self):
        """测试百分比变化计算"""
        result = SafeCalculator.safe_percentage_change(100, 110)
        assert result == 0.1

    def test_safe_percentage_change_zero_base(self):
        """测试基数为零的情况"""
        result = SafeCalculator.safe_percentage_change(0, 10)
        assert pd.isna(result)

    def test_clip_value(self):
        """测试数值限制"""
        result = SafeCalculator.clip_value(15, min_val=10, max_val=20)
        assert result == 15

    def test_clip_value_below_min(self):
        """测试低于最小值"""
        result = SafeCalculator.clip_value(5, min_val=10, max_val=20)
        assert result == 10

    def test_clip_value_above_max(self):
        """测试高于最大值"""
        result = SafeCalculator.clip_value(25, min_val=10, max_val=20)
        assert result == 20

    def test_clip_value_series(self):
        """测试序列限制"""
        s = pd.Series([5, 15, 25])
        result = SafeCalculator.clip_value(s, min_val=10, max_val=20)
        assert result.iloc[0] == 10
        assert result.iloc[1] == 15
        assert result.iloc[2] == 20


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
