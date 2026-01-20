"""
扩展技术指标测试
"""

import numpy as np
import pandas as pd
import pytest

from indicators.indicators_extended import (
    wma, dema, tema, hull_ma, supertrend, ichimoku,
    donchian_channels, pivot_points, vwap_close, aroon,
    acceleration_bands, envelope_sma, rsi_divergence,
    volume_weighted_ma, money_flow_ratio, ease_of_movement,
    mass_index, ultimate_oscillator, decycler,
    zigzag, linear_regression_slope, linear_regression_intercept,
    standardized_volume, volume_profile, squeeze_momentum
)
from indicators.indicators import sma, rsi, atr


class TestAdvancedMovingAverages:
    """高级移动平均线测试"""
    
    @pytest.fixture
    def sample_data(self):
        """创建示例数据"""
        return pd.Series([
            10, 12, 11, 13, 15, 14, 16, 18, 17, 19,
            21, 20, 22, 24, 23, 25, 27, 26, 28, 30
        ])
    
    def test_wma(self, sample_data):
        """测试加权移动平均"""
        result = wma(sample_data, 5)
        assert len(result) == len(sample_data)
        assert not result.isna().all()
    
    def test_dema(self, sample_data):
        """测试双指数移动平均"""
        result = dema(sample_data, 10)
        assert len(result) == len(sample_data)
        # DEMA应该比EMA响应更快
        ema_result = sample_data.ewm(span=10).mean()
        assert not result.isna().all()
    
    def test_tema(self, sample_data):
        """测试三重指数移动平均"""
        result = tema(sample_data, 10)
        assert len(result) == len(sample_data)
        assert not result.isna().all()
    
    def test_hull_ma(self, sample_data):
        """测试赫尔移动平均"""
        result = hull_ma(sample_data, 10)
        assert len(result) == len(sample_data)
        assert not result.isna().all()


class TestTrendIndicators:
    """趋势指标测试"""
    
    @pytest.fixture
    def sample_ohlcv(self):
        """创建示例OHLCV数据"""
        np.random.seed(42)
        n = 100
        return pd.DataFrame({
            'high': 100 + np.random.randn(n).cumsum() + 2,
            'low': 100 + np.random.randn(n).cumsum() - 2,
            'close': 100 + np.random.randn(n).cumsum(),
            'volume': np.random.randint(100000, 500000, n)
        })
    
    def test_supertrend(self, sample_ohlcv):
        """测试SuperTrend指标"""
        result = supertrend(
            sample_ohlcv['high'],
            sample_ohlcv['low'],
            sample_ohlcv['close'],
            period=10,
            multiplier=3.0
        )
        
        assert 'supertrend' in result
        assert 'trend' in result
        assert len(result['supertrend']) == len(sample_ohlcv)
        assert result['trend'].isin([1, 0, -1]).all()
    
    def test_ichimoku(self, sample_ohlcv):
        """测试一目均衡表"""
        result = ichimoku(
            sample_ohlcv['high'],
            sample_ohlcv['low'],
            sample_ohlcv['close']
        )
        
        assert 'tenkan' in result
        assert 'kijun' in result
        assert 'senkou_a' in result
        assert 'senkou_b' in result
        assert 'chikou' in result
    
    def test_donchian_channels(self, sample_ohlcv):
        """测试唐奇安通道"""
        result = donchian_channels(sample_ohlcv['high'], sample_ohlcv['low'], 20)
        
        assert 'upper' in result
        assert 'lower' in result
        assert 'middle' in result
        assert len(result['upper']) == len(sample_ohlcv)
        assert len(result['lower']) == len(sample_ohlcv)
    
    def test_pivot_points(self, sample_ohlcv):
        """测试枢轴点"""
        result = pivot_points(
            sample_ohlcv['high'],
            sample_ohlcv['low'],
            sample_ohlcv['close']
        )
        
        assert 'pivot' in result
        assert 'r1' in result
        assert 's1' in result
        assert 'r2' in result
        assert 's2' in result
        assert len(result['pivot']) == len(sample_ohlcv)
    
    def test_aroon(self, sample_ohlcv):
        """测试阿隆指标"""
        result = aroon(sample_ohlcv['high'], sample_ohlcv['low'], 25)
        
        assert 'aroon_up' in result
        assert 'aroon_down' in result
        assert 'oscillator' in result
        # 跳过NaN值
        assert result['aroon_up'].dropna().between(0, 100).all()
        assert result['aroon_down'].dropna().between(0, 100).all()


class TestEnvelopeIndicators:
    """包络线指标测试"""
    
    @pytest.fixture
    def sample_data(self):
        """创建示例数据"""
        np.random.seed(42)
        return pd.Series(100 + np.random.randn(100).cumsum())
    
    def test_acceleration_bands(self, sample_data):
        """测试加速带"""
        result = acceleration_bands(sample_data, 20)
        
        assert 'upper' in result
        assert 'middle' in result
        assert 'lower' in result
        # 跳过NaN值
        valid_mask = ~(result['upper'].isna() | result['middle'].isna() | result['lower'].isna())
        assert (result['upper'][valid_mask] >= result['middle'][valid_mask]).all()
        assert (result['lower'][valid_mask] <= result['middle'][valid_mask]).all()
    
    def test_envelope_sma(self, sample_data):
        """测试SMA包络线"""
        result = envelope_sma(sample_data, 20, 0.05)
        
        assert 'upper' in result
        assert 'middle' in result
        assert 'lower' in result
        # 跳过NaN值
        valid_mask = ~(result['upper'].isna() | result['middle'].isna() | result['lower'].isna())
        assert (result['upper'][valid_mask] >= result['middle'][valid_mask]).all()
        assert (result['lower'][valid_mask] <= result['middle'][valid_mask]).all()


class TestDivergenceIndicators:
    """背离指标测试"""
    
    @pytest.fixture
    def sample_data(self):
        """创建示例数据"""
        np.random.seed(42)
        close = pd.Series(100 + np.random.randn(100).cumsum())
        return close, rsi(close, 14)
    
    def test_rsi_divergence(self, sample_data):
        """测试RSI背离"""
        close, rsi_values = sample_data
        result = rsi_divergence(close, rsi_values, 20)
        
        assert len(result) == len(close)
        assert result.abs().max() <= 1  # 背离值应该是-1, 0, 1


class TestVolumeIndicators:
    """成交量指标测试"""
    
    @pytest.fixture
    def sample_data(self):
        """创建示例数据"""
        np.random.seed(42)
        n = 100
        return pd.DataFrame({
            'close': pd.Series(100 + np.random.randn(n).cumsum()),
            'volume': np.random.randint(100000, 500000, n)
        })
    
    def test_volume_weighted_ma(self, sample_data):
        """测试成交量加权移动平均"""
        result = volume_weighted_ma(sample_data['close'], sample_data['volume'], 20)
        
        assert len(result) == len(sample_data)
        assert not result.isna().all()
    
    def test_money_flow_ratio(self, sample_data):
        """测试资金流量比"""
        result = money_flow_ratio(sample_data['close'], sample_data['volume'], 14)
        
        assert len(result) == len(sample_data)
        assert not result.isna().all()
    
    def test_ease_of_movement(self, sample_data):
        """测试EMV指标"""
        sample_data['high'] = sample_data['close'] + 2
        sample_data['low'] = sample_data['close'] - 2
        
        result = ease_of_movement(
            sample_data['high'],
            sample_data['low'],
            sample_data['volume'],
            14
        )
        
        assert len(result) == len(sample_data)
        assert not result.isna().all()
    
    def test_standardized_volume(self, sample_data):
        """测试标准化成交量"""
        result = standardized_volume(sample_data['volume'], 20)
        
        assert len(result) == len(sample_data)
        # Z-score应该在合理范围内
        assert result.abs().max() < 10  # 95%置信度
    
    def test_volume_profile(self, sample_data):
        """测试成交量分布"""
        result = volume_profile(sample_data['close'], sample_data['volume'], 50)
        
        assert 'vwap' in result
        assert 'poc' in result
        assert not result['vwap'].isna().all()
        assert not result['poc'].isna().all()


class TestMomentumIndicators:
    """动量指标测试"""
    
    @pytest.fixture
    def sample_ohlcv(self):
        """创建示例OHLCV数据"""
        np.random.seed(42)
        n = 100
        return pd.DataFrame({
            'high': 100 + np.random.randn(n).cumsum() + 2,
            'low': 100 + np.random.randn(n).cumsum() - 2,
            'close': 100 + np.random.randn(n).cumsum(),
            'volume': np.random.randint(100000, 500000, n)
        })
    
    def test_mass_index(self, sample_ohlcv):
        """测试Mass Index"""
        result = mass_index(sample_ohlcv['high'], sample_ohlcv['low'], 25, 9)
        
        assert len(result) == len(sample_ohlcv)
        assert not result.isna().all()
    
    def test_ultimate_oscillator(self, sample_ohlcv):
        """测试终极震荡指标"""
        result = ultimate_oscillator(
            sample_ohlcv['high'],
            sample_ohlcv['low'],
            sample_ohlcv['close']
        )
        
        assert len(result) == len(sample_ohlcv)
        assert not result.isna().all()
    
    def test_decycler(self, sample_ohlcv):
        """测试去周期指标"""
        result = decycler(sample_ohlcv['close'], 125)
        
        assert len(result) == len(sample_ohlcv)
        assert not result.isna().all()


class TestPatternIndicators:
    """形态指标测试"""
    
    @pytest.fixture
    def sample_data(self):
        """创建示例数据"""
        np.random.seed(42)
        return pd.Series(100 + np.random.randn(100).cumsum())
    
    def test_zigzag(self, sample_data):
        """测试ZigZag指标"""
        result = zigzag(sample_data, 0.05)
        
        assert 'zigzag' in result
        assert 'trend' in result
        assert len(result['zigzag']) == len(sample_data)
        # trend可能包含NaN或非整数，检查有效值
        valid_trend = result['trend'].dropna()
        if len(valid_trend) > 0:
            assert valid_trend.isin([1, 0, -1]).all()


class TestRegressionIndicators:
    """回归指标测试"""
    
    @pytest.fixture
    def sample_data(self):
        """创建示例数据"""
        np.random.seed(42)
        return pd.Series(100 + np.random.randn(100).cumsum())
    
    def test_linear_regression_slope(self, sample_data):
        """测试线性回归斜率"""
        result = linear_regression_slope(sample_data, 20)
        
        assert len(result) == len(sample_data)
        assert not result.isna().all()
    
    def test_linear_regression_intercept(self, sample_data):
        """测试线性回归截距"""
        result = linear_regression_intercept(sample_data, 20)
        
        assert len(result) == len(sample_data)
        assert not result.isna().all()


class TestCompositeIndicators:
    """组合指标测试"""
    
    @pytest.fixture
    def sample_ohlcv(self):
        """创建示例OHLCV数据"""
        np.random.seed(42)
        n = 100
        return pd.DataFrame({
            'high': 100 + np.random.randn(n).cumsum() + 2,
            'low': 100 + np.random.randn(n).cumsum() - 2,
            'close': 100 + np.random.randn(n).cumsum(),
            'volume': np.random.randint(100000, 500000, n)
        })
    
    def test_squeeze_momentum(self, sample_ohlcv):
        """测试挤压动量指标"""
        result = squeeze_momentum(sample_ohlcv)
        
        assert 'squeeze' in result
        assert 'momentum' in result
        assert len(result['squeeze']) == len(sample_ohlcv)
        assert len(result['momentum']) == len(sample_ohlcv)
        assert result['squeeze'].isin([0, 1]).all()


class TestEdgeCases:
    """边界情况测试"""
    
    def test_short_data(self):
        """测试短数据"""
        short_data = pd.Series([10, 12, 11, 13])
        
        # 这些函数应该能处理短数据
        result_wma = wma(short_data, 2)
        result_dema = dema(short_data, 2)
        result_tema = tema(short_data, 2)
        
        assert len(result_wma) == len(short_data)
        assert len(result_dema) == len(short_data)
        assert len(result_tema) == len(short_data)
    
    def test_constant_data(self):
        """测试常量数据"""
        constant_data = pd.Series([100] * 50)
        
        result = wma(constant_data, 10)
        # 常量数据的结果应该接近常量
        assert abs(result.iloc[-1] - 100) < 1
    
    def test_nan_handling(self):
        """测试NaN值处理"""
        data_with_nan = pd.Series([10, np.nan, 12, 11, np.nan, 13])
        
        result = wma(data_with_nan, 3)
        # 结果应该处理NaN
        assert len(result) == len(data_with_nan)
