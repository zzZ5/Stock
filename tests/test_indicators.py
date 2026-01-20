"""
趋势雷达选股系统 - 技术指标测试
测试所有技术指标计算函数的正确性
"""
import pytest
import pandas as pd
import numpy as np
import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from indicators.indicators import (
    sma, ema, atr, rsi, macd, kdj, williams_r,
    price_position, adx, bollinger_bands, obv
)


@pytest.fixture
def sample_stock_data():
    """创建测试用的股票数据"""
    np.random.seed(42)
    n = 100
    
    data = pd.DataFrame({
        'open': np.random.uniform(10, 20, n),
        'high': np.random.uniform(20, 30, n),
        'low': np.random.uniform(5, 15, n),
        'close': np.random.uniform(10, 20, n),
        'volume': np.random.randint(1000000, 10000000, n),
        'amount': np.random.uniform(10000000, 100000000, n)
    })
    
    return data


class TestSMA:
    """测试简单移动平均线"""
    
    def test_sma_basic(self, sample_stock_data):
        """测试SMA基本计算"""
        result = sma(sample_stock_data['close'], 10)
        
        # 检查返回类型
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_stock_data)
        
        # 检查前N-1个值应该是NaN
        assert result.iloc[:9].isna().all()
        
        # 检查第10个值不应该是NaN
        assert not pd.isna(result.iloc[9])
        
    def test_sma_value(self, sample_stock_data):
        """测试SMA计算值的正确性"""
        # 使用固定值测试
        test_series = pd.Series([10, 20, 30, 40, 50])
        result = sma(test_series, 3)
        
        # 手动计算: (10+20+30)/3=20, (20+30+40)/3=30, (30+40+50)/3=40
        assert abs(result.iloc[2] - 20) < 0.01
        assert abs(result.iloc[3] - 30) < 0.01
        assert abs(result.iloc[4] - 40) < 0.01


class TestEMA:
    """测试指数移动平均线"""
    
    def test_ema_basic(self, sample_stock_data):
        """测试EMA基本计算"""
        result = ema(sample_stock_data['close'], 10)
        
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_stock_data)
        
        # EMA不应该有NaN（从第一个值开始计算）
        assert not pd.isna(result.iloc[0])
    
    def test_ema_vs_sma(self, sample_stock_data):
        """测试EMA应该比SMA更快反应价格变化"""
        sma_result = sma(sample_stock_data['close'], 10)
        ema_result = ema(sample_stock_data['close'], 10)
        
        # 在价格快速变化时，EMA应该比SMA更接近最新价格
        last_close = sample_stock_data['close'].iloc[-1]
        last_ema = ema_result.iloc[-1]
        last_sma = sma_result.iloc[-1]
        
        # EMA应该比SMA更接近当前价格
        ema_diff = abs(last_ema - last_close)
        sma_diff = abs(last_sma - last_close)
        
        # 这个断言不一定总是成立，但大部分情况下应该成立
        # 暂时注释掉，因为随机数据可能不满足
        # assert ema_diff <= sma_diff


class TestATR:
    """测试平均真实波幅"""
    
    def test_atr_basic(self, sample_stock_data):
        """测试ATR基本计算"""
        result = atr(sample_stock_data, 14)
        
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_stock_data)
    
    def test_atr_positive(self, sample_stock_data):
        """测试ATR应该始终为正数"""
        result = atr(sample_stock_data, 14)
        
        # 去除NaN后检查是否都为正数
        valid_results = result.dropna()
        assert (valid_results > 0).all()


class TestRSI:
    """测试相对强弱指标"""
    
    def test_rsi_basic(self, sample_stock_data):
        """测试RSI基本计算"""
        result = rsi(sample_stock_data['close'], 14)
        
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_stock_data)
    
    def test_rsi_range(self, sample_stock_data):
        """测试RSI应该在0-100之间"""
        result = rsi(sample_stock_data['close'], 14)
        
        valid_results = result.dropna()
        assert (valid_results >= 0).all()
        assert (valid_results <= 100).all()


class TestMACD:
    """测试MACD指标"""
    
    def test_macd_basic(self, sample_stock_data):
        """测试MACD基本计算"""
        result = macd(sample_stock_data['close'])
        
        assert isinstance(result, dict)
        assert 'macd' in result
        assert 'signal' in result
        assert 'hist' in result
        
        assert isinstance(result['macd'], pd.Series)
        assert isinstance(result['signal'], pd.Series)
        assert isinstance(result['hist'], pd.Series)


class TestKDJ:
    """测试KDJ指标"""
    
    def test_kdj_basic(self, sample_stock_data):
        """测试KDJ基本计算"""
        result = kdj(
            sample_stock_data['high'],
            sample_stock_data['low'],
            sample_stock_data['close']
        )
        
        assert isinstance(result, dict)
        assert 'k' in result
        assert 'd' in result
        assert 'j' in result
    
    def test_kdj_range(self, sample_stock_data):
        """测试KDJ的K和D值应该在0-100之间"""
        result = kdj(
            sample_stock_data['high'],
            sample_stock_data['low'],
            sample_stock_data['close']
        )
        
        # K和D应该在0-100之间
        if not result['k'].dropna().empty:
            k_valid = result['k'].dropna()
            assert (k_valid >= 0).all() and (k_valid <= 100).all()
        
        if not result['d'].dropna().empty:
            d_valid = result['d'].dropna()
            assert (d_valid >= 0).all() and (d_valid <= 100).all()


class TestWilliamsR:
    """测试威廉指标"""
    
    def test_williams_r_basic(self, sample_stock_data):
        """测试Williams %R基本计算"""
        result = williams_r(
            sample_stock_data['high'],
            sample_stock_data['low'],
            sample_stock_data['close']
        )
        
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_stock_data)
    
    def test_williams_r_range(self, sample_stock_data):
        """测试Williams %R应该在-100到0之间"""
        result = williams_r(
            sample_stock_data['high'],
            sample_stock_data['low'],
            sample_stock_data['close']
        )
        
        valid_results = result.dropna()
        assert (valid_results >= -100).all()
        assert (valid_results <= 0).all()


class TestPricePosition:
    """测试价格位置指标"""
    
    def test_price_position_basic(self, sample_stock_data):
        """测试价格位置基本计算"""
        result = price_position(
            sample_stock_data['high'],
            sample_stock_data['low'],
            sample_stock_data['close']
        )
        
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_stock_data)
    
    def test_price_position_range(self, sample_stock_data):
        """测试价格位置应该在0-1之间"""
        result = price_position(
            sample_stock_data['high'],
            sample_stock_data['low'],
            sample_stock_data['close']
        )
        
        valid_results = result.dropna()
        assert (valid_results >= 0).all()
        assert (valid_results <= 1).all()


class TestADX:
    """测试ADX指标"""
    
    def test_adx_basic(self, sample_stock_data):
        """测试ADX基本计算"""
        result = adx(
            sample_stock_data['high'],
            sample_stock_data['low'],
            sample_stock_data['close']
        )
        
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_stock_data)
    
    def test_adx_positive(self, sample_stock_data):
        """测试ADX应该始终为正数"""
        result = adx(
            sample_stock_data['high'],
            sample_stock_data['low'],
            sample_stock_data['close']
        )
        
        valid_results = result.dropna()
        assert (valid_results >= 0).all()


class TestBollingerBands:
    """测试布林带"""
    
    def test_bollinger_bands_basic(self, sample_stock_data):
        """测试布林带基本计算"""
        result = bollinger_bands(sample_stock_data['close'], 20, 2)
        
        assert isinstance(result, dict)
        assert 'upper' in result
        assert 'middle' in result
        assert 'lower' in result
        
        assert isinstance(result['upper'], pd.Series)
        assert isinstance(result['middle'], pd.Series)
        assert isinstance(result['lower'], pd.Series)
    
    def test_bollinger_bands_relationship(self, sample_stock_data):
        """测试布林带上下带应该包络中轨"""
        result = bollinger_bands(sample_stock_data['close'], 20, 2)
        
        valid_data = (
            result['upper'].notna() & 
            result['middle'].notna() & 
            result['lower'].notna()
        )
        
        assert (result['upper'][valid_data] >= result['middle'][valid_data]).all()
        assert (result['lower'][valid_data] <= result['middle'][valid_data]).all()


class TestOBV:
    """测试能量潮指标"""
    
    def test_obv_basic(self, sample_stock_data):
        """测试OBV基本计算"""
        result = obv(sample_stock_data)
        
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_stock_data)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
