"""
多维度因子体系测试
"""
import pytest
import pandas as pd
import numpy as np
from core.factors import (
    FundamentalFactors, TechnicalFactors, MoneyFlowFactors,
    SentimentFactors, FactorCalculator,
    FactorScore, get_default_factor_weights, get_default_factor_directions
)


class TestFundamentalFactors:
    """基本面因子测试"""

    def test_calculate_pe_ratio(self):
        """测试市盈率计算"""
        fundamental = FundamentalFactors()
        pe = fundamental.calculate_pe_ratio(100, 10)

        assert pe == 10.0

    def test_calculate_pe_ratio_zero_eps(self):
        """测试EPS为零时的市盈率"""
        fundamental = FundamentalFactors()
        pe = fundamental.calculate_pe_ratio(100, 0)

        assert pd.isna(pe)

    def test_calculate_pb_ratio(self):
        """测试市净率计算"""
        fundamental = FundamentalFactors()
        pb = fundamental.calculate_pb_ratio(100, 20)

        assert pb == 5.0

    def test_calculate_roe(self):
        """测试ROE计算"""
        fundamental = FundamentalFactors()
        roe = fundamental.calculate_roe(10, 100)

        assert roe == 10.0

    def test_calculate_debt_ratio(self):
        """测试资产负债率计算"""
        fundamental = FundamentalFactors()
        debt_ratio = fundamental.calculate_debt_ratio(60, 100)

        assert debt_ratio == 60.0

    def test_calculate_growth_rate(self):
        """测试增长率计算"""
        fundamental = FundamentalFactors()
        growth = fundamental.calculate_growth_rate(120, 100)

        assert growth == 20.0


class TestTechnicalFactors:
    """技术面因子测试"""

    @pytest.fixture
    def price_df(self):
        """创建测试用价格数据"""
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=100)

        # 生成上升趋势数据
        trend = np.linspace(100, 150, 100)
        noise = np.random.randn(100) * 5

        return pd.DataFrame({
            'close': trend + noise,
            'high': trend + noise + 5,
            'low': trend + noise - 5,
            'open': trend + noise - 2,
            'volume': np.random.randint(100000, 500000, 100)
        }, index=dates)

    def test_trend_strength(self, price_df):
        """测试趋势强度因子"""
        tech = TechnicalFactors()
        score = tech.trend_strength(price_df, period=20)

        assert not pd.isna(score)
        assert 0 <= score <= 100

    def test_momentum_factor(self, price_df):
        """测试动量因子"""
        tech = TechnicalFactors()
        score = tech.momentum_factor(price_df, period=20)

        assert not pd.isna(score)
        assert 0 <= score <= 100

    def test_volatility_factor(self, price_df):
        """测试波动率因子"""
        tech = TechnicalFactors()
        score = tech.volatility_factor(price_df, period=20)

        assert not pd.isna(score)
        assert 0 <= score <= 100

    def test_reversal_factor(self, price_df):
        """测试反转因子"""
        tech = TechnicalFactors()
        score = tech.reversal_factor(price_df)

        assert not pd.isna(score)
        assert 0 <= score <= 100


class TestMoneyFlowFactors:
    """资金面因子测试"""

    @pytest.fixture
    def money_flow_df(self):
        """创建测试用资金流数据"""
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=100)

        return pd.DataFrame({
            'close': np.cumsum(np.random.randn(100)) + 100,
            'high': np.cumsum(np.random.randn(100)) + 105,
            'low': np.cumsum(np.random.randn(100)) + 95,
            'open': np.cumsum(np.random.randn(100)) + 98,
            'volume': np.random.randint(100000, 500000, 100),
            'amount': np.random.randint(1000000, 5000000, 100)
        }, index=dates)

    def test_volume_factor(self, money_flow_df):
        """测试成交量因子"""
        mf = MoneyFlowFactors()
        score = mf.volume_factor(money_flow_df, period=20)

        assert not pd.isna(score)
        assert 0 <= score <= 100

    def test_capital_flow(self, money_flow_df):
        """测试资金流向因子"""
        mf = MoneyFlowFactors()
        score = mf.capital_flow(money_flow_df)

        assert not pd.isna(score)
        assert 0 <= score <= 100

    def test_turnover_factor(self, money_flow_df):
        """测试换手率因子"""
        mf = MoneyFlowFactors()
        score = mf.turnover_factor(money_flow_df, market_cap=100000000)  # 10亿

        assert not pd.isna(score)
        assert 0 <= score <= 100


class TestSentimentFactors:
    """情绪面因子测试"""

    @pytest.fixture
    def sentiment_df(self):
        """创建测试用情绪数据"""
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=100)

        return pd.DataFrame({
            'close': np.cumsum(np.random.randn(100)) + 100,
            'high': np.cumsum(np.random.randn(100)) + 105,
            'low': np.cumsum(np.random.randn(100)) + 95,
            'open': np.cumsum(np.random.randn(100)) + 98
        }, index=dates)

    def test_market_sentiment(self, sentiment_df):
        """测试市场情绪因子"""
        sentiment = SentimentFactors()
        score = sentiment.market_sentiment(sentiment_df)

        assert not pd.isna(score)
        assert 0 <= score <= 100

    def test_market_sentiment_with_index(self, sentiment_df):
        """测试带指数的市场情绪因子"""
        sentiment = SentimentFactors()

        index_df = pd.DataFrame({
            'close': np.cumsum(np.random.randn(100)) + 3000
        }, index=pd.date_range('2024-01-01', periods=100))

        score = sentiment.market_sentiment(sentiment_df, index_df)

        assert not pd.isna(score)
        assert 0 <= score <= 100

    def test_institutional_sentiment(self, sentiment_df):
        """测试机构情绪因子"""
        sentiment = SentimentFactors()
        score = sentiment.institutional_sentiment(sentiment_df)

        assert not pd.isna(score)
        assert 0 <= score <= 100


class TestFactorCalculator:
    """因子计算器测试"""

    @pytest.fixture
    def sample_df(self):
        """创建测试用数据"""
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=100)

        return pd.DataFrame({
            'close': np.cumsum(np.random.randn(100)) + 100,
            'high': np.cumsum(np.random.randn(100)) + 105,
            'low': np.cumsum(np.random.randn(100)) + 95,
            'open': np.cumsum(np.random.randn(100)) + 98,
            'volume': np.random.randint(100000, 500000, 100),
            'amount': np.random.randint(1000000, 5000000, 100)
        }, index=dates)

    def test_calculate_all_factors(self, sample_df):
        """测试计算所有因子"""
        calculator = FactorCalculator()
        factors = calculator.calculate_all_factors(sample_df)

        assert isinstance(factors, dict)
        assert len(factors) > 0

    def test_calculate_all_factors_with_fundamental(self, sample_df):
        """测试计算包含基本面的所有因子"""
        calculator = FactorCalculator()

        fundamental_data = {
            'eps': 5,
            'book_value': 20,
            'sales': 30,
            'net_income': 100,
            'equity': 1000,
            'total_assets': 2000,
            'total_liabilities': 600
        }

        factors = calculator.calculate_all_factors(sample_df, fundamental_data)

        assert isinstance(factors, dict)
        assert len(factors) > 0
        assert 'pe_ratio' in factors or 'roe' in factors

    def test_normalize_factors(self, sample_df):
        """测试因子归一化"""
        calculator = FactorCalculator()
        factors = {
            'trend_strength': 75.0,
            'momentum': 60.0,
            'volatility': 45.0
        }

        directions = {
            'trend_strength': 'positive',
            'momentum': 'positive',
            'volatility': 'positive'
        }

        normalized = calculator.normalize_factors(factors, directions)

        assert len(normalized) == 3
        assert all(isinstance(score, FactorScore) for score in normalized.values())

    def test_calculate_composite_score(self, sample_df):
        """测试计算综合得分"""
        calculator = FactorCalculator()

        factor_scores = {
            'trend_strength': FactorScore('trend_strength', 75.0, 75.0, 'positive'),
            'momentum': FactorScore('momentum', 60.0, 60.0, 'positive'),
            'volatility': FactorScore('volatility', 45.0, 45.0, 'positive')
        }

        weights = {
            'trend_strength': 0.4,
            'momentum': 0.4,
            'volatility': 0.2
        }

        composite = calculator.calculate_composite_score(factor_scores, weights)

        assert isinstance(composite, float)
        assert 0 <= composite <= 100

    def test_calculate_composite_score_with_negative_factor(self, sample_df):
        """测试包含负向因子的综合得分"""
        calculator = FactorCalculator()

        factor_scores = {
            'trend_strength': FactorScore('trend_strength', 75.0, 75.0, 'positive'),
            'pe_ratio': FactorScore('pe_ratio', 80.0, 80.0, 'negative')
        }

        weights = {
            'trend_strength': 0.7,
            'pe_ratio': 0.3
        }

        composite = calculator.calculate_composite_score(factor_scores, weights)

        # PE是负向因子，所以综合得分应该低于只考虑trend的情况
        assert isinstance(composite, float)
        assert composite < 75.0


class TestDefaultConfigurations:
    """默认配置测试"""

    def test_get_default_factor_weights(self):
        """测试获取默认因子权重"""
        weights = get_default_factor_weights()

        assert isinstance(weights, dict)
        assert len(weights) > 0

        # 检查权重总和接近1（可能会有一些因子不存在）
        total_weight = sum(weights.values())
        assert total_weight > 0.9 and total_weight <= 1.0

    def test_get_default_factor_directions(self):
        """测试获取默认因子方向"""
        directions = get_default_factor_directions()

        assert isinstance(directions, dict)
        assert len(directions) > 0
        assert all(d in ['positive', 'negative'] for d in directions.values())


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
