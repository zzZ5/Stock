"""
交易成本模型测试
"""
import pytest
import pandas as pd
import numpy as np

from core.transaction_cost import (
    SlippageModel,
    MarketImpactModel,
    CommissionModel,
    TransactionCostCalculator
)


class TestSlippageModel:
    """滑点模型测试"""
    
    def test_fixed_slippage(self):
        """测试固定滑点模型"""
        model = SlippageModel(model_type='fixed', base_slippage=0.001)
        
        price = 100.0
        buy_price = model.calculate_slippage(
            price=price,
            volume=1000000,
            order_size=10000,
            avg_daily_volume=1000000,
            is_buy=True
        )
        sell_price = model.calculate_slippage(
            price=price,
            volume=1000000,
            order_size=10000,
            avg_daily_volume=1000000,
            is_buy=False
        )
        
        # 买入应该有滑点，卖出也应该有滑点
        assert buy_price > price
        assert sell_price < price
        assert abs((buy_price - price) / price - 0.001) < 0.0001
        assert abs((sell_price - price) / price + 0.001) < 0.0001
    
    def test_percentage_slippage(self):
        """测试百分比滑点模型"""
        model = SlippageModel(
            model_type='percentage',
            base_slippage=0.001,
            impact_factor=0.0001
        )
        
        # 大订单应该有更大滑点
        price = 100.0
        small_order_price = model.calculate_slippage(
            price=price,
            volume=1000000,
            order_size=1000,  # 小订单
            avg_daily_volume=1000000,
            is_buy=True
        )
        large_order_price = model.calculate_slippage(
            price=price,
            volume=1000000,
            order_size=100000,  # 大订单
            avg_daily_volume=1000000,
            is_buy=True
        )
        
        assert large_order_price > small_order_price
    
    def test_square_root_slippage(self):
        """测试平方根滑点模型"""
        model = SlippageModel(
            model_type='square_root',
            base_slippage=0.001,
            impact_factor=0.0001
        )
        
        price = 100.0
        buy_price = model.calculate_slippage(
            price=price,
            volume=1000000,
            order_size=10000,
            avg_daily_volume=1000000,
            is_buy=True
        )
        
        assert buy_price > price
        assert buy_price < price * 1.05  # 最大5%限制
    
    def test_max_slippage_limit(self):
        """测试最大滑点限制"""
        model = SlippageModel(
            model_type='percentage',
            base_slippage=0.001,
            impact_factor=1.0  # 很大的影响因子
        )
        
        price = 100.0
        buy_price = model.calculate_slippage(
            price=price,
            volume=1000000,
            order_size=100000000,  # 超大订单
            avg_daily_volume=1000000,
            is_buy=True
        )
        
        # 应该限制在最大5%滑点
        assert buy_price <= price * 1.05


class TestMarketImpactModel:
    """市场冲击模型测试"""
    
    def test_almgren_christoss_impact(self):
        """测试Almgren-Chriss模型"""
        model = MarketImpactModel(model_type='almgren_christoss')
        
        impact = model.calculate_impact(
            order_size=10000,
            avg_daily_volume=1000000,
            avg_price=100.0
        )
        
        assert impact > 0
        assert impact < 0.05  # 最大5%限制
    
    def test_kyle_impact(self):
        """测试Kyle模型"""
        model = MarketImpactModel(model_type='kyle')
        
        impact = model.calculate_impact(
            order_size=10000,
            avg_daily_volume=1000000,
            avg_price=100.0
        )
        
        assert impact > 0
    
    def test_square_root_impact(self):
        """测试平方根模型"""
        model = MarketImpactModel(model_type='square_root')
        
        impact = model.calculate_impact(
            order_size=10000,
            avg_daily_volume=1000000,
            avg_price=100.0
        )
        
        assert impact > 0
    
    def test_volatility_effect(self):
        """测试波动率对冲击的影响"""
        model = MarketImpactModel(model_type='almgren_christoss')
        
        impact_low_vol = model.calculate_impact(
            order_size=10000,
            avg_daily_volume=1000000,
            avg_price=100.0,
            volatility=0.01  # 1%波动率
        )
        impact_high_vol = model.calculate_impact(
            order_size=10000,
            avg_daily_volume=1000000,
            avg_price=100.0,
            volatility=0.03  # 3%波动率
        )
        
        # 高波动率应该有更大的冲击
        assert impact_high_vol > impact_low_vol
    
    def test_participation_rate_effect(self):
        """测试参与率对冲击的影响"""
        model = MarketImpactModel(model_type='almgren_christoss')
        
        impact_small = model.calculate_impact(
            order_size=1000,
            avg_daily_volume=1000000,
            avg_price=100.0
        )
        impact_large = model.calculate_impact(
            order_size=100000,
            avg_daily_volume=1000000,
            avg_price=100.0
        )
        
        # 大订单应该有更大的冲击
        assert impact_large > impact_small


class TestCommissionModel:
    """手续费模型测试"""
    
    def test_percentage_commission(self):
        """测试百分比手续费"""
        model = CommissionModel(
            commission_type='percentage',
            base_rate=0.0003,
            min_commission=5.0
        )
        
        commission = model.calculate_commission(trade_value=10000)
        expected = 10000 * 0.0003
        
        assert commission >= 5.0  # 最低手续费
        assert abs(commission - expected) < 0.01 or commission == 5.0
    
    def test_min_commission(self):
        """测试最低手续费"""
        model = CommissionModel(
            commission_type='percentage',
            base_rate=0.0003,
            min_commission=5.0
        )
        
        # 小额交易应该收最低手续费
        commission = model.calculate_commission(trade_value=1000)
        assert commission == 5.0
    
    def test_tiered_commission(self):
        """测试分级手续费"""
        model = CommissionModel(
            commission_type='tiered',
            min_commission=5.0
        )
        
        # 小额
        commission_small = model.calculate_commission(trade_value=5000)
        # 中额
        commission_medium = model.calculate_commission(trade_value=50000)
        # 大额
        commission_large = model.calculate_commission(trade_value=200000)
        
        # 大额交易的费率应该更低
        rate_small = commission_small / 5000
        rate_large = commission_large / 200000
        
        assert rate_large < rate_small
    
    def test_max_commission(self):
        """测试最高手续费"""
        model = CommissionModel(
            commission_type='percentage',
            base_rate=0.0003,
            min_commission=5.0,
            max_commission=100.0
        )
        
        # 超大额交易应该受最高手续费限制
        commission = model.calculate_commission(trade_value=10000000)
        assert commission <= 100.0


class TestTransactionCostCalculator:
    """综合交易成本计算器测试"""
    
    def test_calculate_buy_cost(self):
        """测试买入成本计算"""
        calculator = TransactionCostCalculator()
        
        result = calculator.calculate_buy_cost(
            price=100.0,
            shares=1000,
            volume=1000000,
            avg_daily_volume=1000000,
            avg_price=100.0
        )
        
        assert 'original_price' in result
        assert 'final_price' in result
        assert 'total_cost' in result
        assert result['total_cost'] > 1000 * 100
        assert result['final_price'] > result['original_price']
    
    def test_calculate_sell_cost(self):
        """测试卖出成本计算"""
        calculator = TransactionCostCalculator()
        
        result = calculator.calculate_sell_cost(
            price=100.0,
            shares=1000,
            volume=1000000,
            avg_daily_volume=1000000,
            avg_price=100.0
        )
        
        assert 'original_price' in result
        assert 'final_price' in result
        assert 'net_proceeds' in result
        assert result['net_proceeds'] < 1000 * 100
        assert result['final_price'] < result['original_price']
    
    def test_round_trip_cost(self):
        """测试往返交易成本"""
        calculator = TransactionCostCalculator()
        
        result = calculator.calculate_round_trip_cost(
            buy_price=100.0,
            sell_price=110.0,
            shares=1000,
            buy_volume=1000000,
            sell_volume=1000000,
            avg_daily_volume=1000000,
            avg_price=105.0
        )
        
        assert 'gross_profit' in result
        assert 'net_profit' in result
        assert 'total_cost_pct' in result
        
        # 净利润应该小于毛利润
        assert result['net_profit'] < result['gross_profit']
        
        # 毛利润应该是正的（卖出价高于买入价）
        assert result['gross_profit'] > 0
    
    def test_cost_breakdown(self):
        """测试成本明细"""
        calculator = TransactionCostCalculator()
        
        result = calculator.calculate_buy_cost(
            price=100.0,
            shares=1000,
            volume=1000000,
            avg_daily_volume=1000000,
            avg_price=100.0
        )
        
        # 检查所有成本组成部分
        assert 'slippage_pct' in result
        assert 'impact_pct' in result
        assert 'total_cost_pct' in result
        
        # 总成本应该由滑点、冲击和手续费组成
        # 这里不做精确检查，因为不同模型组合会有不同结果
        assert result['total_cost_pct'] >= 0
    
    def test_large_order_cost(self):
        """测试大额订单成本"""
        calculator = TransactionCostCalculator()
        
        small_order = calculator.calculate_buy_cost(
            price=100.0,
            shares=100,
            volume=1000000,
            avg_daily_volume=1000000,
            avg_price=100.0
        )
        
        large_order = calculator.calculate_buy_cost(
            price=100.0,
            shares=10000,
            volume=1000000,
            avg_daily_volume=1000000,
            avg_price=100.0
        )
        
        # 大额订单的成本比例应该更高
        assert large_order['total_cost_pct'] > small_order['total_cost_pct']
