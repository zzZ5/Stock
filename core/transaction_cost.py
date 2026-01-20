"""
交易成本模型
包含滑点、市场冲击、手续费等交易成本的高级建模
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass


@dataclass
class SlippageModel:
    """滑点模型"""
    model_type: str = 'fixed'  # fixed, linear, square_root, percentage
    base_slippage: float = 0.001  # 基础滑点 (0.1%)
    impact_factor: float = 0.0001  # 市场冲击系数
    volume_impact: bool = False  # 是否考虑成交量影响
    
    def calculate_slippage(
        self,
        price: float,
        volume: float,
        order_size: float,
        avg_daily_volume: float,
        is_buy: bool = True
    ) -> float:
        """
        计算滑点
        
        参数:
            price: 当前价格
            volume: 当日成交量
            order_size: 订单规模
            avg_daily_volume: 平均日成交量
            is_buy: 是否买入（买入向上滑，卖出向下滑）
        
        返回:
            滑点后的价格
        """
        if self.model_type == 'fixed':
            slippage = self.base_slippage
        
        elif self.model_type == 'percentage':
            # 基于订单规模占比
            participation_rate = min(order_size / avg_daily_volume, 0.1)
            slippage = self.base_slippage + self.impact_factor * participation_rate * 100
        
        elif self.model_type == 'linear':
            # 线性模型
            participation_rate = min(order_size / avg_daily_volume, 0.2)
            slippage = self.base_slippage + self.impact_factor * participation_rate * 50
        
        elif self.model_type == 'square_root':
            # 平方根模型（更接近真实情况）
            participation_rate = min(order_size / avg_daily_volume, 0.2)
            slippage = self.base_slippage + self.impact_factor * np.sqrt(participation_rate) * 10
        
        elif self.model_type == 'volume_impact' and self.volume_impact:
            # 考虑当日成交量影响
            participation_rate = min(order_size / volume, 0.3)
            slippage = self.base_slippage + self.impact_factor * participation_rate * 100
        else:
            slippage = self.base_slippage
        
        # 限制最大滑点
        slippage = min(slippage, 0.05)  # 最多5%
        
        # 买入加滑点，卖出减滑点
        multiplier = 1 + slippage if is_buy else 1 - slippage
        
        return price * multiplier


@dataclass
class MarketImpactModel:
    """市场冲击模型"""
    model_type: str = 'almgren_christoss'  # almgren_christoss, kyle, square_root
    volatility_window: int = 20
    avg_window: int = 60
    
    def calculate_impact(
        self,
        order_size: float,
        avg_daily_volume: float,
        avg_price: float,
        volatility: Optional[float] = None
    ) -> float:
        """
        计算市场冲击成本
        
        参数:
            order_size: 订单规模
            avg_daily_volume: 平均日成交量
            avg_price: 平均价格
            volatility: 波动率（可选）
        
        返回:
            市场冲击成本（比例）
        """
        participation_rate = min(order_size / avg_daily_volume, 0.2)
        
        if self.model_type == 'almgren_christoss':
            # Almgren-Chriss模型
            # Impact = gamma * (Q/V)^alpha
            # 其中Q/V是参与率，alpha通常在0.5-1之间
            gamma = 0.001  # 冲击系数
            alpha = 0.6    # 冲击指数
            impact = gamma * (participation_rate ** alpha)
        
        elif self.model_type == 'kyle':
            # Kyle模型
            # Impact = lambda * (Q/V)
            # lambda是信息不对称参数
            lambda_param = 0.005
            impact = lambda_param * participation_rate
        
        elif self.model_type == 'square_root':
            # 平方根模型
            impact = 0.001 * np.sqrt(participation_rate)
        
        else:
            impact = 0.001 * participation_rate
        
        # 考虑波动率影响
        if volatility:
            impact *= (1 + volatility)
        
        # 限制最大冲击
        impact = min(impact, 0.05)  # 最多5%
        
        return impact


@dataclass
class CommissionModel:
    """手续费模型"""
    commission_type: str = 'percentage'  # percentage, tiered, fixed
    base_rate: float = 0.0003  # 基础费率 (0.03%)
    min_commission: float = 5.0  # 最低手续费（元）
    max_commission: Optional[float] = None  # 最高手续费
    
    def calculate_commission(self, trade_value: float) -> float:
        """
        计算手续费
        
        参数:
            trade_value: 交易金额
        
        返回:
            手续费金额
        """
        if self.commission_type == 'percentage':
            commission = trade_value * self.base_rate
        
        elif self.commission_type == 'tiered':
            # 分级费率
            if trade_value < 10000:
                commission = trade_value * 0.0005
            elif trade_value < 100000:
                commission = trade_value * 0.0003
            else:
                commission = trade_value * 0.0002
        
        elif self.commission_type == 'fixed':
            commission = self.base_rate
        
        else:
            commission = trade_value * self.base_rate
        
        # 应用最低手续费
        commission = max(commission, self.min_commission)
        
        # 应用最高手续费（如果有）
        if self.max_commission:
            commission = min(commission, self.max_commission)
        
        return commission


class TransactionCostCalculator:
    """综合交易成本计算器"""
    
    def __init__(
        self,
        slippage_model: Optional[SlippageModel] = None,
        market_impact_model: Optional[MarketImpactModel] = None,
        commission_model: Optional[CommissionModel] = None
    ):
        """
        初始化交易成本计算器
        
        参数:
            slippage_model: 滑点模型
            market_impact_model: 市场冲击模型
            commission_model: 手续费模型
        """
        self.slippage_model = slippage_model or SlippageModel()
        self.market_impact_model = market_impact_model or MarketImpactModel()
        self.commission_model = commission_model or CommissionModel()
    
    def calculate_buy_cost(
        self,
        price: float,
        shares: int,
        volume: float,
        avg_daily_volume: float,
        avg_price: float,
        volatility: Optional[float] = None
    ) -> Dict:
        """
        计算买入总成本
        
        参数:
            price: 当前价格
            shares: 买入股数
            volume: 当日成交量
            avg_daily_volume: 平均日成交量
            avg_price: 平均价格
            volatility: 波动率
        
        返回:
            成本字典
        """
        order_size = price * shares
        
        # 计算滑点后价格
        price_with_slippage = self.slippage_model.calculate_slippage(
            price, volume, order_size, avg_daily_volume, is_buy=True
        )
        
        # 计算市场冲击
        impact = self.market_impact_model.calculate_impact(
            order_size, avg_daily_volume, avg_price, volatility
        )
        
        # 计算最终价格
        final_price = price_with_slippage * (1 + impact)
        
        # 计算交易金额
        trade_value = final_price * shares
        
        # 计算手续费
        commission = self.commission_model.calculate_commission(trade_value)
        
        # 总成本
        total_cost = trade_value + commission
        
        return {
            'original_price': price,
            'slippage_price': price_with_slippage,
            'impact_ratio': impact,
            'final_price': final_price,
            'trade_value': trade_value,
            'commission': commission,
            'total_cost': total_cost,
            'slippage_pct': (price_with_slippage - price) / price * 100,
            'impact_pct': impact * 100,
            'total_cost_pct': (total_cost - price * shares) / (price * shares) * 100
        }
    
    def calculate_sell_cost(
        self,
        price: float,
        shares: int,
        volume: float,
        avg_daily_volume: float,
        avg_price: float,
        volatility: Optional[float] = None
    ) -> Dict:
        """
        计算卖出净收益
        
        参数:
            price: 当前价格
            shares: 卖出股数
            volume: 当日成交量
            avg_daily_volume: 平均日成交量
            avg_price: 平均价格
            volatility: 波动率
        
        返回:
            收益字典
        """
        order_size = price * shares
        
        # 计算滑点后价格
        price_with_slippage = self.slippage_model.calculate_slippage(
            price, volume, order_size, avg_daily_volume, is_buy=False
        )
        
        # 计算市场冲击
        impact = self.market_impact_model.calculate_impact(
            order_size, avg_daily_volume, avg_price, volatility
        )
        
        # 计算最终价格
        final_price = price_with_slippage * (1 - impact)
        
        # 计算交易金额
        trade_value = final_price * shares
        
        # 计算手续费
        commission = self.commission_model.calculate_commission(trade_value)
        
        # 净收益
        net_proceeds = trade_value - commission
        
        return {
            'original_price': price,
            'slippage_price': price_with_slippage,
            'impact_ratio': impact,
            'final_price': final_price,
            'trade_value': trade_value,
            'commission': commission,
            'net_proceeds': net_proceeds,
            'slippage_pct': (price_with_slippage - price) / price * 100,
            'impact_pct': impact * 100,
            'total_cost_pct': (price * shares - net_proceeds) / (price * shares) * 100
        }
    
    def calculate_round_trip_cost(
        self,
        buy_price: float,
        sell_price: float,
        shares: int,
        buy_volume: float,
        sell_volume: float,
        avg_daily_volume: float,
        avg_price: float,
        volatility: Optional[float] = None
    ) -> Dict:
        """
        计算来回交易总成本
        
        参数:
            buy_price: 买入价格
            sell_price: 卖出价格
            shares: 股数
            buy_volume: 买入日成交量
            sell_volume: 卖出日成交量
            avg_daily_volume: 平均日成交量
            avg_price: 平均价格
            volatility: 波动率
        
        返回:
            往返成本字典
        """
        buy_cost = self.calculate_buy_cost(
            buy_price, shares, buy_volume, avg_daily_volume, avg_price, volatility
        )
        
        sell_cost = self.calculate_sell_cost(
            sell_price, shares, sell_volume, avg_daily_volume, avg_price, volatility
        )
        
        total_buy_cost = buy_cost['total_cost']
        total_sell_proceeds = sell_cost['net_proceeds']
        
        total_round_trip_cost = total_buy_cost - total_sell_proceeds
        gross_profit = sell_price * shares - buy_price * shares
        net_profit = total_sell_proceeds - total_buy_cost
        
        return {
            'buy_cost': buy_cost,
            'sell_cost': sell_cost,
            'total_buy_cost': total_buy_cost,
            'total_sell_proceeds': total_sell_proceeds,
            'gross_profit': gross_profit,
            'net_profit': net_profit,
            'total_cost_pct': total_round_trip_cost / (buy_price * shares) * 100,
            'cost_reduction_pct': (gross_profit - net_profit) / abs(gross_profit) * 100 if gross_profit != 0 else 0
        }
