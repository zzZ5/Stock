"""
风险控制模块
提供动态止损、仓位管理、相关性控制等功能
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy.stats import norm
from core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PositionRisk:
    """持仓风险"""
    ts_code: str
    entry_price: float
    current_price: float
    position_size: float  # 仓位大小（金额）
    stop_loss_price: float
    take_profit_price: float
    atr_stop_price: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    risk_level: str  # 'low', 'medium', 'high', 'extreme'


@dataclass
class PortfolioRisk:
    """组合风险"""
    total_value: float
    total_unrealized_pnl: float
    total_unrealized_pnl_pct: float
    max_position_risk: float
    portfolio_volatility: float
    value_at_risk_95: float  # 95%置信度的VaR
    value_at_risk_99: float  # 99%置信度的VaR
    concentration_risk: float  # 集中度风险
    correlation_risk: float  # 相关性风险


class DynamicStopLoss:
    """动态止损策略"""

    def __init__(self, method: str = 'atr_trailing'):
        """
        初始化动态止损

        参数:
            method: 止损方法 ('atr_trailing', 'fixed', 'parabolic_sar', 'chandelier')
        """
        self.method = method

    def calculate_stop_loss(self, df: pd.DataFrame,
                             entry_price: float,
                             atr_period: int = 14,
                             atr_multiplier: float = 2.5,
                             fixed_stop_pct: float = 0.10) -> float:
        """
        计算止损价格

        参数:
            df: 价格数据
            entry_price: 入场价格
            atr_period: ATR周期
            atr_multiplier: ATR倍数
            fixed_stop_pct: 固定止损百分比

        返回:
            止损价格
        """
        current_price = df['close'].iloc[-1]

        if self.method == 'fixed':
            stop_loss = entry_price * (1 - fixed_stop_pct)

        elif self.method == 'atr_trailing':
            from indicators.indicators import atr
            atr_value = atr(df['high'].values, df['low'].values, df['close'].values, atr_period)
            current_atr = atr_value[-1] if len(atr_value) > 0 else current_price * 0.02

            stop_loss = current_price - current_atr * atr_multiplier

        elif self.method == 'parabolic_sar':
            from indicators.indicators import parabolic_sar
            psar = parabolic_sar(df['high'].values, df['low'].values, df['close'].values)
            stop_loss = psar[-1] if len(psar) > 0 else current_price * 0.95

        elif self.method == 'chandelier':
            from indicators.indicators_extended import chandelier_exit
            chand_exit = chandelier_exit(df['high'].values, df['low'].values, df['close'].values)
            stop_loss = chand_exit[-1] if len(chand_exit) > 0 else current_price * 0.95

        else:
            stop_loss = current_price * 0.95

        return max(stop_loss, current_price * 0.5)  # 止损不能低于现价的50%

    def update_trailing_stop(self, df: pd.DataFrame,
                            current_stop: float,
                            is_long: bool = True,
                            activation_pct: float = 0.02) -> float:
        """
        更新追踪止损

        参数:
            df: 价格数据
            current_stop: 当前止损价
            is_long: 是否做多
            activation_pct: 激活追踪止损的盈亏比例

        返回:
            更新后的止损价
        """
        entry_price = df['close'].iloc[0]
        current_price = df['close'].iloc[-1]

        if is_long:
            pnl_pct = (current_price - entry_price) / entry_price

            # 只有盈利超过阈值才启动追踪止损
            if pnl_pct >= activation_pct:
                new_stop = self.calculate_stop_loss(df, entry_price)
                # 只向上移动止损，不向下
                return max(current_stop, new_stop)
        else:
            # 做空逻辑（暂不实现）
            pass

        return current_stop


class PositionSizing:
    """仓位管理"""

    @staticmethod
    def calculate_fixed_fraction(capital: float,
                                risk_per_trade: float = 0.02) -> float:
        """
        固定比例仓位

        参数:
            capital: 总资金
            risk_per_trade: 单笔交易风险比例

        返回:
            仓位大小
        """
        return capital * risk_per_trade

    @staticmethod
    def calculate_kelly_position(capital: float,
                               win_rate: float,
                               avg_win: float,
                               avg_loss: float) -> float:
        """
        Kelly公式仓位

        参数:
            capital: 总资金
            win_rate: 胜率
            avg_win: 平均盈利
            avg_loss: 平均亏损

        返回:
            仓位大小
        """
        if avg_loss == 0:
            return 0

        # Kelly公式: f = (bp - q) / b
        # b = avg_win / avg_loss (盈亏比)
        # p = win_rate, q = 1 - win_rate

        b = avg_win / avg_loss
        p = win_rate
        q = 1 - p

        kelly_fraction = (b * p - q) / b

        # 限制最大仓位不超过25%
        kelly_fraction = max(0, min(kelly_fraction, 0.25))

        return capital * kelly_fraction

    @staticmethod
    def calculate_atr_position(capital: float,
                             entry_price: float,
                             stop_loss_price: float,
                             risk_per_trade: float = 0.02) -> float:
        """
        基于ATR的仓位计算

        参数:
            capital: 总资金
            entry_price: 入场价格
            stop_loss_price: 止损价格
            risk_per_trade: 单笔交易风险比例

        返回:
            仓位大小
        """
        risk_amount = capital * risk_per_trade
        risk_per_share = abs(entry_price - stop_loss_price)

        if risk_per_share == 0:
            return 0

        shares = int(risk_amount / risk_per_share)
        position_size = shares * entry_price

        return position_size

    @staticmethod
    def calculate_risk_parity_position(capital: float,
                                      volatilities: Dict[str, float],
                                      target_volatility: float = 0.15) -> Dict[str, float]:
        """
        风险平价仓位

        参数:
            capital: 总资金
            volatilities: 各资产波动率字典
            target_volatility: 目标波动率

        返回:
            仓位大小字典
        """
        # 权重与波动率成反比
        inv_vol = {k: 1/v if v > 0 else 0 for k, v in volatilities.items()}
        total_inv_vol = sum(inv_vol.values())

        if total_inv_vol == 0:
            return {k: 0 for k in volatilities.keys()}

        weights = {k: v/total_inv_vol for k, v in inv_vol.items()}

        # 调整到目标波动率
        portfolio_volatility = np.sqrt(
            sum((weights[k] * volatilities[k])**2 for k in weights.keys())
        )

        scaling_factor = target_volatility / portfolio_volatility if portfolio_volatility > 0 else 1
        scaled_weights = {k: w * scaling_factor for k, w in weights.items()}

        # 计算仓位
        positions = {k: capital * w for k, w in scaled_weights.items()}

        return positions


class CorrelationControl:
    """相关性控制"""

    @staticmethod
    def calculate_correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
        """
        计算相关性矩阵

        参数:
            returns: 收益率DataFrame

        返回:
            相关性矩阵
        """
        return returns.corr()

    @staticmethod
    def get_highly_correlated_pairs(correlation_matrix: pd.DataFrame,
                                   threshold: float = 0.7) -> List[Tuple[str, str]]:
        """
        获取高相关性股票对

        参数:
            correlation_matrix: 相关性矩阵
            threshold: 相关性阈值

        返回:
            高相关性股票对列表
        """
        highly_correlated = []

        for i in range(len(correlation_matrix)):
            for j in range(i+1, len(correlation_matrix)):
                corr = correlation_matrix.iloc[i, j]
                if abs(corr) >= threshold:
                    pair = (correlation_matrix.index[i], correlation_matrix.columns[j])
                    highly_correlated.append(pair)

        return highly_correlated

    @staticmethod
    def reduce_correlation(weights: Dict[str, float],
                          correlation_matrix: pd.DataFrame,
                          max_correlation: float = 0.7,
                          reduction_factor: float = 0.5) -> Dict[str, float]:
        """
        降低组合相关性

        参数:
            weights: 权重字典
            correlation_matrix: 相关性矩阵
            max_correlation: 最大允许相关性
            reduction_factor: 降低因子

        返回:
            调整后的权重字典
        """
        adjusted_weights = weights.copy()

        highly_correlated_pairs = CorrelationControl.get_highly_correlated_pairs(
            correlation_matrix, max_correlation
        )

        for asset1, asset2 in highly_correlated_pairs:
            if asset1 in adjusted_weights and asset2 in adjusted_weights:
                # 降低相关性较高的股票对中权重较小的那个
                if adjusted_weights[asset1] < adjusted_weights[asset2]:
                    adjusted_weights[asset1] *= reduction_factor
                else:
                    adjusted_weights[asset2] *= reduction_factor

        # 归一化
        total_weight = sum(adjusted_weights.values())
        if total_weight > 0:
            adjusted_weights = {k: v/total_weight for k, v in adjusted_weights.items()}

        return adjusted_weights


class RiskManager:
    """风险管理器"""

    def __init__(self):
        """初始化风险管理器"""
        self.stop_loss_calculator = DynamicStopLoss(method='atr_trailing')
        self.position_sizer = PositionSizing()
        self.correlation_control = CorrelationControl()

    def assess_position_risk(self, df: pd.DataFrame,
                            ts_code: str,
                            entry_price: float,
                            position_size: float,
                            holding_days: int = 0) -> PositionRisk:
        """
        评估持仓风险

        参数:
            df: 价格数据
            ts_code: 股票代码
            entry_price: 入场价格
            position_size: 仓位大小
            holding_days: 持仓天数

        返回:
            持仓风险
        """
        current_price = df['close'].iloc[-1]

        # 计算止损价
        stop_loss_price = self.stop_loss_calculator.calculate_stop_loss(df, entry_price)

        # 计算ATR止损
        from indicators.indicators import atr
        atr_value = atr(df['high'].values, df['low'].values, df['close'].values, 14)
        current_atr = atr_value[-1] if len(atr_value) > 0 else current_price * 0.02
        atr_stop_price = current_price - current_atr * 2.5

        # 使用更保守的止损
        final_stop_price = max(stop_loss_price, atr_stop_price)

        # 止盈价（1:2的风险收益比）
        risk_amount = entry_price - final_stop_price
        take_profit_price = entry_price + risk_amount * 2

        # 计算盈亏
        unrealized_pnl = (current_price - entry_price) * (position_size / entry_price)
        unrealized_pnl_pct = (current_price - entry_price) / entry_price

        # 风险等级判断
        if unrealized_pnl_pct < -0.08:
            risk_level = 'extreme'
        elif unrealized_pnl_pct < -0.05:
            risk_level = 'high'
        elif unrealized_pnl_pct < -0.02:
            risk_level = 'medium'
        else:
            risk_level = 'low'

        return PositionRisk(
            ts_code=ts_code,
            entry_price=entry_price,
            current_price=current_price,
            position_size=position_size,
            stop_loss_price=final_stop_price,
            take_profit_price=take_profit_price,
            atr_stop_price=atr_stop_price,
            unrealized_pnl=unrealized_pnl,
            unrealized_pnl_pct=unrealized_pnl_pct,
            risk_level=risk_level
        )

    def assess_portfolio_risk(self,
                             positions: List[PositionRisk],
                             returns_df: pd.DataFrame = None) -> PortfolioRisk:
        """
        评估组合风险

        参数:
            positions: 持仓风险列表
            returns_df: 收益率DataFrame

        返回:
            组合风险
        """
        total_value = sum(p.position_size for p in positions)
        total_unrealized_pnl = sum(p.unrealized_pnl for p in positions)
        total_unrealized_pnl_pct = total_unrealized_pnl / total_value if total_value > 0 else 0

        # 最大持仓风险
        max_position_risk = max([abs(p.unrealized_pnl_pct) for p in positions]) if positions else 0

        # 组合波动率
        portfolio_volatility = 0
        if returns_df is not None and not returns_df.empty:
            portfolio_returns = returns_df.mean(axis=1)
            portfolio_volatility = portfolio_returns.std() * np.sqrt(252)

        # VaR计算
        confidence_95 = norm.ppf(0.05)
        confidence_99 = norm.ppf(0.01)

        value_at_risk_95 = total_value * portfolio_volatility * confidence_95
        value_at_risk_99 = total_value * portfolio_volatility * confidence_99

        # 集中度风险（最大持仓比例）
        concentration_risk = max([p.position_size for p in positions]) / total_value if positions and total_value > 0 else 0

        # 相关性风险（简化处理）
        correlation_risk = 0.5  # 默认中等相关性风险

        return PortfolioRisk(
            total_value=total_value,
            total_unrealized_pnl=total_unrealized_pnl,
            total_unrealized_pnl_pct=total_unrealized_pnl_pct,
            max_position_risk=max_position_risk,
            portfolio_volatility=portfolio_volatility,
            value_at_risk_95=value_at_risk_95,
            value_at_risk_99=value_at_risk_99,
            concentration_risk=concentration_risk,
            correlation_risk=correlation_risk
        )

    def check_risk_limits(self, portfolio_risk: PortfolioRisk,
                          max_portfolio_loss: float = 0.05,
                          max_position_loss: float = 0.10,
                          max_volatility: float = 0.30) -> Dict[str, bool]:
        """
        检查风险限制

        参数:
            portfolio_risk: 组合风险
            max_portfolio_loss: 最大组合亏损
            max_position_loss: 最大单只股票亏损
            max_volatility: 最大波动率

        返回:
            风险检查结果字典
        """
        checks = {
            'portfolio_loss_ok': portfolio_risk.total_unrealized_pnl_pct >= -max_portfolio_loss,
            'position_loss_ok': portfolio_risk.max_position_risk <= max_position_loss,
            'volatility_ok': portfolio_risk.portfolio_volatility <= max_volatility,
            'concentration_ok': portfolio_risk.concentration_risk <= 0.3
        }

        return checks

    def calculate_optimal_position(self,
                                  capital: float,
                                  entry_price: float,
                                  stop_loss_price: float,
                                  method: str = 'fixed_fraction') -> float:
        """
        计算最优仓位

        参数:
            capital: 总资金
            entry_price: 入场价格
            stop_loss_price: 止损价格
            method: 仓位计算方法

        返回:
            仓位大小
        """
        if method == 'fixed_fraction':
            return self.position_sizer.calculate_fixed_fraction(capital)
        elif method == 'atr':
            return self.position_sizer.calculate_atr_position(
                capital, entry_price, stop_loss_price
            )
        else:
            return self.position_sizer.calculate_fixed_fraction(capital)
