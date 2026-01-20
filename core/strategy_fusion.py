"""
策略融合框架
提供多策略集成、策略权重、信号合成等功能
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
from enum import Enum
from core.logger import get_logger

logger = get_logger(__name__)


class StrategyType(Enum):
    """策略类型"""
    TREND = 'trend'
    MOMENTUM = 'momentum'
    REVERSAL = 'reversal'
    EVENT = 'event'
    FUNDAMENTAL = 'fundamental'
    ML = 'ml'


@dataclass
class StrategySignal:
    """策略信号"""
    strategy_name: str
    ts_code: str
    signal_value: float  # 信号值 (-1到1)
    signal_score: float  # 信号分数 (0-100)
    confidence: float  # 置信度 (0-1)
    timestamp: pd.Timestamp
    metadata: Dict = None


@dataclass
class FusedSignal:
    """融合信号"""
    ts_code: str
    fused_score: float  # 融合分数 (0-100)
    fused_signal: float  # 融合信号 (-1到1)
    consensus: float  # 共识度 (0-1)
    signal_count: int
    strategy_weights: Dict[str, float]
    strategy_signals: Dict[str, StrategySignal]


class BaseStrategy:
    """基础策略类"""

    def __init__(self, name: str, strategy_type: StrategyType):
        """
        初始化策略

        参数:
            name: 策略名称
            strategy_type: 策略类型
        """
        self.name = name
        self.strategy_type = strategy_type
        self.weight: float = 1.0
        self.is_enabled: bool = True

    def generate_signal(self, df: pd.DataFrame, ts_code: str,
                       timestamp: pd.Timestamp) -> StrategySignal:
        """
        生成信号（子类需要实现）

        参数:
            df: 股票数据
            ts_code: 股票代码
            timestamp: 时间戳

        返回:
            策略信号
        """
        raise NotImplementedError("子类需要实现此方法")


class TrendStrategy(BaseStrategy):
    """趋势策略"""

    def __init__(self, name: str = 'Trend', ma_fast: int = 20, ma_slow: int = 60):
        super().__init__(name, StrategyType.TREND)
        self.ma_fast = ma_fast
        self.ma_slow = ma_slow

    def generate_signal(self, df: pd.DataFrame, ts_code: str,
                       timestamp: pd.Timestamp) -> StrategySignal:
        """生成趋势信号"""
        if len(df) < self.ma_slow:
            return StrategySignal(
                strategy_name=self.name,
                ts_code=ts_code,
                signal_value=0,
                signal_score=50,
                confidence=0,
                timestamp=timestamp
            )

        from indicators.indicators import sma, adx

        close = df['close'].values
        ma_fast_value = sma(close, self.ma_fast)[-1]
        ma_slow_value = sma(close, self.ma_slow)[-1]

        # 趋势判断
        if ma_fast_value > ma_slow_value:
            trend_strength = (ma_fast_value / ma_slow_value - 1) * 100
            signal_value = np.clip(trend_strength / 5, -1, 1)
        else:
            trend_strength = (ma_slow_value / ma_fast_value - 1) * 100
            signal_value = -np.clip(trend_strength / 5, -1, 1)

        # ADX确认趋势强度
        adx_value = adx(df['high'].values, df['low'].values, close, 14)[-1]
        confidence = min(adx_value / 50, 1) if len(adx_value) > 0 else 0

        signal_score = 50 + signal_value * 50

        return StrategySignal(
            strategy_name=self.name,
            ts_code=ts_code,
            signal_value=signal_value,
            signal_score=signal_score,
            confidence=confidence,
            timestamp=timestamp
        )


class MomentumStrategy(BaseStrategy):
    """动量策略"""

    def __init__(self, name: str = 'Momentum', period: int = 20):
        super().__init__(name, StrategyType.MOMENTUM)
        self.period = period

    def generate_signal(self, df: pd.DataFrame, ts_code: str,
                       timestamp: pd.Timestamp) -> StrategySignal:
        """生成动量信号"""
        if len(df) < self.period + 1:
            return StrategySignal(
                strategy_name=self.name,
                ts_code=ts_code,
                signal_value=0,
                signal_score=50,
                confidence=0,
                timestamp=timestamp
            )

        close = df['close'].values

        # 价格动量
        momentum = (close[-1] - close[-self.period-1]) / close[-self.period-1]
        signal_value = np.clip(momentum * 10, -1, 1)

        # RSI确认
        from indicators.indicators import rsi
        rsi_value = rsi(close, 14)[-1]
        confidence = 1 - abs(rsi_value - 50) / 50  # RSI接近50时置信度低

        signal_score = 50 + signal_value * 50

        return StrategySignal(
            strategy_name=self.name,
            ts_code=ts_code,
            signal_value=signal_value,
            signal_score=signal_score,
            confidence=confidence,
            timestamp=timestamp
        )


class ReversalStrategy(BaseStrategy):
    """反转策略"""

    def __init__(self, name: str = 'Reversal', period: int = 14):
        super().__init__(name, StrategyType.REVERSAL)
        self.period = period

    def generate_signal(self, df: pd.DataFrame, ts_code: str,
                       timestamp: pd.Timestamp) -> StrategySignal:
        """生成反转信号"""
        if len(df) < self.period:
            return StrategySignal(
                strategy_name=self.name,
                ts_code=ts_code,
                signal_value=0,
                signal_score=50,
                confidence=0,
                timestamp=timestamp
            )

        from indicators.indicators import rsi, williams_r

        close = df['close'].values

        # RSI反转信号
        rsi_value = rsi(close, self.period)[-1]
        if rsi_value > 70:
            signal_value = -0.5  # 超买，做空
        elif rsi_value < 30:
            signal_value = 0.5  # 超卖，做多
        else:
            signal_value = 0

        # Williams %R确认
        wr = williams_r(df['high'].values, df['low'].values, close, self.period)[-1]
        confidence = (100 - abs(wr)) / 100

        signal_score = 50 + signal_value * 50

        return StrategySignal(
            strategy_name=self.name,
            ts_code=ts_code,
            signal_value=signal_value,
            signal_score=signal_score,
            confidence=confidence,
            timestamp=timestamp
        )


class StrategyFusion:
    """策略融合器"""

    def __init__(self, strategies: List[BaseStrategy] = None):
        """
        初始化策略融合器

        参数:
            strategies: 策略列表
        """
        self.strategies = strategies if strategies else []
        self.strategy_weights: Dict[str, float] = {}
        self.fusion_method: str = 'weighted_average'  # 'weighted_average', 'voting', 'ensemble'
        self.performance_history: Dict[str, List[float]] = {}

    def add_strategy(self, strategy: BaseStrategy, weight: float = None):
        """
        添加策略

        参数:
            strategy: 策略实例
            weight: 策略权重（None表示平均分配）
        """
        self.strategies.append(strategy)

        if weight is not None:
            self.strategy_weights[strategy.name] = weight
        else:
            self._rebalance_weights()

    def remove_strategy(self, strategy_name: str):
        """
        移除策略

        参数:
            strategy_name: 策略名称
        """
        self.strategies = [s for s in self.strategies if s.name != strategy_name]
        if strategy_name in self.strategy_weights:
            del self.strategy_weights[strategy_name]

    def _rebalance_weights(self):
        """重新平衡权重"""
        total_weight = sum(s.weight for s in self.strategies if s.is_enabled)

        if total_weight > 0:
            for strategy in self.strategies:
                if strategy.is_enabled:
                    self.strategy_weights[strategy.name] = strategy.weight / total_weight

    def fuse_signals(self, strategy_signals: List[StrategySignal]) -> FusedSignal:
        """
        融合策略信号

        参数:
            strategy_signals: 策略信号列表

        返回:
            融合信号
        """
        if not strategy_signals:
            raise ValueError("没有可融合的信号")

        ts_code = strategy_signals[0].ts_code

        # 过滤启用的策略
        enabled_signals = [s for s in strategy_signals
                          if s.strategy_name in self.strategy_weights]

        if not enabled_signals:
            enabled_signals = strategy_signals

        # 根据融合方法合成信号
        if self.fusion_method == 'weighted_average':
            return self._weighted_average_fusion(enabled_signals)
        elif self.fusion_method == 'voting':
            return self._voting_fusion(enabled_signals)
        elif self.fusion_method == 'ensemble':
            return self._ensemble_fusion(enabled_signals)
        else:
            return self._weighted_average_fusion(enabled_signals)

    def _weighted_average_fusion(self, signals: List[StrategySignal]) -> FusedSignal:
        """加权平均融合"""
        total_weight = 0
        weighted_signal = 0
        weighted_score = 0

        for signal in signals:
            weight = self.strategy_weights.get(signal.strategy_name, 1.0 / len(signals))
            weighted_signal += signal.signal_value * weight * signal.confidence
            weighted_score += signal.signal_score * weight * signal.confidence
            total_weight += weight * signal.confidence

        if total_weight == 0:
            fused_signal = 0
            fused_score = 50
        else:
            fused_signal = weighted_signal / total_weight
            fused_score = weighted_score / total_weight

        # 计算共识度
        signal_values = [s.signal_value for s in signals]
        consensus = 1 - np.std(signal_values) / 2  # 信号标准差越小，共识度越高

        # 策略信号字典
        strategy_signals_dict = {s.strategy_name: s for s in signals}

        return FusedSignal(
            ts_code=signals[0].ts_code,
            fused_score=np.clip(fused_score, 0, 100),
            fused_signal=np.clip(fused_signal, -1, 1),
            consensus=consensus,
            signal_count=len(signals),
            strategy_weights=self.strategy_weights.copy(),
            strategy_signals=strategy_signals_dict
        )

    def _voting_fusion(self, signals: List[StrategySignal]) -> FusedSignal:
        """投票融合"""
        # 统计买入/卖出/中性票数
        buy_votes = sum(1 for s in signals if s.signal_value > 0.2)
        sell_votes = sum(1 for s in signals if s.signal_value < -0.2)
        neutral_votes = len(signals) - buy_votes - sell_votes

        # 确定融合信号
        if buy_votes > sell_votes and buy_votes > neutral_votes:
            fused_signal = min(buy_votes / len(signals), 1)
        elif sell_votes > buy_votes and sell_votes > neutral_votes:
            fused_signal = -min(sell_votes / len(signals), 1)
        else:
            fused_signal = 0

        fused_score = 50 + fused_signal * 50

        # 共识度
        consensus = max(buy_votes, sell_votes, neutral_votes) / len(signals)

        strategy_signals_dict = {s.strategy_name: s for s in signals}

        return FusedSignal(
            ts_code=signals[0].ts_code,
            fused_score=fused_score,
            fused_signal=fused_signal,
            consensus=consensus,
            signal_count=len(signals),
            strategy_weights=self.strategy_weights.copy(),
            strategy_signals=strategy_signals_dict
        )

    def _ensemble_fusion(self, signals: List[StrategySignal]) -> FusedSignal:
        """集成融合（结合加权平均和投票）"""
        # 先用加权平均
        wa_result = self._weighted_average_fusion(signals)

        # 再用投票
        vote_result = self._voting_fusion(signals)

        # 加权组合
        alpha = 0.6  # 加权平均权重
        fused_signal = alpha * wa_result.fused_signal + (1 - alpha) * vote_result.fused_signal
        fused_score = alpha * wa_result.fused_score + (1 - alpha) * vote_result.fused_score

        return FusedSignal(
            ts_code=signals[0].ts_code,
            fused_score=np.clip(fused_score, 0, 100),
            fused_signal=np.clip(fused_signal, -1, 1),
            consensus=wa_result.consensus,
            signal_count=len(signals),
            strategy_weights=self.strategy_weights.copy(),
            strategy_signals=wa_result.strategy_signals
        )

    def update_strategy_weights(self, performance_dict: Dict[str, float]):
        """
        根据性能更新策略权重

        参数:
            performance_dict: 策略性能字典
        """
        for strategy_name, performance in performance_dict.items():
            if strategy_name not in self.strategy_weights:
                continue

            # 记录性能历史
            if strategy_name not in self.performance_history:
                self.performance_history[strategy_name] = []

            self.performance_history[strategy_name].append(performance)

        # 基于性能调整权重
        total_performance = 0
        for strategy in self.strategies:
            if strategy.name in self.performance_history:
                avg_performance = np.mean(self.performance_history[strategy.name])
                # 使用softmax转换为权重
                total_performance += np.exp(avg_performance)

        if total_performance > 0:
            for strategy in self.strategies:
                if strategy.name in self.performance_history:
                    avg_performance = np.mean(self.performance_history[strategy.name])
                    self.strategy_weights[strategy.name] = np.exp(avg_performance) / total_performance

    def get_strategy_summary(self) -> pd.DataFrame:
        """
        获取策略汇总

        返回:
            策略汇总DataFrame
        """
        data = []
        for strategy in self.strategies:
            data.append({
                'strategy_name': strategy.name,
                'strategy_type': strategy.strategy_type.value,
                'weight': self.strategy_weights.get(strategy.name, 0),
                'enabled': strategy.is_enabled,
                'avg_performance': np.mean(self.performance_history.get(strategy.name, [0])) if strategy.name in self.performance_history else 0
            })

        return pd.DataFrame(data)
