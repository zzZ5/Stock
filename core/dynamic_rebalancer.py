"""
动态调仓模块
提供信号追踪、自动调仓、策略切换等功能
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
from core.logger import get_logger

logger = get_logger(__name__)


class SignalStrength(Enum):
    """信号强度"""
    STRONG_BUY = 5
    BUY = 4
    WEAK_BUY = 3
    NEUTRAL = 2
    WEAK_SELL = 1
    SELL = 0
    STRONG_SELL = -1


@dataclass
class RebalanceSignal:
    """调仓信号"""
    ts_code: str
    action: str  # 'buy', 'sell', 'hold', 'reduce', 'add'
    current_weight: float
    target_weight: float
    signal_strength: SignalStrength
    confidence: float  # 信号置信度
    reason: str


@dataclass
class RebalanceOrder:
    """调仓订单"""
    ts_code: str
    action: str  # 'buy', 'sell'
    shares: int
    price: float
    amount: float
    reason: str


class SignalTracker:
    """信号追踪器"""

    def __init__(self, lookback_period: int = 20,
                strength_threshold: float = 0.6):
        """
        初始化信号追踪器

        参数:
            lookback_period: 回溯周期
            strength_threshold: 信号强度阈值
        """
        self.lookback_period = lookback_period
        self.strength_threshold = strength_threshold
        self.signal_history: Dict[str, List[Dict]] = {}

    def track_signal(self, ts_code: str,
                    signal_value: float,
                    signal_score: float,
                    current_price: float,
                    timestamp: pd.Timestamp) -> Dict:
        """
        追踪信号

        参数:
            ts_code: 股票代码
            signal_value: 信号值
            signal_score: 信号分数 (0-100)
            current_price: 当前价格
            timestamp: 时间戳

        返回:
            信号状态字典
        """
        if ts_code not in self.signal_history:
            self.signal_history[ts_code] = []

        # 添加信号历史
        signal_record = {
            'timestamp': timestamp,
            'signal_value': signal_value,
            'signal_score': signal_score,
            'price': current_price
        }
        self.signal_history[ts_code].append(signal_record)

        # 只保留最近的N条记录
        if len(self.signal_history[ts_code]) > self.lookback_period:
            self.signal_history[ts_code] = self.signal_history[ts_code][-self.lookback_period:]

        # 分析信号趋势
        signal_status = self._analyze_signal_trend(ts_code)

        return signal_status

    def _analyze_signal_trend(self, ts_code: str) -> Dict:
        """
        分析信号趋势

        参数:
            ts_code: 股票代码

        返回:
            信号状态字典
        """
        history = self.signal_history[ts_code]

        if len(history) < 3:
            return {
                'trend': 'unknown',
                'strength': SignalStrength.NEUTRAL,
                'consistency': 0.0,
                'change_rate': 0.0
            }

        # 计算趋势
        recent_scores = [h['signal_score'] for h in history[-5:]]
        avg_score = np.mean(recent_scores)
        std_score = np.std(recent_scores)

        # 计算变化率
        change_rate = (recent_scores[-1] - recent_scores[0]) / recent_scores[0] if recent_scores[0] > 0 else 0

        # 判断信号强度
        if avg_score > 75:
            strength = SignalStrength.STRONG_BUY
        elif avg_score > 60:
            strength = SignalStrength.BUY
        elif avg_score > 50:
            strength = SignalStrength.WEAK_BUY
        elif avg_score > 40:
            strength = SignalStrength.NEUTRAL
        elif avg_score > 25:
            strength = SignalStrength.WEAK_SELL
        elif avg_score > 10:
            strength = SignalStrength.SELL
        else:
            strength = SignalStrength.STRONG_SELL

        # 判断趋势方向
        if change_rate > 0.1:
            trend = 'improving'
        elif change_rate < -0.1:
            trend = 'deteriorating'
        else:
            trend = 'stable'

        # 计算一致性（标准差越小，一致性越高）
        consistency = max(0, 1 - std_score / 50)

        return {
            'trend': trend,
            'strength': strength,
            'consistency': consistency,
            'change_rate': change_rate,
            'avg_score': avg_score
        }

    def get_signal_strength(self, ts_code: str) -> SignalStrength:
        """
        获取信号强度

        参数:
            ts_code: 股票代码

        返回:
            信号强度
        """
        if ts_code not in self.signal_history:
            return SignalStrength.NEUTRAL

        status = self._analyze_signal_trend(ts_code)
        return status['strength']


class AutoRebalancer:
    """自动调仓器"""

    def __init__(self, rebalance_threshold: float = 0.05,
                max_single_position: float = 0.25,
                min_position_size: float = 0.02):
        """
        初始化自动调仓器

        参数:
            rebalance_threshold: 调仓阈值（偏离目标权重多少时调仓）
            max_single_position: 单只股票最大仓位
            min_position_size: 最小仓位
        """
        self.rebalance_threshold = rebalance_threshold
        self.max_single_position = max_single_position
        self.min_position_size = min_position_size
        self.signal_tracker = SignalTracker()

    def calculate_target_weights(self,
                                scores: Dict[str, float],
                                max_positions: int = 20) -> Dict[str, float]:
        """
        计算目标权重

        参数:
            scores: 股票评分字典
            max_positions: 最大持仓数量

        返回:
            目标权重字典
        """
        # 过滤低分股票
        filtered_scores = {k: v for k, v in scores.items() if v > 50}

        if not filtered_scores:
            return {}

        # 选择Top N股票
        sorted_stocks = sorted(filtered_scores.items(), key=lambda x: x[1], reverse=True)
        selected_stocks = sorted_stocks[:max_positions]

        # 计算权重（基于分数的相对大小）
        total_score = sum(score for _, score in selected_stocks)
        weights = {stock: score/total_score for stock, score in selected_stocks}

        return weights

    def check_rebalance_needed(self,
                             current_weights: Dict[str, float],
                             target_weights: Dict[str, float]) -> bool:
        """
        检查是否需要调仓

        参数:
            current_weights: 当前权重
            target_weights: 目标权重

        返回:
            是否需要调仓
        """
        all_stocks = set(current_weights.keys()) | set(target_weights.keys())

        for stock in all_stocks:
            current_weight = current_weights.get(stock, 0)
            target_weight = target_weights.get(stock, 0)

            # 计算偏离度
            if target_weight > 0:
                deviation = abs(current_weight - target_weight) / target_weight
            else:
                deviation = abs(current_weight)

            if deviation > self.rebalance_threshold:
                return True

        return False

    def generate_rebalance_signals(self,
                                  current_weights: Dict[str, float],
                                  target_weights: Dict[str, float],
                                  signal_strengths: Dict[str, SignalStrength] = None) -> List[RebalanceSignal]:
        """
        生成调仓信号

        参数:
            current_weights: 当前权重
            target_weights: 目标权重
            signal_strengths: 信号强度字典

        返回:
            调仓信号列表
        """
        signals = []

        all_stocks = set(current_weights.keys()) | set(target_weights.keys())

        for stock in all_stocks:
            current_weight = current_weights.get(stock, 0)
            target_weight = target_weights.get(stock, 0)
            strength = signal_strengths.get(stock, SignalStrength.NEUTRAL) if signal_strengths else SignalStrength.NEUTRAL

            # 确定调仓动作
            if target_weight > current_weight + self.min_position_size:
                action = 'buy' if current_weight == 0 else 'add'
                reason = '目标权重增加' if action == 'add' else '新建仓位'
            elif target_weight < current_weight - self.min_position_size:
                action = 'reduce' if target_weight > 0 else 'sell'
                reason = '目标权重降低' if action == 'reduce' else '清仓'
            else:
                action = 'hold'
                reason = '权重在目标范围内'

            # 计算置信度（基于信号强度）
            confidence = self._calculate_confidence(strength, action)

            signal = RebalanceSignal(
                ts_code=stock,
                action=action,
                current_weight=current_weight,
                target_weight=target_weight,
                signal_strength=strength,
                confidence=confidence,
                reason=reason
            )

            signals.append(signal)

        return signals

    def _calculate_confidence(self, strength: SignalStrength, action: str) -> float:
        """
        计算信号置信度

        参数:
            strength: 信号强度
            action: 调仓动作

        返回:
            置信度 (0-1)
        """
        base_confidence = {
            SignalStrength.STRONG_BUY: 0.9,
            SignalStrength.BUY: 0.8,
            SignalStrength.WEAK_BUY: 0.6,
            SignalStrength.NEUTRAL: 0.5,
            SignalStrength.WEAK_SELL: 0.4,
            SignalStrength.SELL: 0.3,
            SignalStrength.STRONG_SELL: 0.2
        }.get(strength, 0.5)

        # 根据动作调整置信度
        if action == 'hold':
            confidence = 0.5
        elif action in ['buy', 'sell'] and strength in [SignalStrength.STRONG_BUY, SignalStrength.STRONG_SELL]:
            confidence = base_confidence
        else:
            confidence = base_confidence * 0.8

        return confidence

    def generate_rebalance_orders(self,
                                 rebalance_signals: List[RebalanceSignal],
                                 total_value: float,
                                 prices: Dict[str, float]) -> List[RebalanceOrder]:
        """
        生成调仓订单

        参数:
            rebalance_signals: 调仓信号列表
            total_value: 总资金
            prices: 股票价格字典

        返回:
            调仓订单列表
        """
        orders = []

        for signal in rebalance_signals:
            if signal.action == 'hold':
                continue

            ts_code = signal.ts_code
            price = prices.get(ts_code, 0)

            if price <= 0:
                logger.warning(f"股票 {ts_code} 价格无效")
                continue

            # 计算调仓金额
            target_amount = total_value * signal.target_weight
            current_amount = total_value * signal.current_weight
            amount_diff = target_amount - current_amount

            # 限制单只股票仓位
            if signal.action in ['buy', 'add']:
                max_amount = total_value * self.max_single_position
                if target_amount > max_amount:
                    amount_diff = max_amount - current_amount

            shares = int(amount_diff / price)

            if shares == 0:
                continue

            # 确定买入/卖出
            if amount_diff > 0:
                action = 'buy'
            else:
                action = 'sell'
                shares = abs(shares)

            order = RebalanceOrder(
                ts_code=ts_code,
                action=action,
                shares=shares,
                price=price,
                amount=shares * price,
                reason=signal.reason
            )

            orders.append(order)

        return orders


class StrategySwitcher:
    """策略切换器"""

    def __init__(self, performance_window: int = 30):
        """
        初始化策略切换器

        参数:
            performance_window: 性能评估窗口
        """
        self.performance_window = performance_window
        self.strategy_performance: Dict[str, List[float]] = {}
        self.current_strategy: str = None

    def update_performance(self, strategy_name: str, performance: float):
        """
        更新策略性能

        参数:
            strategy_name: 策略名称
            performance: 性能指标（如收益率、Sharpe比）
        """
        if strategy_name not in self.strategy_performance:
            self.strategy_performance[strategy_name] = []

        self.strategy_performance[strategy_name].append(performance)

        # 只保留最近的N条记录
        if len(self.strategy_performance[strategy_name]) > self.performance_window:
            self.strategy_performance[strategy_name] = self.strategy_performance[strategy_name][-self.performance_window:]

    def get_best_strategy(self) -> str:
        """
        获取最佳策略

        返回:
            最佳策略名称
        """
        if not self.strategy_performance:
            return self.current_strategy

        # 计算各策略的平均性能
        avg_performance = {}
        for strategy, performances in self.strategy_performance.items():
            if performances:
                avg_performance[strategy] = np.mean(performances)

        if not avg_performance:
            return self.current_strategy

        # 选择平均性能最好的策略
        best_strategy = max(avg_performance, key=avg_performance.get)

        return best_strategy

    def should_switch_strategy(self, switch_threshold: float = 0.05) -> Tuple[bool, str]:
        """
        判断是否需要切换策略

        参数:
            switch_threshold: 切换阈值

        返回:
            (是否切换, 新策略名称)
        """
        best_strategy = self.get_best_strategy()

        if best_strategy is None or best_strategy == self.current_strategy:
            return False, None

        if self.current_strategy is None:
            return True, best_strategy

        # 计算性能差异
        current_perf = np.mean(self.strategy_performance.get(self.current_strategy, [0]))
        best_perf = np.mean(self.strategy_performance.get(best_strategy, [0]))

        if best_perf - current_perf > switch_threshold:
            logger.info(f"策略切换: {self.current_strategy} -> {best_strategy}")
            self.current_strategy = best_strategy
            return True, best_strategy

        return False, None
