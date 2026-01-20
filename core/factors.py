"""
多维度因子体系模块
提供基本面、技术面、资金面、情绪面等多维度因子计算
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from scipy import stats
from indicators.indicators import (
    sma, ema, rsi, macd, atr, adx, kdj, williams_r,
    bollinger_bands, obv, cci, momentum
)
from indicators.indicators_extended import (
    wma, dema, tema, hull_ma, supertrend,
    volume_weighted_ma, money_flow_ratio
)
from core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class FactorScore:
    """因子评分"""
    factor_name: str
    factor_value: float
    factor_score: float  # 归一化后的分数 (0-100)
    factor_direction: str  # 'positive', 'negative', 'neutral'


class FundamentalFactors:
    """基本面因子"""

    @staticmethod
    def calculate_pe_ratio(price: float, eps: float) -> float:
        """
        计算市盈率

        参数:
            price: 股价
            eps: 每股收益

        返回:
            市盈率
        """
        if eps <= 0:
            return np.nan
        return price / eps

    @staticmethod
    def calculate_pb_ratio(price: float, book_value_per_share: float) -> float:
        """
        计算市净率

        参数:
            price: 股价
            book_value_per_share: 每股净资产

        返回:
            市净率
        """
        if book_value_per_share <= 0:
            return np.nan
        return price / book_value_per_share

    @staticmethod
    def calculate_ps_ratio(price: float, sales_per_share: float) -> float:
        """
        计算市销率

        参数:
            price: 股价
            sales_per_share: 每股销售额

        返回:
            市销率
        """
        if sales_per_share <= 0:
            return np.nan
        return price / sales_per_share

    @staticmethod
    def calculate_roe(net_income: float, equity: float) -> float:
        """
        计算净资产收益率

        参数:
            net_income: 净利润
            equity: 净资产

        返回:
            净资产收益率
        """
        if equity <= 0:
            return np.nan
        return (net_income / equity) * 100

    @staticmethod
    def calculate_roa(net_income: float, total_assets: float) -> float:
        """
        计算总资产收益率

        参数:
            net_income: 净利润
            total_assets: 总资产

        返回:
            总资产收益率
        """
        if total_assets <= 0:
            return np.nan
        return (net_income / total_assets) * 100

    @staticmethod
    def calculate_debt_ratio(total_liabilities: float, total_assets: float) -> float:
        """
        计算资产负债率

        参数:
            total_liabilities: 总负债
            total_assets: 总资产

        返回:
            资产负债率
        """
        if total_assets <= 0:
            return np.nan
        return (total_liabilities / total_assets) * 100

    @staticmethod
    def calculate_current_ratio(current_assets: float, current_liabilities: float) -> float:
        """
        计算流动比率

        参数:
            current_assets: 流动资产
            current_liabilities: 流动负债

        返回:
            流动比率
        """
        if current_liabilities <= 0:
            return np.nan
        return current_assets / current_liabilities

    @staticmethod
    def calculate_growth_rate(current_value: float, previous_value: float) -> float:
        """
        计算增长率

        参数:
            current_value: 当前值
            previous_value: 前期值

        返回:
            增长率
        """
        if previous_value <= 0:
            return np.nan
        return ((current_value - previous_value) / previous_value) * 100


class TechnicalFactors:
    """技术面因子"""

    @staticmethod
    def trend_strength(df: pd.DataFrame, period: int = 20) -> float:
        """
        趋势强度因子

        参数:
            df: 价格数据
            period: 周期

        返回:
            趋势强度分数
        """
        if len(df) < period:
            return np.nan

        close_series = df['close']
        ma = sma(close_series, period)

        # 价格相对于均线位置
        price_position = (close_series.iloc[-1] - ma.iloc[-1]) / ma.iloc[-1] if ma.iloc[-1] > 0 else 0

        # ADX趋势强度
        try:
            adx_value = adx(df['high'].values, df['low'].values, df['close'].values, 14)
            adx_score = min(adx_value[-1] / 50, 1) if len(adx_value) > 0 and not np.isnan(adx_value[-1]) else 0
        except:
            adx_score = 0

        # 综合分数
        trend_score = 50 + (price_position * 100) + (adx_score * 50)
        return np.clip(trend_score, 0, 100)

    @staticmethod
    def momentum_factor(df: pd.DataFrame, period: int = 20) -> float:
        """
        动量因子

        参数:
            df: 价格数据
            period: 周期

        返回:
            动量分数
        """
        if len(df) < period + 1:
            return np.nan

        close_series = df['close']

        # 价格动量
        price_momentum = (close_series.iloc[-1] - close_series.iloc[-period-1]) / close_series.iloc[-period-1]

        # RSI动量
        try:
            rsi_value = rsi(close_series, 14)
            rsi_score = (rsi_value.iloc[-1] - 50) / 50 if len(rsi_value) > 0 and not np.isnan(rsi_value.iloc[-1]) else 0
        except:
            rsi_score = 0

        # MACD动量
        try:
            macd_line, _, _ = macd(close_series)
            macd_score = macd_line.iloc[-1] / close_series.iloc[-1] * 100 if len(macd_line) > 0 and not np.isnan(macd_line.iloc[-1]) else 0
        except:
            macd_score = 0

        # 综合分数
        momentum_score = 50 + (price_momentum * 500) + (rsi_score * 25) + (macd_score * 2)
        return np.clip(momentum_score, 0, 100)

    @staticmethod
    def volatility_factor(df: pd.DataFrame, period: int = 20) -> float:
        """
        波动率因子

        参数:
            df: 价格数据
            period: 周期

        返回:
            波动率分数
        """
        if len(df) < period:
            return np.nan

        close_series = df['close']

        # 历史波动率
        returns = np.diff(np.log(close_series.values))
        hist_vol = np.std(returns[-period:]) * np.sqrt(252) if len(returns) >= period else 0

        # ATR
        try:
            atr_value = atr(df['high'].values, df['low'].values, close_series.values, 14)
            atr_pct = atr_value[-1] / close_series.iloc[-1] * 100 if len(atr_value) > 0 and not np.isnan(atr_value[-1]) and close_series.iloc[-1] > 0 else 0
        except:
            atr_pct = 0

        # 布林带宽度
        try:
            bb_upper, bb_middle, bb_lower = bollinger_bands(close_series, 20)
            bb_width = (bb_upper.iloc[-1] - bb_lower.iloc[-1]) / bb_middle.iloc[-1] if len(bb_upper) > 0 and not np.isnan(bb_middle.iloc[-1]) else 0
        except:
            bb_width = 0

        # 综合分数（波动率适中为好）
        vol_score = 100 - min(hist_vol * 100, 50) - min(atr_pct * 10, 30) - min(bb_width * 50, 20)
        return np.clip(vol_score, 0, 100)

    @staticmethod
    def reversal_factor(df: pd.DataFrame) -> float:
        """
        反转因子（均值回归）

        参数:
            df: 价格数据

        返回:
            反转分数
        """
        if len(df) < 20:
            return np.nan

        close_series = df['close']
        close_values = close_series.values

        # Williams %R（超买超卖）
        try:
            wr = williams_r(df['high'].values, df['low'].values, close_values, 14)
            wr_score = (50 - wr[-1]) if len(wr) > 0 and not np.isnan(wr[-1]) else 0
        except:
            wr_score = 0

        # KDJ的J值
        try:
            k, d, j = kdj(close_values, df['high'].values, df['low'].values, 9)
            j_score = (50 - j[-1]) / 50 if len(j) > 0 and not np.isnan(j[-1]) else 0
        except:
            j_score = 0

        # CCI（超买超卖）
        try:
            cci_value = cci(df['high'].values, df['low'].values, close_values, 20)
            cci_score = max(0, (100 - abs(cci_value[-1]) / 5)) if len(cci_value) > 0 and not np.isnan(cci_value[-1]) else 50
        except:
            cci_score = 50

        # 综合分数
        reversal_score = 50 + wr_score * 0.3 + j_score * 20 + (cci_score - 50) * 0.5
        return np.clip(reversal_score, 0, 100)


class MoneyFlowFactors:
    """资金面因子"""

    @staticmethod
    def volume_factor(df: pd.DataFrame, period: int = 20) -> float:
        """
        成交量因子

        参数:
            df: 价格数据
            period: 周期

        返回:
            成交量分数
        """
        if 'volume' not in df.columns or len(df) < period:
            return np.nan

        volume_series = df['volume']

        # 成交量相对于均线
        try:
            vol_ma = sma(volume_series, period)
            vol_ratio = volume_series.iloc[-1] / vol_ma.iloc[-1] if len(vol_ma) > 0 and vol_ma.iloc[-1] > 0 and not np.isnan(vol_ma.iloc[-1]) else 0
        except:
            vol_ratio = 0

        # 成交量趋势
        try:
            vol_trend = (vol_ma.iloc[-1] / vol_ma.iloc[-5] - 1) if len(vol_ma) >= 5 else 0
        except:
            vol_trend = 0

        # 成交量稳定性
        try:
            vol_std = np.std(volume_series.values[-period:]) if len(volume_series) >= period else 0
            vol_mean = np.mean(volume_series.values[-period:]) if len(volume_series) >= period else 0
            vol_stability = 1 - min(vol_std / vol_mean, 1) if vol_mean > 0 else 0
        except:
            vol_stability = 0

        # 综合分数
        vol_score = 50 + (vol_ratio - 1) * 50 + vol_trend * 100 + vol_stability * 20
        return np.clip(vol_score, 0, 100)

    @staticmethod
    def capital_flow(df: pd.DataFrame) -> float:
        """
        资金流向因子

        参数:
            df: 价格数据（需要包含amount列）

        返回:
            资金流向分数
        """
        if 'amount' not in df.columns or len(df) < 20:
            return np.nan

        amount = df['amount'].values
        close = df['close'].values

        # 资金流向
        # 定义流入：收盘价上涨时
        price_change = np.diff(close)
        amount_change = amount[1:]

        inflow = amount_change[price_change > 0].sum()
        outflow = abs(amount_change[price_change < 0].sum())

        if (inflow + outflow) == 0:
            return 50

        net_flow_ratio = (inflow - outflow) / (inflow + outflow)

        # OBV
        try:
            volume = df.get('volume', np.ones(len(close))).values
            obv_value = obv(close, volume=volume)
            obv_trend = (obv_value[-1] - obv_value[-20]) / 20 if len(obv_value) >= 20 else 0
        except:
            obv_trend = 0

        # 综合分数
        flow_score = 50 + net_flow_ratio * 50 + np.sign(obv_trend) * min(abs(obv_trend) * 10, 50)
        return np.clip(flow_score, 0, 100)

    @staticmethod
    def turnover_factor(df: pd.DataFrame, market_cap: float = None) -> float:
        """
        换手率因子

        参数:
            df: 价格数据
            market_cap: 总市值

        返回:
            换手率分数
        """
        if 'volume' not in df.columns or market_cap is None or market_cap <= 0:
            return np.nan

        volume = df['volume'].values
        close = df['close'].values

        # 计算换手率
        turnover = volume[-1] * close[-1] / market_cap * 100

        # 换手率适中为好（过低流动性差，过高可能是投机）
        if turnover < 1:
            # 流动性不足
            score = turnover * 50
        elif turnover < 10:
            # 适中
            score = 50 + (turnover - 1) * 5
        else:
            # 换手率过高
            score = 90 - min((turnover - 10) * 5, 40)

        return np.clip(score, 0, 100)


class SentimentFactors:
    """情绪面因子"""

    @staticmethod
    def market_sentiment(df: pd.DataFrame, index_df: pd.DataFrame = None) -> float:
        """
        市场情绪因子

        参数:
            df: 股票数据
            index_df: 指数数据（可选）

        返回:
            市场情绪分数
        """
        if len(df) < 20:
            return np.nan

        close = df['close'].values

        # 相对强度（相对于大盘）
        if index_df is not None and 'close' in index_df.columns:
            stock_return = (close[-1] - close[-20]) / close[-20]
            index_return = (index_df['close'].values[-1] - index_df['close'].values[-20]) / index_df['close'].values[-20]
            relative_strength = stock_return - index_return
            rs_score = 50 + relative_strength * 500
        else:
            rs_score = 50

        # 上涨日占比
        daily_returns = np.diff(close)
        up_days = np.sum(daily_returns > 0)
        up_ratio = up_days / len(daily_returns) if len(daily_returns) > 0 else 0

        # 连续上涨天数
        current_streak = 0
        for ret in reversed(daily_returns):
            if ret > 0:
                current_streak += 1
            else:
                break

        # 综合分数
        sentiment_score = rs_score * 0.6 + up_ratio * 50 + min(current_streak * 5, 50) * 0.1
        return np.clip(sentiment_score, 0, 100)

    @staticmethod
    def institutional_sentiment(volume_df: pd.DataFrame = None) -> float:
        """
        机构情绪因子（基于成交量和价格）

        参数:
            volume_df: 成交量数据

        返回:
            机构情绪分数
        """
        if volume_df is None or len(volume_df) < 20:
            return 50

        # 这里简化处理，实际需要机构买卖数据
        # 使用成交量集中度和价格稳定性作为代理

        if 'close' in volume_df.columns:
            close = volume_df['close'].values
            price_stability = 1 - min(np.std(close[-20:]) / np.mean(close[-20:]), 1)
        else:
            price_stability = 0.5

        if 'volume' in volume_df.columns:
            volume = volume_df['volume'].values
            vol_concentration = np.std(volume[-20:]) / np.mean(volume[-20:]) if len(volume) >= 20 else 0
            vol_score = 1 - min(vol_concentration, 1)
        else:
            vol_score = 0.5

        # 综合分数
        sentiment_score = 50 + (price_stability - 0.5) * 50 + (vol_score - 0.5) * 50
        return np.clip(sentiment_score, 0, 100)


class FactorCalculator:
    """多因子计算器"""

    def __init__(self):
        """初始化因子计算器"""
        self.fundamental = FundamentalFactors()
        self.technical = TechnicalFactors()
        self.money_flow = MoneyFlowFactors()
        self.sentiment = SentimentFactors()

    def calculate_all_factors(self, df: pd.DataFrame,
                             fundamental_data: Dict = None,
                             index_df: pd.DataFrame = None,
                             market_cap: float = None) -> Dict[str, float]:
        """
        计算所有维度因子

        参数:
            df: 股票价格数据
            fundamental_data: 基本面数据字典
            index_df: 指数数据
            market_cap: 总市值

        返回:
            因子值字典
        """
        factors = {}

        # 技术面因子
        try:
            factors['trend_strength'] = self.technical.trend_strength(df)
            factors['momentum'] = self.technical.momentum_factor(df)
            factors['volatility'] = self.technical.volatility_factor(df)
            factors['reversal'] = self.technical.reversal_factor(df)
        except Exception as e:
            logger.warning(f"技术面因子计算失败: {e}")

        # 资金面因子
        try:
            factors['volume'] = self.money_flow.volume_factor(df)
            factors['capital_flow'] = self.money_flow.capital_flow(df)
            factors['turnover'] = self.money_flow.turnover_factor(df, market_cap)
        except Exception as e:
            logger.warning(f"资金面因子计算失败: {e}")

        # 情绪面因子
        try:
            factors['market_sentiment'] = self.sentiment.market_sentiment(df, index_df)
            factors['institutional_sentiment'] = self.sentiment.institutional_sentiment(df)
        except Exception as e:
            logger.warning(f"情绪面因子计算失败: {e}")

        # 基本面因子（如果有数据）
        if fundamental_data:
            try:
                price = df['close'].iloc[-1] if len(df) > 0 else np.nan

                if 'eps' in fundamental_data:
                    factors['pe_ratio'] = self.fundamental.calculate_pe_ratio(
                        price, fundamental_data['eps'])
                if 'book_value' in fundamental_data:
                    factors['pb_ratio'] = self.fundamental.calculate_pb_ratio(
                        price, fundamental_data['book_value'])
                if 'sales' in fundamental_data:
                    factors['ps_ratio'] = self.fundamental.calculate_ps_ratio(
                        price, fundamental_data['sales'])
                if 'net_income' in fundamental_data and 'equity' in fundamental_data:
                    factors['roe'] = self.fundamental.calculate_roe(
                        fundamental_data['net_income'], fundamental_data['equity'])
                if 'net_income' in fundamental_data and 'total_assets' in fundamental_data:
                    factors['roa'] = self.fundamental.calculate_roa(
                        fundamental_data['net_income'], fundamental_data['total_assets'])
                if 'total_liabilities' in fundamental_data and 'total_assets' in fundamental_data:
                    factors['debt_ratio'] = self.fundamental.calculate_debt_ratio(
                        fundamental_data['total_liabilities'], fundamental_data['total_assets'])
            except Exception as e:
                logger.warning(f"基本面因子计算失败: {e}")

        return factors

    def normalize_factors(self, factors: Dict[str, float],
                         factor_directions: Dict[str, str]) -> Dict[str, FactorScore]:
        """
        归一化因子分数

        参数:
            factors: 因子值字典
            factor_directions: 因子方向 {'positive': 正向, 'negative': 负向}

        返回:
            归一化后的因子评分
        """
        normalized = {}

        for name, value in factors.items():
            if pd.isna(value):
                continue

            direction = factor_directions.get(name, 'positive')
            normalized[name] = FactorScore(
                factor_name=name,
                factor_value=value,
                factor_score=np.clip(value, 0, 100),
                factor_direction=direction
            )

        return normalized

    def calculate_composite_score(self,
                                  factor_scores: Dict[str, FactorScore],
                                  factor_weights: Dict[str, float]) -> float:
        """
        计算综合得分

        参数:
            factor_scores: 因子评分字典
            factor_weights: 因子权重字典

        返回:
            综合得分
        """
        total_score = 0.0
        total_weight = 0.0

        for name, score in factor_scores.items():
            weight = factor_weights.get(name, 0)

            # 根据因子方向调整得分
            if score.factor_direction == 'negative':
                adjusted_score = 100 - score.factor_score
            else:
                adjusted_score = score.factor_score

            total_score += adjusted_score * weight
            total_weight += weight

        if total_weight == 0:
            return 50.0

        return total_score / total_weight


def get_default_factor_weights() -> Dict[str, float]:
    """
    获取默认因子权重

    返回:
        因子权重字典
    """
    return {
        # 技术面因子权重 (40%)
        'trend_strength': 0.12,
        'momentum': 0.12,
        'volatility': 0.08,
        'reversal': 0.08,

        # 资金面因子权重 (30%)
        'volume': 0.12,
        'capital_flow': 0.10,
        'turnover': 0.08,

        # 情绪面因子权重 (20%)
        'market_sentiment': 0.10,
        'institutional_sentiment': 0.10,

        # 基本面因子权重 (10%)
        'pe_ratio': 0.03,
        'pb_ratio': 0.02,
        'roe': 0.03,
        'roa': 0.02
    }


def get_default_factor_directions() -> Dict[str, str]:
    """
    获取默认因子方向

    返回:
        因子方向字典
    """
    return {
        # 技术面
        'trend_strength': 'positive',
        'momentum': 'positive',
        'volatility': 'positive',  # 适中为好
        'reversal': 'positive',

        # 资金面
        'volume': 'positive',
        'capital_flow': 'positive',
        'turnover': 'positive',

        # 情绪面
        'market_sentiment': 'positive',
        'institutional_sentiment': 'positive',

        # 基本面
        'pe_ratio': 'negative',  # 越低越好
        'pb_ratio': 'negative',
        'roe': 'positive',
        'roa': 'positive'
    }
