"""
扩展技术指标模块
添加更多高级技术指标
"""
import numpy as np
import pandas as pd
from typing import Dict, Optional
from scipy import stats
from scipy.signal import argrelextrema

from core.validators import (
    ValidationError,
    ParameterValidator,
    SafeCalculator,
)
from core.logger import get_indicator_logger

logger = get_indicator_logger()


def wma(s: pd.Series, n: int) -> pd.Series:
    """
    加权移动平均线 Weighted Moving Average
    
    用途: 给近期价格更高权重，反应更快
    """
    n = ParameterValidator.validate_period(n, "WMA周期", min_period=1, max_period=1000)
    
    weights = np.arange(1, n + 1)
    return s.rolling(n).apply(
        lambda x: np.dot(x, weights) / weights.sum(),
        raw=True
    )


def dema(close: pd.Series, n: int = 21) -> pd.Series:
    """
    双指数移动平均线 Double Exponential Moving Average
    
    用途: 消除EMA的滞后性
    """
    n = ParameterValidator.validate_period(n, "DEMA周期", min_period=1, max_period=200)
    
    ema1 = ema(close, n)
    ema2 = ema(ema1, n)
    dema_val = 2 * ema1 - ema2
    
    return dema_val


def tema(close: pd.Series, n: int = 21) -> pd.Series:
    """
    三重指数移动平均线 Triple Exponential Moving Average
    
    用途: 进一步减少滞后
    """
    n = ParameterValidator.validate_period(n, "TEMA周期", min_period=1, max_period=200)
    
    from indicators.indicators import ema
    ema1 = ema(close, n)
    ema2 = ema(ema1, n)
    ema3 = ema(ema2, n)
    
    tema_val = 3 * ema1 - 3 * ema2 + ema3
    return tema_val


def hull_ma(close: pd.Series, n: int = 20) -> pd.Series:
    """
    赫尔移动平均线 Hull Moving Average
    
    用途: 最小化滞后，同时保持平滑
    """
    n = ParameterValidator.validate_period(n, "HMA周期", min_period=1, max_period=200)
    
    half_n = int(n / 2)
    sqrt_n = int(np.sqrt(n))
    
    wma_half = wma(close, half_n)
    wma_full = wma(close, n)
    
    raw_hma = 2 * wma_half - wma_full
    hull = wma(raw_hma, sqrt_n)
    
    return hull


def supertrend(high: pd.Series, low: pd.Series, close: pd.Series,
               period: int = 10, multiplier: float = 3.0) -> Dict[str, pd.Series]:
    """
    SuperTrend指标（超级趋势）
    
    用途: 趋势跟踪，识别买卖信号
         价格 > SuperTrend: 上升趋势（做多）
         价格 < SuperTrend: 下降趋势（做空）
    
    参数:
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
        period: ATR周期
        multiplier: ATR倍数
    
    返回:
        dict: {'supertrend': 超级趋势线, 'trend': 趋势方向(1/0/-1)}
    """
    from indicators.indicators import atr
    
    atr_val = atr(pd.DataFrame({'high': high, 'low': low, 'close': close}), period)
    
    hl2 = (high + low) / 2
    upper_band = hl2 + multiplier * atr_val
    lower_band = hl2 - multiplier * atr_val
    
    supertrend = pd.Series(index=close.index, dtype=float)
    trend = pd.Series(index=close.index, dtype=int)
    
    current_trend = 0  # 1=up, -1=down
    for i in range(len(close)):
        if i == 0:
            supertrend.iloc[i] = upper_band.iloc[i]
            current_trend = 1
        else:
            prev_close = close.iloc[i-1]
            prev_supertrend = supertrend.iloc[i-1]
            
            if prev_close > prev_supertrend:
                current_trend = 1
                supertrend.iloc[i] = lower_band.iloc[i] if lower_band.iloc[i] > prev_supertrend else prev_supertrend
            else:
                current_trend = -1
                supertrend.iloc[i] = upper_band.iloc[i] if upper_band.iloc[i] < prev_supertrend else prev_supertrend
        
        trend.iloc[i] = current_trend
    
    return {'supertrend': supertrend, 'trend': trend}


def ichimoku(high: pd.Series, low: pd.Series, close: pd.Series,
              conversion_line_n: int = 9, base_line_n: int = 26,
              lagging_span_n: int = 52, displacement: int = 26) -> Dict[str, pd.Series]:
    """
    一目均衡表 Ichimoku Kinko Hyo
    
    用途: 多维度趋势分析
         价格在云上方: 上涨趋势
         价格在云下方: 下跌趋势
    
    参数:
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
        conversion_line_n: 转换线周期
        base_line_n: 基准线周期
        lagging_span_n: 迟行线周期
        displacement: 延迟跨度
    
    返回:
        dict: {'tenkan': 转换线, 'kijun': 基准线, 
                'senkou_a': 前行线A, 'senkou_b': 前行线B,
                'chikou': 迟行线}
    """
    # 转换线
    tenkan_sen = (high.rolling(conversion_line_n).max() + 
                  low.rolling(conversion_line_n).min()) / 2
    
    # 基准线
    kijun_sen = (high.rolling(base_line_n).max() + 
                  low.rolling(base_line_n).min()) / 2
    
    # 前行线A
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(displacement)
    
    # 前行线B
    senkou_span_b = ((high.rolling(lagging_span_n).max() + 
                      low.rolling(lagging_span_n).min()) / 2).shift(displacement)
    
    # 迟行线
    chikou_span = close.shift(-displacement)
    
    return {
        'tenkan': tenkan_sen,
        'kijun': kijun_sen,
        'senkou_a': senkou_span_a,
        'senkou_b': senkou_span_b,
        'chikou': chikou_span
    }


def donchian_channels(high: pd.Series, low: pd.Series, n: int = 20) -> Dict[str, pd.Series]:
    """
    唐奇安通道 Donchian Channels
    
    用途: 突破交易
         价格突破上轨: 买入信号
         价格跌破下轨: 卖出信号
    
    参数:
        high: 最高价序列
        low: 最低价序列
        n: 周期
    
    返回:
        dict: {'upper': 上轨, 'lower': 下轨, 'middle': 中轨}
    """
    upper_channel = high.rolling(n).max()
    lower_channel = low.rolling(n).min()
    middle_channel = (upper_channel + lower_channel) / 2
    
    return {
        'upper': upper_channel,
        'lower': lower_channel,
        'middle': middle_channel
    }


def pivot_points(high: pd.Series, low: pd.Series, close: pd.Series) -> Dict[str, pd.Series]:
    """
    枢轴点 Pivot Points
    
    用途: 识别支撑阻力位
    
    返回:
        dict: {'pivot': 枢轴点, 'r1': 阻力位1, 'r2': 阻力位2, 'r3': 阻力位3,
                's1': 支撑位1, 's2': 支撑位2, 's3': 支撑位3}
    """
    pivot = (high + low + close) / 3
    r1 = 2 * pivot - low
    s1 = 2 * pivot - high
    r2 = pivot + (high - low)
    s2 = pivot - (high - low)
    r3 = high + 2 * (pivot - low)
    s3 = low - 2 * (high - pivot)
    
    return {
        'pivot': pivot,
        'r1': r1,
        'r2': r2,
        'r3': r3,
        's1': s1,
        's2': s2,
        's3': s3
    }


def vwap_close(close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    简单版VWAP（仅基于收盘价）
    """
    from indicators.indicators import ema
    typical_price = close
    vwap_value = (typical_price * volume).cumsum() / volume.cumsum()
    return vwap_value


def aroon(high: pd.Series, low: pd.Series, n: int = 25) -> Dict[str, pd.Series]:
    """
    阿隆指标 Aroon
    
    用途: 趋势强度和方向
         Aroon Up > Aroon Down: 上升趋势
         Aroon Up < Aroon Down: 下降趋势
         两者都 < 50: 趋势可能反转
    
    参数:
        high: 最高价序列
        low: 最低价序列
        n: 周期
    
    返回:
        dict: {'aroon_up': 上升线, 'aroon_down': 下降线, 'oscillator': 振荡器}
    """
    aroon_up = 100 * high.rolling(n).apply(
        lambda x: x.argmax() / len(x) if not x.isna().all() else 0,
        raw=False
    )
    
    aroon_down = 100 * low.rolling(n).apply(
        lambda x: x.argmin() / len(x) if not x.isna().all() else 0,
        raw=False
    )
    
    oscillator = aroon_up - aroon_down
    
    return {
        'aroon_up': aroon_up,
        'aroon_down': aroon_down,
        'oscillator': oscillator
    }


def acceleration_bands(close: pd.Series, n: int = 20, 
                     up_factor: float = 1.06, down_factor: float = 0.94) -> Dict[str, pd.Series]:
    """
    加速带 Acceleration Bands
    
    用途: 识别价格加速和减速
    """
    from indicators.indicators import sma
    
    middle = sma(close, n)
    std = close.rolling(n).std()
    
    upper = middle * up_factor
    lower = middle * down_factor
    
    return {
        'upper': upper,
        'middle': middle,
        'lower': lower
    }


def envelope_sma(close: pd.Series, n: int = 20, pct: float = 0.05) -> Dict[str, pd.Series]:
    """
    SMA包络线 Envelope
    
    用途: 识别超买超卖
    """
    from indicators.indicators import sma
    
    middle = sma(close, n)
    upper = middle * (1 + pct)
    lower = middle * (1 - pct)
    
    return {
        'upper': upper,
        'middle': middle,
        'lower': lower
    }


def rsi_divergence(close: pd.Series, rsi_values: pd.Series, 
                   lookback: int = 20) -> pd.Series:
    """
    RSI背离检测
    
    用途: 识别潜在的反转点
    """
    def find_pivots(series, mode='peak'):
        """寻找峰值或谷值"""
        if mode == 'peak':
            return series.rolling(lookback, center=True).apply(
                lambda x: 1 if (x[len(x)//2] == x.max()) else 0,
                raw=True
            )
        else:
            return series.rolling(lookback, center=True).apply(
                lambda x: 1 if (x[len(x)//2] == x.min()) else 0,
                raw=True
            )
    
    close_peaks = find_pivots(close, 'peak')
    close_troughs = find_pivots(close, 'trough')
    rsi_peaks = find_pivots(rsi_values, 'peak')
    rsi_troughs = find_pivots(rsi_values, 'trough')
    
    # 看涨背离：价格创新低但RSI未创新低
    bullish_div = ((close_troughs.shift(1) > 0) & 
                  (close_troughs > 0) & 
                  (rsi_values.shift(lookback//2) < rsi_values))
    
    # 看跌背离：价格创新高但RSI未创新高
    bearish_div = ((close_peaks.shift(1) > 0) & 
                  (close_peaks > 0) & 
                  (rsi_values.shift(lookback//2) > rsi_values))
    
    return bullish_div.astype(int) - bearish_div.astype(int)


def volume_weighted_ma(close: pd.Series, volume: pd.Series, n: int = 20) -> pd.Series:
    """
    成交量加权移动平均 Volume Weighted MA
    
    用途: 成交量加权的价格平均
    """
    weighted_close = close * volume
    vwma = (weighted_close.rolling(n).sum() / volume.rolling(n).sum())
    return vwma


def money_flow_ratio(close: pd.Series, volume: pd.Series, n: int = 14) -> pd.Series:
    """
    资金流量比 Money Flow Ratio
    
    用途: 资金流入流出比
    """
    price_change = close.diff()
    positive_money_flow = (price_change > 0).astype(float) * close * volume
    negative_money_flow = (price_change < 0).astype(float) * close * volume
    
    mfr = (positive_money_flow.rolling(n).sum() / 
            SafeCalculator.safe_divide(
                negative_money_flow.rolling(n).sum(),
                1.0
            ))
    return mfr


def ease_of_movement(high: pd.Series, low: pd.Series, 
                   volume: pd.Series, n: int = 14) -> pd.Series:
    """
    EMV指标 Ease of Movement
    
    用途: 判断价格变动是否容易
    """
    distance = (high + low) / 2 - (high.shift(1) + low.shift(1)) / 2
    box_ratio = volume / 100000000 / (high - low)
    
    emv = SafeCalculator.safe_divide(
        distance,
        box_ratio,
        default=0
    )
    
    # 平滑EMV
    emv_ma = emv.rolling(n).mean()
    return emv_ma


def mass_index(high: pd.Series, low: pd.Series, n: int = 25, n2: int = 9) -> pd.Series:
    """
    Mass Index指标
    
    用途: 识别趋势反转
         MI > 27: 可能反转
    """
    from indicators.indicators import ema, atr
    
    high_low_range = high - low
    ema1 = ema(high_low_range, n)
    ema2 = ema(ema1, n)
    
    ratio = SafeCalculator.safe_divide(ema1, ema2, default=1)
    mass_idx = ratio.rolling(n2).sum()
    
    return mass_idx


def ultimate_oscillator(high: pd.Series, low: pd.Series, close: pd.Series,
                      n1: int = 7, n2: int = 14, n3: int = 28) -> pd.Series:
    """
    终极震荡指标 Ultimate Oscillator
    
    用途: 捕捉价格动量
    """
    bp = close - np.minimum(low, close.shift(1))
    tr = np.maximum(high - low, 
                  np.maximum(abs(high - close.shift(1)), 
                            abs(low - close.shift(1))))
    
    avg7 = bp.rolling(n1).sum() / tr.rolling(n1).sum()
    avg14 = bp.rolling(n2).sum() / tr.rolling(n2).sum()
    avg28 = bp.rolling(n3).sum() / tr.rolling(n3).sum()
    
    uo = 100 * (4 * avg7 + 2 * avg14 + avg28) / (4 + 2 + 1)
    return uo


def decycler(close: pd.Series, hp_period: int = 125) -> pd.Series:
    """
    去周期指标 Decycler
    
    用途: 消除周期性，保留趋势
    """
    from indicators.indicators import ema
    
    alpha = (1 - np.cos(2 * np.pi / hp_period)) / (1.585 * np.sqrt(2 * np.pi / hp_period))
    
    hp = pd.Series(index=close.index, dtype=float)
    hp.iloc[0] = 0
    
    for i in range(1, len(close)):
        hp.iloc[i] = (alpha / 2 * (close.iloc[i] - close.iloc[i-1] * 2 + close.iloc[i-2]) + 
                     (1 - alpha) * hp.iloc[i-1])
    
    decycler_val = close - hp
    return decycler_val


def zigzag(close: pd.Series, deviation: float = 0.05) -> Dict[str, pd.Series]:
    """
    ZigZag指标（摆动点检测）
    
    用途: 识别价格波峰波谷
    
    参数:
        close: 收盘价序列
        deviation: 最小变动百分比
    
    返回:
        dict: {'zigzag': ZigZag值, 'trend': 趋势方向}
    """
    zigzag = pd.Series(index=close.index, dtype=float)
    trend = pd.Series(index=close.index, dtype=int)
    
    last_peak_idx = 0
    last_trough_idx = 0
    last_peak_val = close.iloc[0]
    last_trough_val = close.iloc[0]
    
    current_trend = 0  # 1=up, -1=down, 0=flat
    
    for i in range(1, len(close)):
        current_val = close.iloc[i]
        change = (current_val - close.iloc[last_peak_idx]) / close.iloc[last_peak_idx]
        
        if abs(change) > deviation:
            if current_trend == 0 or current_trend == -1:
                # 检测到峰
                if current_val > last_peak_val * (1 + deviation):
                    zigzag.iloc[i] = current_val
                    last_peak_idx = i
                    last_peak_val = current_val
                    current_trend = 1
            else:
                # 检测到谷
                if current_val < last_trough_val * (1 - deviation):
                    zigzag.iloc[i] = current_val
                    last_trough_idx = i
                    last_trough_val = current_val
                    current_trend = -1
        
        trend.iloc[i] = current_trend
    
    return {'zigzag': zigzag, 'trend': trend}


def linear_regression_slope(close: pd.Series, n: int = 20) -> pd.Series:
    """
    线性回归斜率 Linear Regression Slope
    
    用途: 衡量趋势强度
    """
    def slope_calc(x):
        if len(x) < n:
            return 0
        y = x.values
        x_idx = np.arange(len(y))
        slope, _ = np.polyfit(x_idx, y, 1)
        return slope
    
    slope = close.rolling(n).apply(slope_calc, raw=False)
    return slope


def linear_regression_intercept(close: pd.Series, n: int = 20) -> pd.Series:
    """
    线性回归截距 Linear Regression Intercept
    
    用途: 趋势线位置
    """
    def intercept_calc(x):
        if len(x) < n:
            return 0
        y = x.values
        x_idx = np.arange(len(y))
        _, intercept = np.polyfit(x_idx, y, 1)
        return intercept
    
    intercept = close.rolling(n).apply(intercept_calc, raw=False)
    return intercept


def standardized_volume(volume: pd.Series, n: int = 20) -> pd.Series:
    """
    标准化成交量 Standardized Volume
    
    用途: 将成交量标准化为相对值
    """
    mean_vol = volume.rolling(n).mean()
    std_vol = volume.rolling(n).std()
    
    std_vol = SafeCalculator.clip_value(std_vol, 0.01, None)
    z_score = (volume - mean_vol) / std_vol
    
    return z_score


def volume_profile(close: pd.Series, volume: pd.Series, n_bins: int = 50) -> Dict[str, pd.Series]:
    """
    成交量分布 Volume Profile
    
    用途: 识别关键价位
    
    返回:
        dict: {'vwap': 成交量加权均价, 'poc': 价格重心}
    """
    # 计算VWAP
    vwap = (close * volume).cumsum() / volume.cumsum()
    
    # 计算POC（Price of Center）
    weighted_price = close * volume
    poc = weighted_price.rolling(n_bins).sum() / volume.rolling(n_bins).sum()
    
    return {'vwap': vwap, 'poc': poc}


def squeeze_momentum(df: pd.DataFrame, bb_length: int = 20, bb_mult: float = 2.0,
                    kc_length: int = 20, kc_mult: float = 1.5, atr_length: int = 20) -> Dict[str, pd.Series]:
    """
    挤压动量指标 Squeeze Momentum
    
    用途: 识别突破前的盘整期和突破
    """
    from indicators.indicators import sma, atr
    
    close = df['close']
    high = df['high']
    low = df['low']
    
    # 布林带
    bb_basis = sma(close, bb_length)
    bb_std = close.rolling(bb_length).std()
    bb_upper = bb_basis + bb_mult * bb_std
    bb_lower = bb_basis - bb_mult * bb_std
    
    # 肯特纳通道
    kc_basis = sma(close, kc_length)
    kc_atr = atr(pd.DataFrame({'high': high, 'low': low, 'close': close}), atr_length)
    kc_upper = kc_basis + kc_mult * kc_atr
    kc_lower = kc_basis - kc_mult * kc_atr
    
    # 挤压指标
    squeeze = (bb_upper - bb_lower) < (kc_upper - kc_lower)
    
    # 动量
    momentum = close - bb_basis
    
    return {'squeeze': squeeze.astype(int), 'momentum': momentum}


# 导入需要的函数
from indicators.indicators import ema, atr
