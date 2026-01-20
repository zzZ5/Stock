"""
趋势雷达选股系统 - 技术指标计算模块
包含常用的技术指标计算函数
"""
import numpy as np
import pandas as pd
from typing import Dict


def sma(s: pd.Series, n: int) -> pd.Series:
    """简单移动平均线 Simple Moving Average"""
    return s.rolling(n, min_periods=n).mean()


def true_range(high, low, prev_close):
    """真实波幅 True Range"""
    return np.maximum(high - low, np.maximum((high - prev_close).abs(), (low - prev_close).abs()))


def atr(df: pd.DataFrame, n: int) -> pd.Series:
    """平均真实波幅 Average True Range"""
    prev_close = df["close"].shift(1)
    tr = true_range(df["high"], df["low"], prev_close)
    return tr.rolling(n, min_periods=n).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    """相对强弱指数 Relative Strength Index"""
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.rolling(n, min_periods=n).mean()
    roll_down = down.rolling(n, min_periods=n).mean()
    rs = roll_up / (roll_down.replace(0, np.nan))
    return 100 - (100 / (1 + rs))


def ema(s: pd.Series, n: int) -> pd.Series:
    """指数移动平均线 Exponential Moving Average"""
    return s.ewm(span=n, adjust=False).mean()


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> dict:
    """
    MACD指标（移动平均收敛发散指标）

    返回:
        dict: {'macd': Series, 'signal': Series, 'hist': Series}
    """
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line

    return {
        'macd': macd_line,
        'signal': signal_line,
        'hist': hist
    }


def bollinger_bands(close: pd.Series, n: int = 20, num_std: float = 2.0) -> dict:
    """
    布林带 Bollinger Bands

    参数:
        close: 收盘价序列
        n: 周期
        num_std: 标准差倍数

    返回:
        dict: {'upper': Series, 'middle': Series, 'lower': Series, 'width': Series}
    """
    middle = sma(close, n)
    std = close.rolling(n, min_periods=n).std()
    upper = middle + num_std * std
    lower = middle - num_std * std
    width = (upper - lower) / middle

    return {
        'upper': upper,
        'middle': middle,
        'lower': lower,
        'width': width
    }


def adx(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    """
    平均趋向指标 Average Directional Index

    用途: 判断趋势强度
         ADX > 25: 强趋势
         ADX < 20: 弱趋势/震荡

    参数:
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
        n: 计算周期

    返回:
        ADX值序列
    """
    plus_dm = high.diff()
    minus_dm = low.diff()

    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    minus_dm = minus_dm.abs()

    tr = true_range(high, low, close.shift(1))
    atr_val = atr(pd.DataFrame({'high': high, 'low': low, 'close': close}), n)

    plus_di = 100 * (plus_dm.rolling(n, min_periods=n).mean() / atr_val)
    minus_di = 100 * (minus_dm.rolling(n, min_periods=n).mean() / atr_val)

    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx_val = dx.rolling(n, min_periods=n).mean()

    return adx_val


def kdj(high: pd.Series, low: pd.Series, close: pd.Series,
        n: int = 9, m1: int = 3, m2: int = 3) -> Dict[str, pd.Series]:
    """
    KDJ指标（随机指标）

    用途: 超买超卖判断
         J > 100: 超买
         J < 0: 超卖
         K上穿D: 金叉（买入信号）
         K下穿D: 死叉（卖出信号）

    参数:
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
        n: RSV周期
        m1: K值平滑周期
        m2: D值平滑周期

    返回:
        dict: {'k': K线, 'd': D线, 'j': J线}
    """
    lowest_low = low.rolling(n, min_periods=n).min()
    highest_high = high.rolling(n, min_periods=n).max()

    rsv = 100 * (close - lowest_low) / (highest_high - lowest_low)

    k = rsv.ewm(com=m1-1, adjust=False).mean()
    d = k.ewm(com=m2-1, adjust=False).mean()
    j = 3 * k - 2 * d

    return {'k': k, 'd': d, 'j': j}


def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    """
    威廉指标 Williams %R

    用途: 超买超卖判断
         WR < -80: 超买
         WR > -20: 超卖

    参数:
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
        n: 计算周期

    返回:
        Williams %R值序列
    """
    highest_high = high.rolling(n, min_periods=n).max()
    lowest_low = low.rolling(n, min_periods=n).min()

    wr = -100 * (highest_high - close) / (highest_high - lowest_low)
    return wr


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    能量潮指标 On-Balance Volume

    用途: 资金流向判断
         OBV创新高: 资金持续流入
         OBV与价格背离: 反转信号

    参数:
        close: 收盘价序列
        volume: 成交量序列

    返回:
        OBV值序列
    """
    obv_val = (volume * np.sign(close.diff())).cumsum()
    return obv_val


def money_flow_index(high: pd.Series, low: pd.Series, close: pd.Series,
                     volume: pd.Series, n: int = 14) -> pd.Series:
    """
    资金流量指标 Money Flow Index

    用途: 识别资金进出情况
         MFI < 20: 超卖
         MFI > 80: 超买

    参数:
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
        volume: 成交量序列
        n: 计算周期

    返回:
        MFI值序列
    """
    typical_price = (high + low + close) / 3
    money_flow = typical_price * volume

    positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
    negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)

    mfr = positive_flow.rolling(n, min_periods=n).sum() / negative_flow.rolling(n, min_periods=n).sum()
    mfi = 100 - (100 / (1 + mfr))

    return mfi


def chandelier_exit(high: pd.Series, low: pd.Series, close: pd.Series,
                    atr_n: int = 22, mult: float = 3.0) -> Dict[str, pd.Series]:
    """
    吊灯止损指标 Chandelier Exit

    用途: 动态止损位设置
         多头止损: 最高价 - ATR * 多重
         空头止损: 最低价 + ATR * 多重

    参数:
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
        atr_n: ATR计算周期
        mult: ATR倍数

    返回:
        dict: {'long_exit': 多头止损, 'short_exit': 空头止损}
    """
    atr_val = atr(pd.DataFrame({'high': high, 'low': low, 'close': close}), atr_n)

    long_exit = high.rolling(atr_n, min_periods=atr_n).max() - mult * atr_val
    short_exit = low.rolling(atr_n, min_periods=atr_n).min() + mult * atr_val

    return {'long_exit': long_exit, 'short_exit': short_exit}


def volatility_ratio(close: pd.Series, short_n: int = 5, long_n: int = 20) -> pd.Series:
    """
    波动率比率 Volatility Ratio

    用途: 识别异常波动（突破信号辅助）
         VR > 1.5: 波动率异常放大，可能突破

    参数:
        close: 收盘价序列
        short_n: 短期波动率周期
        long_n: 长期波动率周期

    返回:
        波动率比率序列
    """
    vol_short = close.pct_change().rolling(short_n, min_periods=short_n).std()
    vol_long = close.pct_change().rolling(long_n, min_periods=long_n).std()

    vr = vol_short / vol_long
    return vr


def price_position(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 20) -> pd.Series:
    """
    价格位置 Price Position

    用途: 判断价格在近期区间的位置
         0-1范围，1为区间最高，0为区间最低
         > 0.8: 高位，避免追高
         < 0.2: 低位，可能超卖

    参数:
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
        n: 回看周期

    返回:
        价格位置序列 (0-1)
    """
    highest_high = high.rolling(n, min_periods=n).max()
    lowest_low = low.rolling(n, min_periods=n).min()

    position = (close - lowest_low) / (highest_high - lowest_low)
    return position


def volume_price_trend(close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    量价趋势 Volume Price Trend (VPT)

    用途: 量价配合度判断
         VPT上升且价格上升: 量价齐升
         VPT下降且价格上升: 量价背离

    参数:
        close: 收盘价序列
        volume: 成交量序列

    返回:
        VPT值序列
    """
    vpt = ((close.diff() / close.shift(1)) * volume).cumsum()
    return vpt


def cci(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 20) -> pd.Series:
    """
    顺势指标 Commodity Channel Index

    用途: 趋势跟随
         CCI > 100: 超买
         CCI < -100: 超卖

    参数:
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
        n: 计算周期

    返回:
        CCI值序列
    """
    tp = (high + low + close) / 3
    sma_tp = tp.rolling(n, min_periods=n).mean()
    mad = tp.rolling(n, min_periods=n).apply(lambda x: np.abs(x - x.mean()).mean())

    cci_val = (tp - sma_tp) / (0.015 * mad)
    return cci_val


def momentum(close: pd.Series, n: int = 10) -> pd.Series:
    """
    动量指标 Momentum

    用途: 衡量价格变化速度
         Momentum > 0: 上涨动能
         Momentum < 0: 下跌动能

    参数:
        close: 收盘价序列
        n: 动量周期

    返回:
        动量值序列
    """
    mom = close - close.shift(n)
    return mom


def roc(close: pd.Series, n: int = 12) -> pd.Series:
    """
    变动率指标 Rate of Change

    用途: 衡量价格变化率
         ROC > 0: 上涨
         ROC < 0: 下跌

    参数:
        close: 收盘价序列
        n: 计算周期

    返回:
        ROC值序列（百分比）
    """
    roc_val = (close - close.shift(n)) / close.shift(n) * 100
    return roc_val
