"""
趋势雷达选股系统 - 技术指标计算模块
包含常用的技术指标计算函数
"""
import numpy as np
import pandas as pd
from typing import Dict

from core.validators import (
    ValidationError,
    ParameterValidator,
    SafeCalculator,
    DataFrameValidator
)
from core.logger import get_indicator_logger

logger = get_indicator_logger()


def _validate_series_length(series: pd.Series, min_length: int, param_name: str = "Series"):
    """
    验证序列长度

    参数:
        series: 待验证的序列
        min_length: 最小长度
        param_name: 参数名称
    """
    if len(series) < min_length:
        logger.warning(f"{param_name}长度不足: {len(series)} < {min_length}")


def sma(s: pd.Series, n: int) -> pd.Series:
    """简单移动平均线 Simple Moving Average"""
    # 验证参数
    n = ParameterValidator.validate_period(n, "SMA周期", min_period=1, max_period=1000)

    if len(s) < n:
        logger.debug(f"SMA: 数据长度{len(s)}小于周期{n}")

    return s.rolling(n, min_periods=n).mean()


def true_range(high, low, prev_close):
    """真实波幅 True Range"""
    # 验证输入长度
    if isinstance(high, pd.Series):
        _validate_series_length(high, 2, "True Range high")

    return np.maximum(high - low, np.maximum((high - prev_close).abs(), (low - prev_close).abs()))


def atr(df: pd.DataFrame, n: int) -> pd.Series:
    """平均真实波幅 Average True Range"""
    n = ParameterValidator.validate_period(n, "ATR周期", min_period=1, max_period=100)

    df = DataFrameValidator.validate_dataframe(df, ['high', 'low', 'close'], "ATR输入")
    prev_close = df["close"].shift(1)
    tr = true_range(df["high"], df["low"], prev_close)
    return tr.rolling(n, min_periods=n).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    """相对强弱指数 Relative Strength Index"""
    n = ParameterValidator.validate_period(n, "RSI周期", min_period=2, max_period=100)

    if len(close) < n + 1:
        logger.debug(f"RSI: 数据长度{len(close)}不足{n+1}")

    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.rolling(n, min_periods=n).mean()
    roll_down = down.rolling(n, min_periods=n).mean()

    # 使用安全除法
    rs = SafeCalculator.safe_divide(roll_up, roll_down, default=np.nan)
    return 100 - (100 / (1 + rs))


def ema(s: pd.Series, n: int) -> pd.Series:
    """指数移动平均线 Exponential Moving Average"""
    n = ParameterValidator.validate_period(n, "EMA周期", min_period=1, max_period=1000)
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
    n = ParameterValidator.validate_period(n, "布林带周期", min_period=2, max_period=200)
    num_std = ParameterValidator.validate_positive_number(num_std, "布林带标准差倍数")

    middle = sma(close, n)
    std = close.rolling(n, min_periods=n).std()
    upper = middle + num_std * std
    lower = middle - num_std * std

    # 安全计算宽度
    width = SafeCalculator.safe_divide(upper - lower, middle, default=np.nan)

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
    n = ParameterValidator.validate_period(n, "ADX周期", min_period=2, max_period=100)

    if len(high) < n + 1:
        logger.debug(f"ADX: 数据长度{len(high)}不足{n+1}")

    plus_dm = high.diff()
    minus_dm = low.diff()

    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    minus_dm = minus_dm.abs()

    tr = true_range(high, low, close.shift(1))
    atr_val = atr(pd.DataFrame({'high': high, 'low': low, 'close': close}), n)

    plus_di = SafeCalculator.safe_divide(
        100 * plus_dm.rolling(n, min_periods=n).mean(),
        atr_val
    )
    minus_di = SafeCalculator.safe_divide(
        100 * minus_dm.rolling(n, min_periods=n).mean(),
        atr_val
    )

    dx = SafeCalculator.safe_divide(
        100 * abs(plus_di - minus_di),
        plus_di + minus_di
    )
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
    n = ParameterValidator.validate_period(n, "KDJ周期", min_period=1, max_period=100)
    m1 = ParameterValidator.validate_period(m1, "KDJ m1", min_period=1, max_period=50)
    m2 = ParameterValidator.validate_period(m2, "KDJ m2", min_period=1, max_period=50)

    if len(high) < n:
        logger.debug(f"KDJ: 数据长度{len(high)}不足{n}")

    lowest_low = low.rolling(n, min_periods=n).min()
    highest_high = high.rolling(n, min_periods=n).max()

    # 安全计算RSV
    rsv = SafeCalculator.safe_divide(
        100 * (close - lowest_low),
        highest_high - lowest_low,
        default=50  # 缺失时使用中性值
    )

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
    n = ParameterValidator.validate_period(n, "Williams %R周期", min_period=1, max_period=100)

    highest_high = high.rolling(n, min_periods=n).max()
    lowest_low = low.rolling(n, min_periods=n).min()

    # 安全计算Williams %R
    wr = SafeCalculator.safe_divide(
        -100 * (highest_high - close),
        highest_high - lowest_low,
        default=-50
    )
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
    n = ParameterValidator.validate_period(n, "MFI周期", min_period=2, max_period=100)

    if len(high) < n + 1:
        logger.debug(f"MFI: 数据长度{len(high)}不足{n+1}")

    typical_price = (high + low + close) / 3
    money_flow = typical_price * volume

    positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
    negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)

    # 安全计算MFI
    mfr = SafeCalculator.safe_divide(
        positive_flow.rolling(n, min_periods=n).sum(),
        negative_flow.rolling(n, min_periods=n).sum(),
        default=1.0
    )
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
    n = ParameterValidator.validate_period(n, "价格位置周期", min_period=2, max_period=500)

    highest_high = high.rolling(n, min_periods=n).max()
    lowest_low = low.rolling(n, min_periods=n).min()

    # 安全计算位置，并限制在0-1范围内
    position = SafeCalculator.safe_divide(
        close - lowest_low,
        highest_high - lowest_low,
        default=0.5
    )
    return SafeCalculator.clip_value(position, 0, 1)


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


def dpo(close: pd.Series, n: int = 20) -> pd.Series:
    """
    去趋势价格振荡 Detrended Price Oscillator

    用途: 消除趋势影响，识别周期性波动
         DPO > 0: 价格高于平均水平
         DPO < 0: 价格低于平均水平

    参数:
        close: 收盘价序列
        n: 周期长度

    返回:
        DPO值序列
    """
    sma_n = close.rolling(n).mean()
    dpo_value = close.shift(n//2 + 1) - sma_n
    return dpo_value


def trix(close: pd.Series, n: int = 15, signal: int = 9) -> dict:
    """
    TRIX指标（三重指数平滑平均线）

    用途: 过滤短期波动，识别趋势转折
         TRIX上穿信号线: 买入信号
         TRIX下穿信号线: 卖出信号

    参数:
        close: 收盘价序列
        n: TRIX周期
        signal: 信号线周期

    返回:
        {'trix': Series, 'signal': Series, 'hist': Series}
    """
    ema1 = close.ewm(span=n, adjust=False).mean()
    ema2 = ema1.ewm(span=n, adjust=False).mean()
    ema3 = ema2.ewm(span=n, adjust=False).mean()

    trix_value = 100 * (ema3.diff(1) / ema3.shift(1))
    signal_line = trix_value.ewm(span=signal, adjust=False).mean()

    return {
        'trix': trix_value,
        'signal': signal_line,
        'hist': trix_value - signal_line
    }


def parabolic_sar(high: pd.Series, low: pd.Series,
                 af: float = 0.02, max_af: float = 0.2) -> pd.Series:
    """
    抛物线转向指标 Parabolic SAR

    用途: 趋势跟踪止损，设置动态止损位
         价格 > SAR: 多头趋势
         价格 < SAR: 空头趋势

    参数:
        high: 最高价序列
        low: 最低价序列
        af: 加速因子初始值
        max_af: 加速因子最大值

    返回:
        SAR值序列
    """
    sar = np.zeros(len(high))
    trend = 1  # 1为多头，-1为空头
    ep = high[0]  # 极值点
    af_val = af   # 加速因子

    sar[0] = low[0]

    for i in range(1, len(high)):
        if trend == 1:  # 多头
            sar[i] = sar[i-1] + af_val * (ep - sar[i-1])

            if low[i] < sar[i]:
                trend = -1
                sar[i] = ep
                ep = low[i]
                af_val = af
            else:
                if high[i] > ep:
                    ep = high[i]
                    af_val = min(af_val + af, max_af)
        else:  # 空头
            sar[i] = sar[i-1] + af_val * (ep - sar[i-1])

            if high[i] > sar[i]:
                trend = 1
                sar[i] = ep
                ep = high[i]
                af_val = af
            else:
                if low[i] < ep:
                    ep = low[i]
                    af_val = min(af_val + af, max_af)

    return pd.Series(sar, index=high.index)


def vwap(df: pd.DataFrame) -> pd.Series:
    """
    成交量加权平均价 Volume Weighted Average Price

    用途: 判断当日平均成本，识别偏离程度
         价格 > VWAP: 买方强势
         价格 < VWAP: 卖方强势

    参数:
        df: 包含high, low, close, volume的DataFrame

    返回:
        VWAP值序列
    """
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    vwap_value = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
    return vwap_value


def cci(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 20) -> pd.Series:
    """
    顺势指标 Commodity Channel Index

    用途: 识别超买超卖
         CCI > 100: 超买
         CCI < -100: 超卖

    参数:
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
        n: 周期

    返回:
        CCI值序列
    """
    typical_price = (high + low + close) / 3
    sma_tp = typical_price.rolling(n).mean()
    mad = typical_price.rolling(n).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)

    cci_value = (typical_price - sma_tp) / (0.015 * mad)
    return cci_value


def stoch_rsi(close: pd.Series, n: int = 14, d: int = 3, k: int = 3) -> dict:
    """
    随机RSI指标 Stochastic RSI

    用途: 识别超买超卖
         StochRSI > 0.8: 超买
         StochRSI < 0.2: 超卖

    参数:
        close: 收盘价序列
        n: RSI周期
        d: %D平滑周期
        k: %K平滑周期

    返回:
        {'stoch_rsi': Series, 'k': Series, 'd': Series}
    """
    # 计算RSI
    rsi_val = rsi(close, n)

    # Stochastic RSI
    stoch_rsi = (rsi_val - rsi_val.rolling(n).min()) / \
                (rsi_val.rolling(n).max() - rsi_val.rolling(n).min())

    # %K和%D
    k_line = stoch_rsi.rolling(k).mean()
    d_line = k_line.rolling(d).mean()

    return {
        'stoch_rsi': stoch_rsi,
        'k': k_line,
        'd': d_line
    }


def vpt(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """
    量价趋势指标 Volume Price Trend

    用途: 衡量买卖压力
         VPT上升: 买方压力
         VPT下降: 卖方压力

    参数:
        df: 包含close和volume的DataFrame
        n: 平滑周期

    返回:
        VPT值序列
    """
    pct_change = df['close'].pct_change()
    vpt_raw = (pct_change * df['volume']).cumsum()
    vpt_value = vpt_raw.rolling(n).mean()
    return vpt_value


def vortex(df: pd.DataFrame, n: int = 14) -> dict:
    """
    Vortex指标

    用途: 识别趋势方向
         VI+ > VI-: 上升趋势
         VI+ < VI-: 下降趋势

    参数:
        df: 包含high, low, close的DataFrame
        n: 周期

    返回:
        {'vi_plus': Series, 'vi_minus': Series}
    """
    high = df['high']
    low = df['low']
    close = df['close']

    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)

    vm_plus = (high - low.shift(1)).abs()
    vm_minus = (low - high.shift(1)).abs()

    vi_plus = vm_plus.rolling(n).sum() / tr.rolling(n).sum()
    vi_minus = vm_minus.rolling(n).sum() / tr.rolling(n).sum()

    return {
        'vi_plus': vi_plus,
        'vi_minus': vi_minus
    }


def fisher_transform(df: pd.DataFrame, n: int = 10) -> dict:
    """
    Fisher变换指标

    用途: 将价格分布转换为接近正态分布，识别转折点
         Fisher > 0: 趋势向上
         Fisher < 0: 趋势向下

    参数:
        df: 包含high和low的DataFrame
        n: 周期

    返回:
        {'fisher': Series, 'signal': Series}
    """
    high = df['high']
    low = df['low']

    mid = (high + low) / 2
    nd_low = mid.rolling(n).min()
    nd_high = mid.rolling(n).max()

    # 避免除零
    nd_high = nd_high.replace(0, np.nan)
    value = 2 * ((mid - nd_low) / (nd_high - nd_low) - 0.5)
    value = value.clip(-0.999, 0.999)

    fisher = 0.5 * np.log((1 + value) / (1 - value))
    signal = fisher.shift(1)

    return {
        'fisher': fisher,
        'signal': signal
    }
