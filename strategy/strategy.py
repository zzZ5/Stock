"""
趋势雷达选股系统 - 选股策略模块
包含股票筛选、评分、策略逻辑等核心功能
"""
import pandas as pd
import numpy as np

from config.settings import settings
from indicators.indicators import (
    sma, atr, rsi,
    adx, kdj, williams_r, price_position
)
from core.validators import (
    DataFrameValidator,
    ValidationError
)
from core.logger import get_strategy_logger

logger = get_strategy_logger()


class StockStrategy:
    """股票选股策略"""

    def __init__(self, basic_df: pd.DataFrame = None):
        """
        初始化策略

        参数:
            basic_df: 股票基础信息DataFrame
        """
        if basic_df is not None:
            try:
                basic_df = DataFrameValidator.validate_dataframe(
                    basic_df,
                    ['ts_code', 'name'],
                    '股票基础信息'
                )
            except ValidationError as e:
                logger.warning(f"股票基础信息验证失败: {e}")
                basic_df = pd.DataFrame()

        self.basic = basic_df if basic_df is not None else pd.DataFrame()
        self.weekly_data = pd.DataFrame()
        self.monthly_data = pd.DataFrame()

    def set_multi_timeframe_data(self, weekly_data: pd.DataFrame = None,
                                  monthly_data: pd.DataFrame = None):
        """
        设置多周期数据（周线、月线）

        参数:
            weekly_data: 周线数据DataFrame
            monthly_data: 月线数据DataFrame
        """
        if weekly_data is not None:
            self.weekly_data = weekly_data
        if monthly_data is not None:
            self.monthly_data = monthly_data

    def filter_basic(self, basic: pd.DataFrame, trade_date: str,
                    trade_dates: list[str] = None) -> pd.DataFrame:
        """
        基于stock_basic做过滤

        参数:
            basic: 股票基础信息
            trade_date: 交易日期
            trade_dates: 交易日列表（用于计算上市天数）

        返回:
            过滤后的DataFrame
        """
        basic = basic.copy()

        # 剔除 ST、退市整理等
        name = basic["name"].astype(str)
        bad_name = name.str.contains("ST|\\*ST|退|停", regex=True)
        basic = basic[~bad_name].copy()

        # 次新过滤
        basic["list_date"] = pd.to_datetime(basic["list_date"],
                                         format="%Y%m%d", errors="coerce")
        today = pd.to_datetime(trade_date, format="%Y%m%d")

        min_list_days = settings.MIN_LIST_DAYS if hasattr(settings, 'MIN_LIST_DAYS') else 120

        if trade_dates is not None:
            # 使用实际交易日计算
            trade_date_set = set(trade_dates)
            basic = basic[basic["list_date"].apply(
                lambda x: len([d for d in trade_dates
                             if d >= x.strftime("%Y%m%d") and d <= trade_date])
            ) >= min_list_days].copy()
        else:
            # 回退方案：用日历天*1.6粗略估算
            basic = basic[(today - basic["list_date"]).dt.days >= min_list_days * 1.6].copy()

        return basic

    def is_one_word_limitup(self, row) -> bool:
        """判断是否为一字涨停板"""
        if pd.isna(row.get("pct_chg")):
            return False
        if float(row.get("pct_chg", 0)) < 9.5:
            return False
        if not (float(row["high"]) == float(row["low"]) == float(row["close"])):
            return False
        return float(row.get("amount", 0)) < 1000  # 1000千元=100万

    def analyze_stocks(self, hist: pd.DataFrame, market_ok: bool,
                     progress_callback=None) -> pd.DataFrame:
        """
        分析所有股票，计算得分和推荐理由

        参数:
            hist: 历史行情数据
            market_ok: 市场环境是否良好
            progress_callback: 进度回调函数

        返回:
            包含得分、理由等的DataFrame
        """
        # 验证输入数据
        try:
            hist = DataFrameValidator.validate_dataframe(
                hist,
                ['ts_code', 'trade_date', 'open', 'high', 'low', 'close', 'amount'],
                '历史行情数据'
            )
            hist = DataFrameValidator.validate_numeric_columns(
                hist,
                ['open', 'high', 'low', 'close', 'amount'],
                '历史行情数据'
            )
        except ValidationError as e:
            logger.error(f"历史行情数据验证失败: {e}")
            return pd.DataFrame()

        out_rows = []
        hist_sorted = hist.sort_values(["ts_code", "trade_date"])
        unique_codes = hist_sorted["ts_code"].unique()

        logger.info(f"开始分析股票（共 {len(unique_codes)} 只）...")

        for i, code in enumerate(unique_codes):
            try:
                g = hist_sorted[hist_sorted["ts_code"] == code]

                if len(g) < max(settings.BREAKOUT_N, settings.MA_SLOW, settings.VOL_LOOKBACK, settings.ATR_N) + 6:
                    continue

                result = self._analyze_single_stock(g, code, market_ok)
                if result is not None:
                    out_rows.append(result)

                if progress_callback:
                    progress_callback(1)

            except Exception as e:
                logger.error(f"分析股票{code}时发生错误: {e}")
                continue

        if not out_rows:
            logger.warning("没有符合条件的股票")
            return pd.DataFrame()

        out = pd.DataFrame(out_rows)
        out = out.sort_values(["candidate", "score"],
                             ascending=[False, False])
        logger.info(f"分析完成，共{len(out)}只股票符合条件")
        return out

    def _analyze_single_stock(self, stock_data: pd.DataFrame,
                            code: str, market_ok: bool) -> dict:
        """分析单只股票"""
        try:
            # 验证股票数据
            stock_data = DataFrameValidator.validate_dataframe(
                stock_data,
                ['open', 'high', 'low', 'close', 'amount'],
                f'{code}股票数据'
            )

            last = stock_data.iloc[-1]
            close = stock_data["close"].astype(float)
            high = stock_data["high"].astype(float)
            low = stock_data["low"].astype(float)
            amount = stock_data["amount"].astype(float)

            # 验证至少有一个有效价格
            if close.isna().all() or high.isna().all() or low.isna().all():
                logger.warning(f"{code}: 价格数据全为NaN")
                return None

        except Exception as e:
            logger.error(f"{code}数据验证失败: {e}")
            return None

        # 计算基础指标
        ma20 = sma(close, settings.MA_FAST).iloc[-1]
        ma60 = sma(close, settings.MA_SLOW).iloc[-1]
        ma60_5ago = sma(close, settings.MA_SLOW).iloc[-6]
        ma60_slope = ma60 - ma60_5ago

        rsi14 = rsi(close, 14).iloc[-1]
        atr14 = atr(stock_data[["high", "low", "close"]].assign(
            high=high, low=low, close=close), settings.ATR_N).iloc[-1]

        # 计算新指标
        adx_val = adx(high, low, close).iloc[-1]
        kdj_data = kdj(high, low, close)
        k_val, d_val, j_val = kdj_data['k'].iloc[-1], kdj_data['d'].iloc[-1], kdj_data['j'].iloc[-1]
        wr_val = williams_r(high, low, close).iloc[-1]
        pos_val = price_position(high, low, close).iloc[-1]

        # 多周期突破判断
        breakout_info = self._check_multi_timeframe_breakout(
            code, close, high, float(last["close"]), float(last["high"])
        )

        # 使用多周期突破信息
        breakout = breakout_info['daily_breakout']
        weekly_breakout = breakout_info['weekly_breakout']
        monthly_breakout = breakout_info['monthly_breakout']

        # 综合突破信号（多周期模式）
        if settings.MULTI_TIMEFRAME_MODE:
            # 至少2个周期突破才算强信号
            breakout_signals = sum([breakout, weekly_breakout, monthly_breakout])
            breakout = breakout_signals >= 2
            breakout_price_close = breakout_info['daily_breakout_price']
        else:
            breakout_price_close = breakout_info['daily_breakout_price']

        # 量能确认
        avg_amt20 = amount.iloc[-settings.VOL_LOOKBACK:].mean()
        if avg_amt20 <= 0:
            vol_ratio = np.nan
            vol_confirm = False
        else:
            vol_ratio = float(last["amount"]) / avg_amt20
            vol_confirm = (np.isfinite(vol_ratio) and vol_ratio >= settings.VOL_CONFIRM_MULT)

        # 趋势结构
        trend_struct = (ma20 > ma60) and (ma60_slope > 0)

        # 新指标判断
        trend_strong = np.isfinite(adx_val) and adx_val > 25  # ADX>25为强趋势
        not_overbought = not (np.isfinite(j_val) and j_val > 100)  # J值不过热
        not_too_high = not (np.isfinite(pos_val) and pos_val > 0.9)  # 价格不在过高位置

        # 20日收益
        if len(close) >= 21 and close.iloc[-21] > 0:
            ret20 = (close.iloc[-1] / close.iloc[-21] - 1)
        else:
            ret20 = np.nan

        # 过滤
        try:
            if float(last["close"]) < settings.MIN_PRICE:
                return None
            if not np.isfinite(avg_amt20) or avg_amt20 < settings.MIN_AVG_AMOUNT_20D:
                return None
            if settings.EXCLUDE_ONE_WORD_LIMITUP and self.is_one_word_limitup(last):
                return None
        except Exception as e:
            logger.error(f"{code}过滤条件检查失败: {e}")
            return None

        # 候选/观察逻辑（增强版：加入新指标判断）
        signal_hits = sum([breakout, trend_struct, vol_confirm, trend_strong])
        is_candidate = (signal_hits >= 4) and not_overbought and not_too_high and \
                      (np.isfinite(rsi14) and rsi14 <= settings.RSI_MAX)

        # 安全计算距离突破的距离
        if float(last["close"]) > 0 and np.isfinite(breakout_price_close):
            dist_to_break = (float(breakout_price_close) / float(last["close"]) - 1)
        else:
            dist_to_break = np.nan

        is_watch = (signal_hits >= 3) and \
                  (np.isfinite(dist_to_break) and dist_to_break <= 0.015) and \
                  not_overbought

        if not (is_candidate or is_watch):
            return None

        # 止损价
        hard_stop = float(last["close"]) * (1 - settings.MAX_LOSS_PCT)
        if np.isfinite(atr14) and atr14 > 0:
            atr_stop = float(last["close"]) - settings.ATR_MULT * float(atr14)
        else:
            atr_stop = hard_stop
        stop_price = max(hard_stop, atr_stop)

        # 安全计算止损百分比
        if float(last["close"]) > 0:
            stop_pct = (stop_price / float(last["close"]) - 1)
        else:
            logger.warning(f"{code}: 入场价格为0")
            stop_pct = -settings.MAX_LOSS_PCT

        # 评分（增强版）
        total, reasons = self._calculate_score(
            breakout, trend_struct, breakout_price_close, float(last["close"]),
            vol_ratio, vol_confirm, rsi14, close, stop_pct,
            market_ok, dist_to_break, ma20, ma60, ma60_slope,
            adx_val, j_val, pos_val, trend_strong,
            weekly_breakout, monthly_breakout, breakout_info
        )

        # 获取名称和行业
        stock_name = self._get_stock_name(code)
        industry = self._get_stock_industry(code)

        return {
            "ts_code": code,
            "name": stock_name,
            "industry": industry,
            "close": float(last["close"]),
            "high_today": float(last["high"]),
            "breakout_price": float(breakout_price_close),
            "dist_to_break_pct": float(dist_to_break * 100) if np.isfinite(dist_to_break) else np.nan,
            "score": total,
            "candidate": bool(is_candidate),
            "watch": bool(is_watch and not is_candidate),
            "stop_price": stop_price,
            "reasons": reasons,
            "adx": float(adx_val) if np.isfinite(adx_val) else np.nan,
            "kdj_j": float(j_val) if np.isfinite(j_val) else np.nan,
            "price_position": float(pos_val) if np.isfinite(pos_val) else np.nan,
            "weekly_breakout": bool(weekly_breakout),
            "monthly_breakout": bool(monthly_breakout),
        }

    def _check_multi_timeframe_breakout(self, code: str, daily_close, daily_high,
                                         entry_price: float, high_today: float) -> dict:
        """
        检查多周期突破（日、周、月）

        参数:
            code: 股票代码
            daily_close: 日线收盘价序列
            daily_high: 日线最高价序列
            entry_price: 当前收盘价
            high_today: 当日最高价

        返回:
            包含各周期突破信息的字典
        """
        result = {
            'daily_breakout': False,
            'daily_breakout_price': 0.0,
            'weekly_breakout': False,
            'weekly_breakout_price': 0.0,
            'monthly_breakout': False,
            'monthly_breakout_price': 0.0,
        }

        # 日线突破
        daily_breakout_price_close = daily_close.iloc[-settings.BREAKOUT_N:].max()
        daily_breakout_price_high = daily_high.iloc[-settings.BREAKOUT_N:].max()
        result['daily_breakout'] = (entry_price >= float(daily_breakout_price_close)) and \
                                   (high_today >= float(daily_breakout_price_high))
        result['daily_breakout_price'] = float(daily_breakout_price_close)

        # 周线突破
        if not self.weekly_data.empty:
            weekly_data = self.weekly_data[self.weekly_data["ts_code"] == code]
            if not weekly_data.empty and len(weekly_data) >= settings.WEEKLY_BREAKOUT_N + 2:
                weekly_close = weekly_data["close"].astype(float)
                weekly_high = weekly_data["high"].astype(float)
                weekly_breakout_price_close = weekly_close.iloc[-settings.WEEKLY_BREAKOUT_N:].max()
                weekly_breakout_price_high = weekly_high.iloc[-settings.WEEKLY_BREAKOUT_N:].max()
                result['weekly_breakout'] = (entry_price >= float(weekly_breakout_price_close)) and \
                                           (high_today >= float(weekly_breakout_price_high))
                result['weekly_breakout_price'] = float(weekly_breakout_price_close)

        # 月线突破
        if not self.monthly_data.empty:
            monthly_data = self.monthly_data[self.monthly_data["ts_code"] == code]
            if not monthly_data.empty and len(monthly_data) >= settings.MONTHLY_BREAKOUT_N + 2:
                monthly_close = monthly_data["close"].astype(float)
                monthly_high = monthly_data["high"].astype(float)
                monthly_breakout_price_close = monthly_close.iloc[-settings.MONTHLY_BREAKOUT_N:].max()
                monthly_breakout_price_high = monthly_high.iloc[-settings.MONTHLY_BREAKOUT_N:].max()
                result['monthly_breakout'] = (entry_price >= float(monthly_breakout_price_close)) and \
                                             (high_today >= float(monthly_breakout_price_high))
                result['monthly_breakout_price'] = float(monthly_breakout_price_close)

        return result

    def _calculate_score(self, breakout, trend_struct, breakout_price, entry,
                       vol_ratio, vol_confirm, rsi14, close, stop_pct,
                       market_ok, dist_to_break, ma20, ma60, ma60_slope,
                       adx_val, j_val, pos_val, trend_strong,
                       weekly_breakout=False, monthly_breakout=False, breakout_info=None):
        """计算得分和理由（增强版）"""
        reasons = []

        # 突破强度
        if np.isfinite(breakout_price) and breakout_price > 0:
            breakout_strength = max(0.0, min(0.08, (entry / float(breakout_price) - 1)))
        else:
            breakout_strength = 0.0

        # 多周期突破得分（日周月共振）
        if settings.MULTI_TIMEFRAME_MODE and breakout_info is not None:
            breakout_signals = sum([breakout, weekly_breakout, monthly_breakout])
            # 3个周期突破: 1.0, 2个周期突破: 0.8, 1个周期突破: 0.5
            if breakout_signals == 3:
                breakout_score = 1.0
                reasons.append(f"日周月三周期共振突破")
            elif breakout_signals == 2:
                breakout_score = 0.8
                periods = []
                if breakout: periods.append("日")
                if weekly_breakout: periods.append("周")
                if monthly_breakout: periods.append("月")
                reasons.append(f"{'+'.join(periods)}双周期共振突破")
            else:
                breakout_score = 0.5
                reasons.append(f"{'日' if breakout else ('周' if weekly_breakout else '月')}单周期突破")

            # 加入具体突破价信息
            reasons.append(f"日突破价≈{breakout_info['daily_breakout_price']:.2f}")
            if weekly_breakout:
                reasons.append(f"周突破价≈{breakout_info['weekly_breakout_price']:.2f}")
            if monthly_breakout:
                reasons.append(f"月突破价≈{breakout_info['monthly_breakout_price']:.2f}")
        else:
            breakout_score = 1.0 if breakout else 0.5
            if breakout:
                reasons.append(f"收盘+最高价同时突破近{settings.BREAKOUT_N}日高位（突破价≈{breakout_price:.2f}）")
            else:
                reasons.append(f"接近突破：距{settings.BREAKOUT_N}日突破价≈{dist_to_break*100:.2f}%（突破价≈{breakout_price:.2f}）")

        score_trend = 30 * (
            0.40 * breakout_score
            + 0.20 * (1 if trend_struct else 0)
            + 0.20 * (1 if trend_strong else 0)
            + 0.10 * (1 if weekly_breakout else 0)
            + 0.10 * (1 if monthly_breakout else 0)
        )

        reasons.append(
            f"趋势结构：MA{settings.MA_FAST}({ma20:.2f}) {'>' if ma20>ma60 else '<='} "
            f"MA{settings.MA_SLOW}({ma60:.2f})；MA{settings.MA_SLOW}斜率≈{ma60_slope:.3f}"
        )

        if np.isfinite(adx_val):
            reasons.append(f"ADX趋势强度≈{adx_val:.1f} ({'强趋势' if adx_val > 25 else '弱趋势'})")

        # 量能得分
        vr = float(vol_ratio) if np.isfinite(vol_ratio) else 0.0
        vr_clipped = max(0.0, min(2.5, vr))
        score_vol = 25 * (0.6 * (1 if vol_confirm else 0) + 0.4 * (vr_clipped / 2.5))
        reasons.append(f"量能：成交额/20日均额≈{vr:.2f}（阈值≥{settings.VOL_CONFIRM_MULT}）")

        # RSI和KDJ
        if np.isfinite(rsi14):
            reasons.append(f"RSI14≈{rsi14:.1f}（过热阈值<{settings.RSI_MAX}）")

        if np.isfinite(j_val):
            reasons.append(f"KDJ-J≈{j_val:.1f} ({'过热' if j_val > 100 else '超卖' if j_val < 0 else '正常'})")

        # 20日收益
        if len(close) >= 21 and close.iloc[-21] > 0:
            ret20 = (close.iloc[-1] / close.iloc[-21] - 1)
            if np.isfinite(ret20):
                reasons.append(f"近20日收益≈{ret20*100:.2f}%")

        # 价格位置
        if np.isfinite(pos_val):
            reasons.append(f"价格位置≈{pos_val*100:.1f}% ({'高位' if pos_val > 0.8 else '低位' if pos_val < 0.2 else '中位'})")

        # 风险得分
        if len(close) >= 21:
            vol20 = close.pct_change().iloc[-21:].std()
            vol_penalty = 0.0 if (np.isfinite(vol20) and vol20 < 0.035) else 0.15
        else:
            vol_penalty = 0.15

        # 价格位置惩罚（高位惩罚）
        pos_penalty = 0.0
        if np.isfinite(pos_val):
            if pos_val > 0.9:
                pos_penalty = 0.20
            elif pos_val > 0.8:
                pos_penalty = 0.10

        stop_score = max(0.0, 1.0 - abs(stop_pct + 0.05) / 0.05)
        score_risk = 25 * (stop_score - vol_penalty - pos_penalty)

        # 止损价
        hard_stop = entry * (1 - settings.MAX_LOSS_PCT)
        reasons.append(f"风控：止损价≈{hard_stop:.2f}（约{stop_pct*100:.2f}%），不超过-10%")

        # 市场得分
        score_mkt = 20 * (1.0 if market_ok else 0.35)
        total = score_trend + score_vol + score_risk + score_mkt

        return total, reasons

    def _get_stock_name(self, code: str) -> str:
        """获取股票名称"""
        if self.basic.empty:
            return ""
        match = self.basic[self.basic["ts_code"] == code]
        if not match.empty:
            name_data = match.iloc[0]["name"]
            if pd.notna(name_data) and str(name_data).strip() != "":
                return str(name_data).strip()
        return ""

    def _get_stock_industry(self, code: str) -> str:
        """获取股票行业"""
        if self.basic.empty:
            return ""
        match = self.basic[self.basic["ts_code"] == code]
        if not match.empty:
            industry_data = match.iloc[0]["industry"]
            if pd.notna(industry_data) and str(industry_data).strip() != "":
                return str(industry_data).strip()
        return ""
