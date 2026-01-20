"""
趋势雷达选股系统 - 选股策略模块
包含股票筛选、评分、策略逻辑等核心功能
"""
import pandas as pd
import numpy as np

from config import (
    BREAKOUT_N, MA_FAST, MA_SLOW, VOL_LOOKBACK,
    VOL_CONFIRM_MULT, RSI_MAX, MIN_PRICE,
    MIN_AVG_AMOUNT_20D, EXCLUDE_ONE_WORD_LIMITUP,
    MAX_LOSS_PCT, ATR_N, ATR_MULT
)
from indicators import (
    sma, atr, rsi,
    adx, kdj, williams_r, price_position
)


class StockStrategy:
    """股票选股策略"""

    def __init__(self, basic_df: pd.DataFrame = None):
        """
        初始化策略

        参数:
            basic_df: 股票基础信息DataFrame
        """
        self.basic = basic_df if basic_df is not None else pd.DataFrame()

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

        if trade_dates is not None:
            # 使用实际交易日计算
            trade_date_set = set(trade_dates)
            basic = basic[basic["list_date"].apply(
                lambda x: len([d for d in trade_dates
                             if d >= x.strftime("%Y%m%d") and d <= trade_date])
            ) >= 120].copy()
        else:
            # 回退方案：用日历天*1.6粗略估算
            basic = basic[(today - basic["list_date"]).dt.days >= 120 * 1.6].copy()

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
        out_rows = []
        hist_sorted = hist.sort_values(["ts_code", "trade_date"])
        unique_codes = hist_sorted["ts_code"].unique()

        print(f"\n[分析] 开始分析股票（共 {len(unique_codes)} 只）...")

        for i, code in enumerate(unique_codes):
            g = hist_sorted[hist_sorted["ts_code"] == code]

            if len(g) < max(BREAKOUT_N, MA_SLOW, VOL_LOOKBACK, ATR_N) + 6:
                continue

            result = self._analyze_single_stock(g, code, market_ok)
            if result is not None:
                out_rows.append(result)

            if progress_callback:
                progress_callback(1)

        if not out_rows:
            return pd.DataFrame()

        out = pd.DataFrame(out_rows)
        out = out.sort_values(["candidate", "score"],
                             ascending=[False, False])
        return out

    def _analyze_single_stock(self, stock_data: pd.DataFrame,
                            code: str, market_ok: bool) -> dict:
        """分析单只股票"""
        last = stock_data.iloc[-1]
        close = stock_data["close"].astype(float)
        high = stock_data["high"].astype(float)
        low = stock_data["low"].astype(float)
        amount = stock_data["amount"].astype(float)

        # 计算基础指标
        ma20 = sma(close, MA_FAST).iloc[-1]
        ma60 = sma(close, MA_SLOW).iloc[-1]
        ma60_5ago = sma(close, MA_SLOW).iloc[-6]
        ma60_slope = ma60 - ma60_5ago

        rsi14 = rsi(close, 14).iloc[-1]
        atr14 = atr(stock_data[["high", "low", "close"]].assign(
            high=high, low=low, close=close), ATR_N).iloc[-1]

        # 计算新指标
        adx_val = adx(high, low, close).iloc[-1]
        kdj_data = kdj(high, low, close)
        k_val, d_val, j_val = kdj_data['k'].iloc[-1], kdj_data['d'].iloc[-1], kdj_data['j'].iloc[-1]
        wr_val = williams_r(high, low, close).iloc[-1]
        pos_val = price_position(high, low, close).iloc[-1]

        # 突破判断（同时检查收盘价和最高价）
        breakout_price_close = close.iloc[-BREAKOUT_N:].max()
        breakout_price_high = high.iloc[-BREAKOUT_N:].max()
        entry = float(last["close"])
        high_today = float(last["high"])
        breakout = (entry >= float(breakout_price_close)) and \
                  (high_today >= float(breakout_price_high))

        # 量能确认
        avg_amt20 = amount.iloc[-VOL_LOOKBACK:].mean()
        vol_ratio = float(last["amount"]) / avg_amt20 if avg_amt20 > 0 else np.nan
        vol_confirm = (np.isfinite(vol_ratio) and vol_ratio >= VOL_CONFIRM_MULT)

        # 趋势结构
        trend_struct = (ma20 > ma60) and (ma60_slope > 0)

        # 新指标判断
        trend_strong = np.isfinite(adx_val) and adx_val > 25  # ADX>25为强趋势
        not_overbought = not (np.isfinite(j_val) and j_val > 100)  # J值不过热
        not_too_high = not (np.isfinite(pos_val) and pos_val > 0.9)  # 价格不在过高位置

        # 20日收益
        ret20 = (close.iloc[-1] / close.iloc[-21] - 1) if len(close) >= 21 else np.nan

        # 过滤
        if float(last["close"]) < MIN_PRICE:
            return None
        if avg_amt20 < MIN_AVG_AMOUNT_20D:
            return None
        if EXCLUDE_ONE_WORD_LIMITUP and self.is_one_word_limitup(last):
            return None

        # 候选/观察逻辑（增强版：加入新指标判断）
        signal_hits = sum([breakout, trend_struct, vol_confirm, trend_strong])
        is_candidate = (signal_hits >= 4) and not_overbought and not_too_high and \
                      (np.isfinite(rsi14) and rsi14 <= RSI_MAX)

        dist_to_break = (float(breakout_price_close) / entry - 1) if entry > 0 else np.nan
        is_watch = (signal_hits >= 3) and \
                  (np.isfinite(dist_to_break) and dist_to_break <= 0.015) and \
                  not_overbought

        if not (is_candidate or is_watch):
            return None

        # 止损价
        hard_stop = entry * (1 - MAX_LOSS_PCT)
        atr_stop = entry - ATR_MULT * float(atr14) if np.isfinite(atr14) else hard_stop
        stop_price = max(hard_stop, atr_stop)
        stop_pct = (stop_price / entry - 1)

        # 评分（增强版）
        total, reasons = self._calculate_score(
            breakout, trend_struct, breakout_price_close, entry,
            vol_ratio, vol_confirm, rsi14, close, stop_pct,
            market_ok, dist_to_break, ma20, ma60, ma60_slope,
            adx_val, j_val, pos_val, trend_strong
        )

        # 获取名称和行业
        stock_name = self._get_stock_name(code)
        industry = self._get_stock_industry(code)

        return {
            "ts_code": code,
            "name": stock_name,
            "industry": industry,
            "close": entry,
            "high_today": high_today,
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
        }

    def _calculate_score(self, breakout, trend_struct, breakout_price, entry,
                       vol_ratio, vol_confirm, rsi14, close, stop_pct,
                       market_ok, dist_to_break, ma20, ma60, ma60_slope,
                       adx_val, j_val, pos_val, trend_strong):
        """计算得分和理由（增强版）"""
        reasons = []

        # 突破强度
        breakout_strength = max(0.0, min(0.08, (entry / float(breakout_price) - 1)))
        score_trend = 30 * (
            0.50 * (1 if breakout else 0)
            + 0.20 * (1 if trend_struct else 0)
            + 0.15 * (1 if trend_strong else 0)
            + 0.15 * (breakout_strength / 0.08)
        )

        if breakout:
            reasons.append(f"收盘+最高价同时突破近{BREAKOUT_N}日高位（突破价≈{breakout_price:.2f}）")
        else:
            reasons.append(f"接近突破：距{BREAKOUT_N}日突破价≈{dist_to_break*100:.2f}%（突破价≈{breakout_price:.2f}）")

        reasons.append(
            f"趋势结构：MA{MA_FAST}({ma20:.2f}) {'>' if ma20>ma60 else '<='} "
            f"MA{MA_SLOW}({ma60:.2f})；MA{MA_SLOW}斜率≈{ma60_slope:.3f}"
        )

        if np.isfinite(adx_val):
            reasons.append(f"ADX趋势强度≈{adx_val:.1f} ({'强趋势' if adx_val > 25 else '弱趋势'})")

        # 量能得分
        vr = float(vol_ratio) if np.isfinite(vol_ratio) else 0.0
        vr_clipped = max(0.0, min(2.5, vr))
        score_vol = 25 * (0.6 * (1 if vol_confirm else 0) + 0.4 * (vr_clipped / 2.5))
        reasons.append(f"量能：成交额/20日均额≈{vr:.2f}（阈值≥{VOL_CONFIRM_MULT}）")

        # RSI和KDJ
        if np.isfinite(rsi14):
            reasons.append(f"RSI14≈{rsi14:.1f}（过热阈值<{RSI_MAX}）")

        if np.isfinite(j_val):
            reasons.append(f"KDJ-J≈{j_val:.1f} ({'过热' if j_val > 100 else '超卖' if j_val < 0 else '正常'})")

        # 20日收益
        ret20 = (close.iloc[-1] / close.iloc[-21] - 1) if len(close) >= 21 else np.nan
        if np.isfinite(ret20):
            reasons.append(f"近20日收益≈{ret20*100:.2f}%")

        # 价格位置
        if np.isfinite(pos_val):
            reasons.append(f"价格位置≈{pos_val*100:.1f}% ({'高位' if pos_val > 0.8 else '低位' if pos_val < 0.2 else '中位'})")

        # 风险得分
        vol20 = close.pct_change().iloc[-21:].std() if len(close) >= 21 else np.nan
        vol_penalty = 0.0 if (np.isfinite(vol20) and vol20 < 0.035) else 0.15

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
        hard_stop = entry * (1 - MAX_LOSS_PCT)
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
