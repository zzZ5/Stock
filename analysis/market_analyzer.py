"""
趋势雷达选股系统 - 市场晴雨表分析模块
分析大盘和板块的情绪、趋势、资金流向等
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from indicators.indicators import sma, atr, rsi, adx
from core.logger import get_analyzer_logger

logger = get_analyzer_logger()


class MarketAnalyzer:
    """大盘晴雨表分析器"""

    def __init__(self):
        """初始化大盘分析器"""
        pass

    def analyze(self, index_df: pd.DataFrame, index_code: str) -> Dict:
        """
        分析大盘晴雨表

        参数:
            index_df: 指数历史数据
            index_code: 指数代码

        返回:
            晴雨表分析结果字典
        """
        if index_df.empty:
            return {"status": "无数据"}

        # 确保数据按日期排序
        df = index_df.sort_values("trade_date").copy()

        # 转换数据类型
        for col in ['open', 'high', 'low', 'close', 'vol', 'amount']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # 获取最新数据
        latest = df.iloc[-1]

        # 计算技术指标
        analysis = {}

        # 1. 趋势分析
        analysis.update(self._analyze_trend(df))

        # 2. 情绪分析
        analysis.update(self._analyze_sentiment(df))

        # 3. 技术面分析
        analysis.update(self._analyze_technical(df))

        # 4. 综合评分
        analysis.update(self._calculate_score(analysis))

        return analysis

    def _analyze_trend(self, df: pd.DataFrame) -> Dict:
        """分析趋势"""
        if len(df) < 60:
            return {"trend": "数据不足"}

        close = df['close'].values
        current = close[-1]
        ma5 = sma(close, 5)[-1]
        ma10 = sma(close, 10)[-1]
        ma20 = sma(close, 20)[-1]
        ma60 = sma(close, 60)[-1]

        # 日涨幅
        daily_change = (current - close[-2]) / close[-2] * 100

        # 累计涨幅
        weekly_change = (current - close[-5]) / close[-5] * 100 if len(df) >= 5 else 0
        monthly_change = (current - close[-20]) / close[-20] * 100 if len(df) >= 20 else 0

        # 均线多头/空头排列
        ma_bullish = (ma5 > ma10 > ma20)
        ma_bearish = (ma5 < ma10 < ma20)

        if ma_bullish and current > ma5:
            trend = "强势上升"
        elif ma_bullish:
            trend = "震荡上行"
        elif ma_bearish and current < ma5:
            trend = "弱势下跌"
        elif ma_bearish:
            trend = "震荡下行"
        else:
            trend = "横盘整理"

        return {
            "trend": trend,
            "close": current,
            "ma5": ma5,
            "ma10": ma10,
            "ma20": ma20,
            "ma60": ma60,
            "daily_change": daily_change,
            "weekly_change": weekly_change,
            "monthly_change": monthly_change,
            "ma_bullish": ma_bullish,
            "ma_bearish": ma_bearish
        }

    def _analyze_sentiment(self, df: pd.DataFrame) -> Dict:
        """分析市场情绪"""
        if len(df) < 20:
            return {"sentiment": "数据不足"}

        # RSI
        rsi_values = rsi(df['close'], 14)
        current_rsi = rsi_values[-1]

        # ADX
        adx_values = adx(df['high'], df['low'], df['close'], 14)
        current_adx = adx_values[-1]

        # ATR（波动率）
        atr_values = atr(df['high'], df['low'], df['close'], 20)
        current_atr = atr_values[-1]
        atr_pct = current_atr / df['close'].iloc[-1] * 100

        # 情绪判断
        if current_rsi > 70:
            sentiment = "极度贪婪"
            sentiment_emoji = "🔥"
        elif current_rsi > 60:
            sentiment = "贪婪"
            sentiment_emoji = "😊"
        elif current_rsi < 30:
            sentiment = "极度恐惧"
            sentiment_emoji = "😱"
        elif current_rsi < 40:
            sentiment = "恐惧"
            sentiment_emoji = "😰"
        else:
            sentiment = "中性"
            sentiment_emoji = "😐"

        # 趋势强度
        if current_adx > 40:
            trend_strength = "极强"
        elif current_adx > 25:
            trend_strength = "强"
        elif current_adx > 20:
            trend_strength = "中等"
        else:
            trend_strength = "弱"

        return {
            "sentiment": sentiment,
            "sentiment_emoji": sentiment_emoji,
            "rsi": current_rsi,
            "adx": current_adx,
            "trend_strength": trend_strength,
            "atr_pct": atr_pct,
            "volatility": "高" if atr_pct > 2 else "低"
        }

    def _analyze_technical(self, df: pd.DataFrame) -> Dict:
        """分析技术面"""
        if len(df) < 60:
            return {"technical": "数据不足"}

        # 成交量分析
        vol = df['vol'].values
        ma_vol5 = sma(vol, 5)[-1]
        ma_vol10 = sma(vol, 10)[-1]
        vol_ratio = vol[-1] / ma_vol10 if ma_vol10 > 0 else 1

        # 位置分析
        close = df['close'].values
        high20 = df['high'][-20:].max()
        low20 = df['low'][-20:].min()
        price_position = (close[-1] - low20) / (high20 - low20) * 100 if high20 != low20 else 50

        # 支撑压力
        support = df['low'][-20:].min()
        resistance = df['high'][-20:].max()

        return {
            "vol_ratio": vol_ratio,
            "price_position": price_position,
            "support": support,
            "resistance": resistance,
            "vol_surge": vol_ratio > 1.5
        }

    def _calculate_score(self, analysis: Dict) -> Dict:
        """计算综合评分"""
        score = 50  # 基础分

        # 趋势加分
        if "ma_bullish" in analysis:
            if analysis["ma_bullish"]:
                score += 20
            elif analysis.get("ma_bearish", False):
                score -= 20

        # RSI调整
        if "rsi" in analysis:
            rsi = analysis["rsi"]
            if 40 <= rsi <= 60:
                score += 10
            elif rsi > 70:
                score -= 15
            elif rsi < 30:
                score += 5

        # 日涨跌调整
        if "daily_change" in analysis:
            daily_change = analysis["daily_change"]
            score += min(daily_change / 2, 20)

        # 成交量调整
        if "vol_ratio" in analysis:
            vol_ratio = analysis["vol_ratio"]
            if vol_ratio > 1.5:
                score += 10

        # 限制在0-100
        score = max(0, min(100, int(score)))

        # 天气等级
        if score >= 80:
            weather = "☀️ 晴朗"
        elif score >= 65:
            weather = "⛅ 多云"
        elif score >= 45:
            weather = "☁️ 阴天"
        elif score >= 30:
            weather = "🌧️ 小雨"
        else:
            weather = "⛈️ 暴雨"

        return {
            "score": score,
            "weather": weather
        }


class SectorAnalyzer:
    """板块晴雨表分析器"""

    def __init__(self):
        """初始化板块分析器"""
        pass

    def analyze(self, daily_df: pd.DataFrame, basic_df: pd.DataFrame,
                trade_date: str, trade_dates: List[str] = None) -> Dict:
        """
        分析板块晴雨表

        参数:
            daily_df: 全市场日线数据
            basic_df: 股票基础信息（含行业分类）
            trade_date: 分析日期
            trade_dates: 交易日列表

        返回:
            板块分析结果字典
        """
        if daily_df.empty or basic_df.empty:
            return {"sectors": []}

        # 过滤当日有行情的股票
        df_today = daily_df[daily_df['trade_date'].astype(str) == str(trade_date)].copy()

        if df_today.empty:
            return {"sectors": []}

        # 合并行业信息
        df_merged = df_today.merge(basic_df[['ts_code', 'name', 'industry']], on='ts_code', how='left')

        # 剔除ST股票
        df_merged = df_merged[~df_merged['name'].str.contains('ST|\\*ST|退', regex=True)]

        # 计算各板块涨跌幅
        sector_stats = df_merged.groupby('industry').agg({
            'ts_code': 'count',
            'close': 'mean',
            'pct_chg': 'mean',
            'amount': 'sum'
        }).reset_index()

        sector_stats.columns = ['industry', 'stock_count', 'avg_price', 'avg_pct_chg', 'total_amount']

        # 计算涨跌家数
        up_down_stats = df_merged.groupby('industry')['pct_chg'].agg([
            lambda x: (x > 0).sum(),  # 上涨家数
            lambda x: (x < 0).sum(),  # 下跌家数
            lambda x: (x > 9.9).sum(),  # 涨停家数
            lambda x: (x < -9.9).sum(),  # 跌停家数
        ]).reset_index()

        up_down_stats.columns = ['industry', 'up_count', 'down_count', 'limit_up', 'limit_down']

        # 合并统计
        sector_stats = sector_stats.merge(up_down_stats, on='industry')

        # 计算涨跌比
        sector_stats['up_down_ratio'] = sector_stats['up_count'] / (sector_stats['down_count'] + 1)

        # 计算活跃度（成交额占比）
        sector_stats['active_ratio'] = sector_stats['total_amount'] / sector_stats['total_amount'].sum() * 100

        # 计算板块评分
        sector_stats['score'] = self._calculate_sector_score(sector_stats)

        # 排序
        sector_stats = sector_stats.sort_values('score', ascending=False).reset_index(drop=True)

        # 分类
        top_sectors = sector_stats.head(5)
        weak_sectors = sector_stats.tail(5)

        # 计算整体市场情绪
        total_up = sector_stats['up_count'].sum()
        total_down = sector_stats['down_count'].sum()
        market_breadth = total_up / (total_up + total_down) * 100

        return {
            "sectors": sector_stats.to_dict('records'),
            "top_sectors": top_sectors.to_dict('records'),
            "weak_sectors": weak_sectors.to_dict('records'),
            "market_breadth": market_breadth
        }

    def _calculate_sector_score(self, sector_stats: pd.DataFrame) -> pd.Series:
        """计算板块评分"""
        scores = []

        for _, row in sector_stats.iterrows():
            score = 50

            # 涨跌幅加权
            score += row['avg_pct_chg'] * 2

            # 涨跌比加权
            score += (row['up_count'] - row['down_count']) * 0.5

            # 涨停家数加分
            score += row['limit_up'] * 3

            # 活跃度加权（成交额占比）
            score += row['active_ratio'] * 0.2

            # 限制在0-100
            scores.append(max(0, min(100, score)))

        return pd.Series(scores)
