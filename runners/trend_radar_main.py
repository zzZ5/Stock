"""
趋势雷达选股系统 - 主程序
整合原版和优化版功能
"""
import os
import sys
import numpy as np
import pandas as pd
import argparse
from typing import Dict, Tuple, Optional
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import settings
from core.logger import Logger, get_logger
from core.utils import ProgressTracker, RateLimiter
from core.data_fetcher import DataFetcher
from strategy.strategy import StockStrategy
from analysis.reporter import Reporter
from analysis.market_analyzer import MarketAnalyzer, SectorAnalyzer
from indicators.indicators import sma, atr, rsi, adx

# 初始化日志系统
Logger.setup_logging(
    log_level=settings.LOG_LEVEL,
    log_dir=settings.LOG_DIR,
    console_output=settings.LOG_CONSOLE_OUTPUT,
    file_output=settings.LOG_FILE_OUTPUT,
    max_file_size=settings.LOG_MAX_FILE_SIZE,
    backup_count=settings.LOG_BACKUP_COUNT
)
logger = get_logger(__name__)


class SimpleBacktester:
    """简化版回测器"""

    def __init__(self, hist_data: pd.DataFrame):
        """初始化回测器"""
        self.hist = hist_data

    def evaluate_future_performance(self, top_df: pd.DataFrame,
                                   trade_date: str, holding_days: int) -> dict:
        """评估未来N天的表现"""
        if top_df.empty:
            return {"count": 0, "avg_return": 0, "win_rate": 0}

        hist_dates = self.hist["trade_date"].astype(str).unique()
        trade_date_set = set([d for d in hist_dates if d > trade_date])
        future_dates = sorted(trade_date_set)[:holding_days]

        results = []
        for _, row in top_df.iterrows():
            code = row["ts_code"]
            entry_price = row["close"]

            future_data = self.hist[
                (self.hist["ts_code"].astype(str) == code) &
                (self.hist["trade_date"].astype(str).isin(future_dates))
            ]

            if not future_data.empty:
                exit_price = future_data.iloc[0]["close"]
                ret = (exit_price - entry_price) / entry_price
                results.append({"code": code, "return": ret})

        if not results:
            return {"count": 0, "avg_return": 0, "win_rate": 0}

        returns = [r["return"] for r in results]
        win_count = sum(1 for r in returns if r > 0)

        return {
            "count": len(results),
            "avg_return": np.mean(returns) * 100,
            "win_rate": win_count / len(results) * 100,
            "max_return": max(returns) * 100,
            "min_return": min(returns) * 100,
            "details": results
        }


def run_analysis(args):
    """运行主分析流程"""
    # 初始化
    token = os.getenv("TUSHARE_TOKEN", args.token)
    rate_limiter = RateLimiter(max_calls_per_minute=200)

    fetcher = DataFetcher(token, rate_limiter)

    # 1) 获取交易日历
    today = datetime.now().strftime("%Y%m%d")
    trade_dates = fetcher.get_trade_cal(end_date=today, lookback_calendar_days=800)

    if not trade_dates:
        raise RuntimeError("未获取到交易日历")

    trade_date = trade_dates[-1]
    print(f"分析日期：{trade_date}（最新交易日）")

    # 2) 获取股票基础信息并过滤
    basic_all = fetcher.get_stock_basic()
    strategy = StockStrategy(basic_all)
    basic = strategy.filter_basic(basic_all, trade_date=trade_date, trade_dates=trade_dates)
    universe_codes = set(basic["ts_code"].tolist())

    # 3) 获取指数数据
    need_days = 120
    progress_idx = ProgressTracker(need_days, f"指数数据({args.index_code})", rate_limiter)
    idx_hist = fetcher.get_index_window(args.index_code, trade_dates, need_days,
                                     progress_callback=progress_idx.update)
    progress_idx.finish()

    # 验证指数数据
    if idx_hist.empty:
        raise RuntimeError(f"未获取到指数数据: {args.index_code}")

    actual_days = len(idx_hist)
    idx_hist = idx_hist.sort_values("trade_date")
    idx_close = idx_hist["close"].astype(float)

    # 大盘晴雨表分析
    print("\n[大盘分析] 正在分析大盘晴雨表...")
    market_analyzer = MarketAnalyzer()
    market_analysis = market_analyzer.analyze(idx_hist, args.index_code)

    # 输出大盘晴雨表
    print(f"\n{'='*60}")
    print(f"{'大盘晴雨表':^58}")
    print(f"{'='*60}")
    print(f"天气状况: {market_analysis.get('weather', '未知')}")
    print(f"综合评分: {market_analysis.get('score', 0)}/100")
    print(f"市场情绪: {market_analysis.get('sentiment', '未知')} {market_analysis.get('sentiment_emoji', '')}")
    print(f"趋势: {market_analysis.get('trend', '未知')} (强度: {market_analysis.get('trend_strength', '未知')})")
    print(f"日涨跌: {market_analysis.get('daily_change', 0):.2f}% | 周涨跌: {market_analysis.get('weekly_change', 0):.2f}% | 月涨跌: {market_analysis.get('monthly_change', 0):.2f}%")
    print(f"{'='*60}\n")

    # 根据实际数据量动态调整指标计算
    print(f"指数数据实际获取: {actual_days}天")

    # 计算MA20（需要至少20天数据）
    if actual_days < 20:
        raise RuntimeError(
            f"指数数据严重不足，需要至少20个交易日计算MA20，实际获取{actual_days}个。\n"
            f"请检查：1) 指数代码{args.index_code}是否正确 2) 交易日历数据是否完整"
        )

    # 根据实际数据量计算MA20
    ma_period = min(20, actual_days)
    idx_ma20 = sma(idx_close, ma_period).iloc[-1]

    # 计算MA60（需要至少60天数据）
    if actual_days >= 60:
        idx_ma60 = sma(idx_close, 60).iloc[-1]
    else:
        # 数据不足60天时，使用实际数据量的一半或最大可用值
        ma_period = min(60, actual_days)
        idx_ma60 = sma(idx_close, ma_period).iloc[-1]
        print(f"警告：数据不足60天，MA60使用{ma_period}日均线")

    # 波动率计算
    vol_period = min(21, actual_days)
    if actual_days >= 2:
        idx_vol20 = idx_close.pct_change().iloc[-vol_period:].std()
    else:
        idx_vol20 = 0.0
        print(f"警告：数据不足，无法计算波动率")

    # 处理可能的NaN值
    if pd.isna(idx_ma20):
        idx_ma20 = idx_close.iloc[-1]
        print(f"警告：MA20计算失败，使用最新收盘价")

    if pd.isna(idx_ma60):
        idx_ma60 = idx_close.iloc[-1]
        print(f"警告：MA60计算失败，使用最新收盘价")

    if pd.isna(idx_vol20):
        idx_vol20 = 0.0
        print(f"警告：波动率计算失败，使用0")

    market_ok = bool(idx_ma20 > idx_ma60)
    market_status = {
        "ma20": float(idx_ma20),
        "ma60": float(idx_ma60),
        "vol20": float(idx_vol20),
        "environment": "bullish" if market_ok else "bearish"
    }

    # 4) 获取日线数据
    need_days_daily = 160
    progress_daily = ProgressTracker(need_days_daily, "日线数据", rate_limiter)
    daily_all = fetcher.get_daily_window(trade_dates, need_days_daily,
                                      progress_callback=progress_daily.update)
    progress_daily.finish()

    daily_all = daily_all[daily_all["ts_code"].isin(universe_codes)].copy()

    # 优化数据类型
    numeric_cols = ['open', 'high', 'low', 'close', 'vol', 'amount']
    for col in numeric_cols:
        if col in daily_all.columns:
            daily_all[col] = pd.to_numeric(daily_all[col], downcast='float')
    daily_all['ts_code'] = daily_all['ts_code'].astype('category')

    # 5) 获取周线和月线数据（多周期模式）
    weekly_all = pd.DataFrame()
    monthly_all = pd.DataFrame()

    if settings.MULTI_TIMEFRAME_MODE:
        print(f"\n多周期模式：获取周线和月线数据...")

        # 获取周线数据
        need_weeks = settings.WEEKLY_BREAKOUT_N + 5
        weekly_start_date = trade_dates[-int(need_weeks*5.5)]
        weekly_end_date = trade_dates[-1]
        codes_list = list(universe_codes)

        print(f"获取周线数据 ({len(codes_list)}只股票)...")
        progress_weekly = ProgressTracker(len(codes_list), "周线数据", rate_limiter)
        weekly_all = fetcher.get_weekly_data(codes_list, weekly_start_date, weekly_end_date,
                                           progress_callback=progress_weekly.update)
        progress_weekly.finish()
        if not weekly_all.empty:
            print(f"周线数据获取成功: {len(weekly_all)}条记录")
        else:
            print("警告：周线数据获取失败")

        # 获取月线数据
        need_months = settings.MONTHLY_BREAKOUT_N + 3
        monthly_start_date = trade_dates[-int(need_months*22)]
        monthly_end_date = trade_dates[-1]

        print(f"获取月线数据 ({len(codes_list)}只股票)...")
        progress_monthly = ProgressTracker(len(codes_list), "月线数据", rate_limiter)
        monthly_all = fetcher.get_monthly_data(codes_list, monthly_start_date, monthly_end_date,
                                            progress_callback=progress_monthly.update)
        progress_monthly.finish()
        if not monthly_all.empty:
            print(f"月线数据获取成功: {len(monthly_all)}条记录")
        else:
            print("警告：月线数据获取失败")

        # 优化周线和月线数据类型
        if not weekly_all.empty:
            for col in numeric_cols:
                if col in weekly_all.columns:
                    weekly_all[col] = pd.to_numeric(weekly_all[col], downcast='float')
            weekly_all['ts_code'] = weekly_all['ts_code'].astype('category')

        if not monthly_all.empty:
            for col in numeric_cols:
                if col in monthly_all.columns:
                    monthly_all[col] = pd.to_numeric(monthly_all[col], downcast='float')
            monthly_all['ts_code'] = monthly_all['ts_code'].astype('category')

    # 6) 设置多周期数据

    excluded_stats = {
        "基础过滤后剩余": len(universe_codes),
        "窗口期有行情数据的股票数": daily_all["ts_code"].nunique(),
    }

    # 7) 当日行情
    df_last = daily_all[daily_all["trade_date"].astype(str) == str(trade_date)].copy()
    excluded_stats["当日有行情"] = df_last["ts_code"].nunique()

    # 板块晴雨表分析
    print("\n[板块分析] 正在分析板块晴雨表...")
    sector_analyzer = SectorAnalyzer()
    sector_analysis = sector_analyzer.analyze(daily_all, basic, trade_date, trade_dates)

    # 输出板块晴雨表
    if sector_analysis and sector_analysis.get('top_sectors'):
        print(f"\n{'='*70}")
        print(f"{'板块晴雨表':^66}")
        print(f"{'='*70}")
        print(f"市场广度: {sector_analysis.get('market_breadth', 50):.1f}%")
        print(f"\n【领涨板块】")
        print(f"{'排名':<6} {'板块':<20} {'评分':<8} {'平均涨幅':<10}")
        print(f"{'-'*70}")
        for idx, sector in enumerate(sector_analysis['top_sectors'][:5]):
            print(f"{idx+1:<6} {sector['industry']:<20} {sector['score']:<8.0f} {sector['avg_pct_chg']:<10.2f}%")
        print(f"{'='*70}\n")

    # 8) 选股分析
    progress_analysis = ProgressTracker(daily_all["ts_code"].nunique(), "股票分析", rate_limiter)

    # 设置多周期数据到策略类
    strategy.set_multi_timeframe_data(weekly_all, monthly_all)

    top = strategy.analyze_stocks(daily_all, market_ok,
                               progress_callback=progress_analysis.update)
    progress_analysis.finish()

    if not top.empty:
        top = top.head(args.top_n).reset_index(drop=True)

    excluded_stats["最终进入Top列表"] = 0 if top.empty else len(top)

    # 9) 生成报告
    report = Reporter.render_markdown(trade_date, market_status, top, excluded_stats,
                                       market_analysis, sector_analysis)
    Reporter.print_console(report, top)

    # 10) 回测
    backtester = SimpleBacktester(daily_all)
    backtest_result = backtester.evaluate_future_performance(
        top, trade_date, args.holding_days
    )
    backtest_summary = Reporter.render_backtest_summary(backtest_result, args.holding_days)
    print(backtest_summary)

    # 11) 保存报告
    if args.save_report:
        full_report = report + backtest_summary
        Reporter.save_report(trade_date, full_report, settings.REPORT_DIR, save_pdf=not args.no_pdf)

    # 多周期统计信息
    if settings.MULTI_TIMEFRAME_MODE and not top.empty:
        weekly_count = top.get("weekly_breakout", pd.Series()).sum()
        monthly_count = top.get("monthly_breakout", pd.Series()).sum()
        print(f"\n多周期突破统计:")
        print(f"  周线突破: {weekly_count}只 ({weekly_count/len(top)*100:.1f}%)")
        print(f"  月线突破: {monthly_count}只 ({monthly_count/len(top)*100:.1f}%)")

    # 显示API等待统计
    if fetcher.rate_limiter:
        total_waits = fetcher.rate_limiter.get_wait_count()
        if total_waits > 0:
            print(f"\n📊 API调用统计:")
            print(f"  因API限制共等待{total_waits}次")
            print(f"  每次等待约1分钟，这是正常现象")
            print(f"  使用缓存可以大幅减少等待次数")

    return {
        "trade_date": trade_date,
        "top_count": len(top),
        "avg_return": backtest_result["avg_return"],
        "win_rate": backtest_result["win_rate"]
    }


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='趋势雷达选股系统')
    parser.add_argument('--top-n', type=int, default=settings.TOP_N, help='返回Top N股票')
    parser.add_argument('--index-code', type=str, default=settings.INDEX_CODE, help='指数代码')
    parser.add_argument('--holding-days', type=int, default=settings.DEFAULT_HOLDING_DAYS, help='持有天数')
    parser.add_argument('--save-report', action='store_true', default=settings.SAVE_REPORT, help='保存报告')
    parser.add_argument('--no-pdf', action='store_true', help='不生成PDF报告')
    parser.add_argument('--token', type=str, default='706b1dbca05800fea1d77c3a727f6ad5e0b3a1d0687f8a4e3266fe9c', help='Tushare Token')
    parser.add_argument('--verbose', action='store_true', help='详细输出')
    parser.add_argument('--multi-tf', action='store_true', default=settings.MULTI_TIMEFRAME_MODE, help='启用多周期突破（日周月）')
    parser.add_argument('--daily-only', action='store_true', help='仅使用日线突破')

    args = parser.parse_args()

    # 处理多周期模式
    if args.daily_only:
        settings.MULTI_TIMEFRAME_MODE = False
        print("模式：日线突破仅")
    elif args.multi_tf:
        settings.MULTI_TIMEFRAME_MODE = True
        print(f"模式：多周期突破（日+周+月）")
    else:
        mode_str = "多周期（日+周+月）" if settings.MULTI_TIMEFRAME_MODE else "日线仅"
        print(f"模式：{mode_str}")

    result = run_analysis(args)

    if args.verbose:
        print(f"\n=== 运行摘要 ===")
        print(f"分析日期: {result['trade_date']}")
        print(f"选中股票: {result['top_count']}只")
        print(f"平均收益: {result['avg_return']:.2f}%")
        print(f"胜率: {result['win_rate']:.2f}%")


if __name__ == "__main__":
    main()
