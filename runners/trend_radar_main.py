"""
趋势雷达选股系统 - 主程序
执行单日选股并生成报告
"""
import os
import sys
import numpy as np
import pandas as pd

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import (
    INDEX_CODE, SAVE_REPORT, REPORT_DIR,
    DEFAULT_HOLDING_DAYS, TOP_N,
    LOG_LEVEL, LOG_DIR, LOG_CONSOLE_OUTPUT,
    LOG_FILE_OUTPUT, LOG_MAX_FILE_SIZE, LOG_BACKUP_COUNT
)
from core.logger import Logger, get_logger
from core.utils import ProgressTracker, RateLimiter
from core.data_fetcher import DataFetcher
from strategy.strategy import StockStrategy
from analysis.reporter import Reporter
from indicators.indicators import sma

# 初始化日志系统
Logger.setup_logging(
    log_level=LOG_LEVEL,
    log_dir=LOG_DIR,
    console_output=LOG_CONSOLE_OUTPUT,
    file_output=LOG_FILE_OUTPUT,
    max_file_size=LOG_MAX_FILE_SIZE,
    backup_count=LOG_BACKUP_COUNT
)
logger = get_logger(__name__)


def simple_backtest(top_df: pd.DataFrame, hist: pd.DataFrame,
                  trade_date: str, holding_days: int = 10) -> dict:
    """
    简单回测：计算选中股票在未来N天的收益情况

    参数:
        top_df: Top候选股票DataFrame
        hist: 历史行情数据
        trade_date: 当前交易日期
        holding_days: 持有天数

    返回:
        回测结果字典
    """
    if top_df.empty:
        return {"count": 0, "avg_return": 0, "win_rate": 0, "details": []}

    results = []
    # 统一转换为字符串类型后比较
    hist_dates = hist["trade_date"].astype(str).unique()
    trade_date_set = set([d for d in hist_dates if d > trade_date])
    future_dates = sorted(trade_date_set)[:holding_days]

    for _, row in top_df.iterrows():
        code = row["ts_code"]
        entry_price = row["close"]

        # 获取未来N天的收盘价
        future_data = hist[(hist["ts_code"].astype(str) == code) &
                          (hist["trade_date"].astype(str).isin(future_dates))]

        if not future_data.empty:
            exit_price = future_data.iloc[0]["close"]
            ret = (exit_price - entry_price) / entry_price
            results.append({
                "code": code,
                "entry": entry_price,
                "exit": exit_price,
                "return": ret
            })

    if not results:
        return {"count": 0, "avg_return": 0, "win_rate": 0, "details": []}

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


def main():
    """主函数"""
    # 初始化
    token = os.getenv("TUSHARE_TOKEN", "706b1dbca05800fea1d77c3a727f6ad5e0b3a1d0687f8a4e3266fe9c")
    rate_limiter = RateLimiter(max_calls_per_minute=200)

    fetcher = DataFetcher(token, rate_limiter)

    # 1) 获取交易日历
    from datetime import datetime
    today = datetime.now().strftime("%Y%m%d")
    trade_dates = fetcher.get_trade_cal(end_date=today, lookback_calendar_days=800)

    if not trade_dates:
        raise RuntimeError("未获取到交易日历")

    # 使用最新的交易日作为分析日期
    trade_date = trade_dates[-1]
    print(f"分析日期：{trade_date}（最新交易日）")

    # 2) 获取股票基础信息并过滤
    basic_all = fetcher.get_stock_basic()
    strategy = StockStrategy(basic_all)
    basic = strategy.filter_basic(basic_all, trade_date=trade_date, trade_dates=trade_dates)
    universe_codes = set(basic["ts_code"].tolist())

    # 3) 获取指数数据（沪深300）
    need_days = 120
    progress_idx = ProgressTracker(need_days, f"指数数据({INDEX_CODE})")
    idx_hist = fetcher.get_index_window(INDEX_CODE, trade_dates, need_days,
                                     progress_callback=progress_idx.update)
    progress_idx.finish()

    idx_hist = idx_hist.sort_values("trade_date")
    idx_close = idx_hist["close"].astype(float)
    idx_ma20 = sma(idx_close, 20).iloc[-1]
    idx_ma60 = sma(idx_close, 60).iloc[-1]
    idx_vol20 = idx_close.pct_change().iloc[-21:].std()

    market_ok = bool(idx_ma20 > idx_ma60)
    market_hint = "趋势环境较好（可正常跟随突破）" if market_ok else \
                 "趋势环境偏弱（建议更挑剔：只做最强突破/降低出手）"
    market_status = {
        "ma20": float(idx_ma20),
        "ma60": float(idx_ma60),
        "vol20": float(idx_vol20),
        "hint": market_hint
    }

    # 4) 获取日线数据
    need_days_daily = 160
    progress_daily = ProgressTracker(need_days_daily, "日线数据")
    daily_all = fetcher.get_daily_window(trade_dates, need_days_daily,
                                      progress_callback=progress_daily.update)
    progress_daily.finish()

    daily_all = daily_all[daily_all["ts_code"].isin(universe_codes)].copy()

    excluded_stats = {
        "基础过滤后剩余（非ST/非次新等）": len(universe_codes),
        "窗口期有行情数据的股票数": daily_all["ts_code"].nunique(),
    }

    # 5) 当日行情
    df_last = daily_all[daily_all["trade_date"].astype(str) == str(trade_date)].copy()
    excluded_stats["当日有行情（可交易候选基础）"] = df_last["ts_code"].nunique()

    # 6) 选股分析
    progress_analysis = ProgressTracker(daily_all["ts_code"].nunique(), "股票分析")
    top = strategy.analyze_stocks(daily_all, market_ok,
                               progress_callback=progress_analysis.update)
    progress_analysis.finish()

    # 限制Top N
    if not top.empty:
        top = top.head(TOP_N).reset_index(drop=True)

    excluded_stats["最终进入Top列表"] = 0 if top.empty else len(top)

    # 7) 生成报告
    report = Reporter.render_markdown(trade_date, market_status, top, excluded_stats)
    Reporter.print_console(report, top)

    # 8) 简单回测
    holding_days = DEFAULT_HOLDING_DAYS
    backtest_result = simple_backtest(top, daily_all, trade_date, holding_days)
    backtest_summary = Reporter.render_backtest_summary(backtest_result, holding_days)
    print(backtest_summary)

    # 9) 保存报告
    if SAVE_REPORT:
        full_report = report + backtest_summary
        Reporter.save_report(trade_date, full_report, REPORT_DIR)


if __name__ == "__main__":
    main()
