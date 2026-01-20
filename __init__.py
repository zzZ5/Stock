"""
趋势雷达选股系统 - 模块化重构版
A股趋势雷达选股系统，基于技术分析和量化筛选

主要模块:
- config: 配置参数
- utils: 工具函数（进度跟踪、限流器等）
- cache_manager: 缓存管理
- data_fetcher: 数据获取
- indicators: 技术指标计算
- strategy: 选股策略
- reporter: 报告生成
- stock_query: 股票信息查询
"""

# 直接导入各个模块
from config import *
from utils import ProgressTracker, RateLimiter
from data_fetcher import DataFetcher
from indicators import sma, atr, rsi, ema, macd, bollinger_bands
from strategy import StockStrategy
from reporter import Reporter
from stock_query import StockQuery, query_stock_industry
from trend_radar_main import main

__version__ = "2.1.0"
__author__ = "Stock Radar Team"

__all__ = [
    # Config
    "TOP_N", "BREAKOUT_N", "MA_FAST", "MA_SLOW",
    "RSI_MAX", "MAX_LOSS_PCT", "SAVE_REPORT",

    # Utils
    "ProgressTracker", "RateLimiter",

    # Data
    "DataFetcher",

    # Indicators
    "sma", "atr", "rsi", "ema", "macd", "bollinger_bands",

    # Strategy
    "StockStrategy",

    # Reporter
    "Reporter",

    # Query
    "StockQuery", "query_stock_industry",

    # Main
    "main",
]
