"""
趋势雷达选股系统 - 模块化重构版
A股趋势雷达选股系统，基于技术分析和量化筛选

主要模块:
- config: 配置参数
- core: 核心功能（数据获取、缓存、工具）
- indicators: 技术指标计算
- strategy: 选股策略
- analysis: 分析模块（回测、优化、报告）
- runners: 运行脚本
"""

# 直接导入各个模块
from config.settings import *
from core import DataFetcher, CacheManager, RateLimiter, ProgressTracker
from indicators import (
    sma, atr, rsi, ema, macd, bollinger_bands,
    adx, kdj, williams_r, obv, money_flow_index,
    chandelier_exit, volatility_ratio, price_position,
    volume_price_trend, cci, momentum, roc
)
from strategy import StockStrategy, query_stock_industry, query_stock_detail
from analysis import BacktestEngine, ParameterOptimizer, Reporter

__version__ = "2.1.0"
__author__ = "Stock Radar Team"

__all__ = [
    # Config
    "TOP_N", "BREAKOUT_N", "MA_FAST", "MA_SLOW",
    "RSI_MAX", "MAX_LOSS_PCT", "SAVE_REPORT",

    # Core
    "DataFetcher", "CacheManager", "RateLimiter", "ProgressTracker",

    # Indicators
    "sma", "atr", "rsi", "ema", "macd", "bollinger_bands",
    "adx", "kdj", "williams_r", "obv", "money_flow_index",
    "chandelier_exit", "volatility_ratio", "price_position",
    "volume_price_trend", "cci", "momentum", "roc",

    # Strategy
    "StockStrategy", "query_stock_industry", "query_stock_detail",

    # Analysis
    "BacktestEngine", "ParameterOptimizer", "Reporter",
]
