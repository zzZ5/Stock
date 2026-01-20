"""
趋势雷达选股系统 - 优化的数据获取模块
支持并发请求、批量获取、自动重试、智能缓存
"""
import pandas as pd
import tushare as ts
from datetime import datetime
from dateutil.relativedelta import relativedelta
from typing import List, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

from core.cache_manager_optimized import CacheManager
from core.utils_optimized import (
    RateLimiter, 
    ConcurrentRateLimiter, 
    ThreadPool, 
    BatchProcessor,
    retry_on_failure
)
from core.logger import get_datafetcher_logger
from core.validators import (
    ValidationError,
    DateValidator,
    DataFrameValidator,
    PriceValidator
)
from config.settings import (
    STOCK_BASIC_TTL_DAYS,
    TRADE_CAL_TTL_DAYS,
    SLEEP_PER_CALL,
    MAX_CALLS_PER_MINUTE
)


class DataFetcherOptimized:
    """增强的数据获取器 - 支持并发、批量获取"""

    def __init__(self, 
                 token: str, 
                 rate_limiter: RateLimiter = None,
                 use_concurrent_limiter: bool = True,
                 max_workers: int = 4):
        """
        初始化优化的数据获取器

        参数:
            token: Tushare API token
            rate_limiter: 限流器实例
            use_concurrent_limiter: 是否使用并发限流器
            max_workers: 最大并发线程数
        """
        self.logger = get_datafetcher_logger()

        # 验证token
        if not token or not isinstance(token, str) or len(token.strip()) < 10:
            raise ValidationError("Tushare token 不能为空或太短")

        try:
            ts.set_token(token)
            self.pro = ts.pro_api()
            self.logger.info("Tushare API 初始化成功")
        except Exception as e:
            self.logger.error(f"Tushare API 初始化失败: {e}")
            raise

        # 使用优化的缓存管理器
        self.cache = CacheManager(
            memory_cache_size=100,
            enable_compression=True
        )

        # 初始化限流器
        if use_concurrent_limiter:
            self.rate_limiter = ConcurrentRateLimiter(
                max_rate=MAX_CALLS_PER_MINUTE,
                capacity=MAX_CALLS_PER_MINUTE
            )
            self.logger.info("使用并发限流器")
        else:
            self.rate_limiter = rate_limiter or RateLimiter(MAX_CALLS_PER_MINUTE)

        self.max_workers = max_workers
        self.thread_pool = ThreadPool(max_workers=max_workers, rate_limiter=self.rate_limiter)

        self.logger.info("DataFetcherOptimized初始化完成")

    @retry_on_failure(max_retries=3, delay=1.0)
    def _safe_api_call(self, api_func, **kwargs):
        """
        安全的API调用，带自动重试

        参数:
            api_func: Tushare API函数
            **kwargs: API参数

        返回:
            DataFrame或None
        """
        try:
            self.rate_limiter.wait_if_needed()

            df = api_func(**kwargs)

            if df is None or df.empty:
                self.logger.warning(f"API返回空数据: {api_func.__name__}")
                return None

            self.logger.debug(f"API调用成功: {api_func.__name__}, 返回{len(df)}条数据")
            return df

        except Exception as e:
            self.logger.error(f"API调用失败: {api_func.__name__}, 错误: {e}")
            return None

    def get_trade_cal(self, end_date: str, lookback_calendar_days: int = 500) -> List[str]:
        """
        获取交易日历，优先使用缓存

        参数:
            end_date: 结束日期（YYYYMMDD格式）
            lookback_calendar_days: 向前回溯的日历天数

        返回:
            按日期升序的交易日列表（YYYYMMDD字符串）
        """
        self.logger.debug(f"获取交易日历: end_date={end_date}, lookback={lookback_calendar_days}")

        # 验证参数
        DateValidator.validate_date_format(end_date, date_formats=["%Y%m%d"])
        if lookback_calendar_days < 1 or lookback_calendar_days > 2000:
            raise ValidationError(f"lookback_calendar_days 应在1-2000之间: {lookback_calendar_days}")

        # 尝试从缓存获取
        cache_key = (end_date, lookback_calendar_days)
        cached_result = self.cache.get('trade_cal', *cache_key, ttl_days=TRADE_CAL_TTL_DAYS)
        if cached_result is not None:
            return cached_result

        # 缓存未命中，重新获取
        self.logger.info("交易日历缓存未命中，重新获取...")
        start = (datetime.strptime(end_date, "%Y%m%d") -
                relativedelta(days=lookback_calendar_days)).strftime("%Y%m%d")

        cal = self._safe_api_call(
            self.pro.trade_cal,
            exchange="",
            start_date=start,
            end_date=end_date
        )

        if cal is None:
            self.logger.warning("交易日历获取失败")
            return []

        # 验证返回数据
        if 'cal_date' not in cal.columns or 'is_open' not in cal.columns:
            self.logger.error("交易日历数据格式错误")
            return []

        # 提取交易日
        trade_days = cal[cal['is_open'] == 1]['cal_date'].tolist()
        trade_days.sort()

        # 存入缓存
        self.cache.put('trade_cal', trade_days, *cache_key)

        self.logger.info(f"交易日历获取成功，共{len(trade_days)}个交易日")
        return trade_days

    def get_stock_basic(self, use_cache: bool = True) -> pd.DataFrame:
        """
        获取股票基础信息，支持缓存

        参数:
            use_cache: 是否使用缓存

        返回:
            股票基础信息DataFrame
        """
        self.logger.debug("获取股票基础信息")

        if use_cache:
            cached_result = self.cache.get('stock_basic', 'basic', ttl_days=STOCK_BASIC_TTL_DAYS)
            if cached_result is not None:
                return cached_result

        # 缓存未命中，重新获取
        self.logger.info("股票基础信息缓存未命中，重新获取...")

        basic = self._safe_api_call(self.pro.stock_basic, 
                                   exchange='',
                                   list_status='L',
                                   fields='ts_code,symbol,name,area,industry,list_date')

        if basic is None:
            self.logger.warning("股票基础信息获取失败")
            return pd.DataFrame()

        # 存入缓存
        self.cache.put('stock_basic', basic, 'basic')

        self.logger.info(f"股票基础信息获取成功，共{len(basic)}只股票")
        return basic

    def get_daily_by_date(self, trade_date: str, use_cache: bool = True) -> pd.DataFrame:
        """
        获取指定交易日的日线数据

        参数:
            trade_date: 交易日（YYYYMMDD格式）
            use_cache: 是否使用缓存

        返回:
            日线数据DataFrame
        """
        self.logger.debug(f"获取日线数据: {trade_date}")

        # 验证日期格式
        DateValidator.validate_date_format(trade_date, date_formats=["%Y%m%d"])

        if use_cache:
            cached_result = self.cache.get('daily', trade_date, ttl_days=1)
            if cached_result is not None:
                return cached_result

        # 缓存未命中，重新获取
        daily = self._safe_api_call(self.pro.daily, 
                                    trade_date=trade_date)

        if daily is None or daily.empty:
            self.logger.warning(f"日线数据获取失败: {trade_date}")
            return pd.DataFrame()

        # 存入缓存
        self.cache.put('daily', daily, trade_date)

        self.logger.debug(f"日线数据获取成功: {trade_date}, {len(daily)}条记录")
        return daily

    def get_daily_window(self, trade_dates: List[str], use_cache: bool = True) -> pd.DataFrame:
        """
        批量获取多天的日线数据

        参数:
            trade_dates: 交易日列表
            use_cache: 是否使用缓存

        返回:
            合并的日线数据DataFrame
        """
        self.logger.info(f"批量获取{len(trade_dates)}天的日线数据")

        if not trade_dates:
            return pd.DataFrame()

        # 使用线程池并发获取
        results = self.thread_pool.map(
            lambda date: self.get_daily_by_date(date, use_cache),
            trade_dates
        )

        # 过滤掉None值并合并
        valid_results = [df for df in results if df is not None and not df.empty]
        
        if valid_results:
            combined_df = pd.concat(valid_results, ignore_index=True)
            self.logger.info(f"批量日线数据获取成功，共{len(combined_df)}条记录")
            return combined_df
        else:
            self.logger.warning("批量日线数据获取失败，返回空DataFrame")
            return pd.DataFrame()

    def get_daily_by_ts_code(self, ts_code: str, 
                           start_date: str, 
                           end_date: str,
                           use_cache: bool = True) -> pd.DataFrame:
        """
        获取指定股票在时间范围内的日线数据

        参数:
            ts_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            use_cache: 是否使用缓存

        返回:
            日线数据DataFrame
        """
        self.logger.debug(f"获取股票{ts_code}日线数据: {start_date} - {end_date}")

        # 验证日期格式
        DateValidator.validate_date_format(start_date, date_formats=["%Y%m%d"])
        DateValidator.validate_date_format(end_date, date_formats=["%Y%m%d"])

        cache_key = (ts_code, start_date, end_date)
        if use_cache:
            cached_result = self.cache.get('daily_stock', *cache_key, ttl_days=1)
            if cached_result is not None:
                return cached_result

        # 缓存未命中，重新获取
        daily = self._safe_api_call(self.pro.daily,
                                    ts_code=ts_code,
                                    start_date=start_date,
                                    end_date=end_date)

        if daily is None or daily.empty:
            self.logger.warning(f"股票{ts_code}日线数据获取失败")
            return pd.DataFrame()

        # 存入缓存
        self.cache.put('daily_stock', daily, *cache_key)

        return daily

    def get_daily_batch_by_ts_codes(self, ts_codes: List[str],
                                   start_date: str,
                                   end_date: str,
                                   use_cache: bool = True) -> Dict[str, pd.DataFrame]:
        """
        批量获取多只股票的日线数据

        参数:
            ts_codes: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            use_cache: 是否使用缓存

        返回:
            股票代码到DataFrame的字典
        """
        self.logger.info(f"批量获取{len(ts_codes)}只股票的日线数据")

        def fetch_single(ts_code: str) -> tuple:
            """获取单只股票数据"""
            df = self.get_daily_by_ts_code(ts_code, start_date, end_date, use_cache)
            return (ts_code, df)

        # 使用线程池并发获取
        results = self.thread_pool.map(fetch_single, ts_codes)

        # 构建结果字典
        result_dict = {code: df for code, df in results if df is not None and not df.empty}

        self.logger.info(f"批量日线数据获取成功，{len(result_dict)}/{len(ts_codes)}只股票有数据")
        return result_dict

    def get_index_daily(self, ts_code: str, 
                      start_date: str, 
                      end_date: str,
                      use_cache: bool = True) -> pd.DataFrame:
        """
        获取指数日线数据

        参数:
            ts_code: 指数代码
            start_date: 开始日期
            end_date: 结束日期
            use_cache: 是否使用缓存

        返回:
            指数日线数据DataFrame
        """
        self.logger.debug(f"获取指数{ts_code}日线数据: {start_date} - {end_date}")

        cache_key = (ts_code, start_date, end_date)
        if use_cache:
            cached_result = self.cache.get('index', *cache_key, ttl_days=1)
            if cached_result is not None:
                return cached_result

        # 缓存未命中，重新获取
        index_daily = self._safe_api_call(self.pro.index_daily,
                                        ts_code=ts_code,
                                        start_date=start_date,
                                        end_date=end_date)

        if index_daily is None or index_daily.empty:
            self.logger.warning(f"指数{ts_code}日线数据获取失败")
            return pd.DataFrame()

        # 存入缓存
        self.cache.put('index', index_daily, *cache_key)

        return index_daily

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        获取缓存统计信息

        返回:
            统计字典
        """
        return self.cache.get_stats()

    def print_cache_stats(self):
        """打印缓存统计信息"""
        self.cache.print_stats()

    def clear_cache(self, cache_type: str = None):
        """
        清理缓存

        参数:
            cache_type: 缓存类型
        """
        self.cache.clear(cache_type)
        self.logger.info(f"缓存清理完成: {cache_type if cache_type else '全部'}")

    def shutdown(self):
        """关闭数据获取器，清理资源"""
        if hasattr(self, 'thread_pool'):
            self.thread_pool.shutdown(wait=True)
        self.logger.info("DataFetcherOptimized已关闭")
