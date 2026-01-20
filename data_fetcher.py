"""
趋势雷达选股系统 - 数据获取模块
负责从Tushare API获取各类数据，并管理缓存
"""
import pandas as pd
import tushare as ts
from datetime import datetime
from dateutil.relativedelta import relativedelta
import time

from cache_manager import CacheManager
from utils import RateLimiter
from config import (
    STOCK_BASIC_TTL_DAYS,
    TRADE_CAL_TTL_DAYS,
    SLEEP_PER_CALL
)


class DataFetcher:
    """数据获取器"""

    def __init__(self, token: str, rate_limiter: RateLimiter = None):
        """
        初始化数据获取器

        参数:
            token: Tushare API token
            rate_limiter: 限流器实例
        """
        ts.set_token(token)
        self.pro = ts.pro_api()
        self.cache = CacheManager()
        self.rate_limiter = rate_limiter or RateLimiter(200)

    def get_trade_cal(self, end_date: str, lookback_calendar_days: int = 500) -> list[str]:
        """
        获取交易日历，优先使用缓存

        参数:
            end_date: 结束日期（YYYYMMDD格式）
            lookback_calendar_days: 向前回溯的日历天数

        返回:
            按日期升序的交易日列表（YYYYMMDD字符串）
        """
        path = self.cache.cache_path_trade_cal()

        if self.cache.is_cache_expired(path, TRADE_CAL_TTL_DAYS):
            # 缓存过期，重新获取
            start = (datetime.strptime(end_date, "%Y%m%d") -
                    relativedelta(days=lookback_calendar_days)).strftime("%Y%m%d")
            cal = self._safe_api_call(
                self.pro.trade_cal,
                exchange="",
                start_date=start,
                end_date=end_date
            )
            if cal is not None:
                self.cache.save_csv(cal, path)

        # 读取缓存
        cal = self.cache.read_csv_if_exists(path)
        if cal is None or cal.empty:
            return []

        cal = cal[cal["is_open"] == 1].sort_values("cal_date")
        return cal["cal_date"].astype(str).tolist()

    def get_stock_basic(self) -> pd.DataFrame:
        """
        获取股票基础信息，优先使用缓存

        返回:
            包含 ts_code, name, industry, list_date 等字段的DataFrame
        """
        path = self.cache.cache_path_stock_basic()

        if self.cache.is_cache_expired(path, STOCK_BASIC_TTL_DAYS):
            # 缓存过期，重新获取
            basic = self._safe_api_call(
                self.pro.stock_basic,
                exchange="",
                list_status="L",
                fields="ts_code,name,industry,list_date"
            )
            if basic is not None and not basic.empty:
                self.cache.save_csv(basic, path)

        result = self.cache.read_csv_if_exists(path)
        if result is None or result.empty:
            return pd.DataFrame()
        return result

    def get_daily_by_date(self, trade_date: str) -> pd.DataFrame:
        """
        获取某交易日全市场日线数据，优先使用缓存

        参数:
            trade_date: 交易日（YYYYMMDD格式）

        返回:
            包含 ts_code, trade_date, open, high, low, close, amount 等字段的DataFrame
        """
        path = self.cache.cache_path_daily(trade_date)

        if not self.cache.is_cache_expired(path, 365):  # 日线数据缓存1年
            df = self.cache.read_csv_if_exists(path)
            if df is not None and not df.empty:
                return df

        # 缓存不存在或过期，获取数据
        df = self._safe_api_call(
            self.pro.daily,
            trade_date=trade_date
        )

        if df is not None and not df.empty:
            self.cache.save_csv(df, path)

        return df if df is not None else pd.DataFrame()

    def get_index_daily(self, ts_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        获取指数日线数据

        参数:
            ts_code: 指数代码（如 "000300.SH"）
            start_date: 开始日期（YYYYMMDD格式）
            end_date: 结束日期（YYYYMMDD格式）

        返回:
            指数日线数据DataFrame
        """
        df = self._safe_api_call(
            self.pro.index_daily,
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date
        )

        if df is not None and not df.empty:
            df = df.sort_values("trade_date")
        return df if df is not None else pd.DataFrame()

    def get_index_window(self, ts_code: str, trade_dates: list[str], n: int,
                        progress_callback=None) -> pd.DataFrame:
        """
        获取指数最近N个交易日数据

        参数:
            ts_code: 指数代码
            trade_dates: 交易日列表
            n: 需要的交易日数
            progress_callback: 进度回调函数

        返回:
            指数数据DataFrame
        """
        if len(trade_dates) < n:
            n = len(trade_dates)

        dates_needed = trade_dates[-n:]
        results = []

        for i, date in enumerate(dates_needed):
            path = self.cache.cache_path_index(date, ts_code)

            if self.cache.is_cache_expired(path, 365):
                df = self._safe_api_call(
                    self.pro.index_daily,
                    ts_code=ts_code,
                    start_date=date,
                    end_date=date
                )
                if df is not None and not df.empty:
                    self.cache.save_csv(df, path)
            else:
                df = self.cache.read_csv_if_exists(path)

            if df is not None and not df.empty:
                results.append(df)

            if progress_callback:
                progress_callback(1)

            if self.rate_limiter:
                self.rate_limiter.wait_if_needed()
                self.rate_limiter.sleep(SLEEP_PER_CALL)

        if results:
            return pd.concat(results, ignore_index=True).sort_values("trade_date")
        return pd.DataFrame()

    def get_daily_window(self, trade_dates: list[str], n: int,
                        progress_callback=None) -> pd.DataFrame:
        """
        获取全市场最近N个交易日数据

        参数:
            trade_dates: 交易日列表
            n: 需要的交易日数
            progress_callback: 进度回调函数

        返回:
            日线数据DataFrame
        """
        if len(trade_dates) < n:
            n = len(trade_dates)

        dates_needed = trade_dates[-n:]
        results = []

        for date in dates_needed:
            df = self.get_daily_by_date(date)
            if not df.empty:
                results.append(df)

            if progress_callback:
                progress_callback(1)

            if self.rate_limiter:
                self.rate_limiter.sleep(SLEEP_PER_CALL)

        if results:
            return pd.concat(results, ignore_index=True)
        return pd.DataFrame()

    def _safe_api_call(self, api_func, *args, **kwargs) -> pd.DataFrame | None:
        """
        安全的API调用，带重试和错误处理

        参数:
            api_func: Tushare API函数
            *args, **kwargs: API参数

        返回:
            DataFrame或None
        """
        max_retries = 3
        for attempt in range(max_retries):
            try:
                if self.rate_limiter:
                    self.rate_limiter.wait_if_needed()

                result = api_func(*args, **kwargs)

                if self.rate_limiter:
                    self.rate_limiter.sleep(SLEEP_PER_CALL)

                return result

            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2
                    print(f"[警告] API调用失败，{wait_time}秒后重试: {e}")
                    time.sleep(wait_time)
                else:
                    print(f"[错误] API调用失败: {e}")
                    return None
