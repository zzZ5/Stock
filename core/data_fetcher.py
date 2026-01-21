"""
趋势雷达选股系统 - 数据获取模块
整合原版和优化版功能，支持并发、批量获取、重试机制
"""
import pandas as pd
import tushare as ts
from datetime import datetime
from dateutil.relativedelta import relativedelta
import time
from typing import List, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

from .cache_manager import CacheManager
from .utils import RateLimiter
from .logger import get_datafetcher_logger
from .validators import (
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


class DataFetcher:
    """数据获取器"""

    def __init__(self, token: str, rate_limiter: RateLimiter = None):
        """
        初始化数据获取器

        参数:
            token: Tushare API token
            rate_limiter: 限流器实例
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

        self.cache = CacheManager()
        self.rate_limiter = rate_limiter or RateLimiter(200)
        self.logger.info("DataFetcher初始化完成")

    def get_trade_cal(self, end_date: str, lookback_calendar_days: int = 500) -> list[str]:
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

        path = self.cache.cache_path_trade_cal()

        if self.cache.is_cache_expired(path, TRADE_CAL_TTL_DAYS):
            # 缓存过期，重新获取
            self.logger.info("交易日历缓存过期，重新获取...")
            start = (datetime.strptime(end_date, "%Y%m%d") -
                    relativedelta(days=lookback_calendar_days)).strftime("%Y%m%d")
            cal = self._safe_api_call(
                self.pro.trade_cal,
                exchange="",
                start_date=start,
                end_date=end_date
            )
            if cal is not None:
                # 验证返回数据
                if 'cal_date' not in cal.columns or 'is_open' not in cal.columns:
                    self.logger.error("交易日历数据格式错误")
                    return []

                self.cache.save_csv(cal, path)
                self.logger.info(f"交易日历已保存，共{len(cal)}条记录")
            else:
                self.logger.warning("交易日历获取失败")
        else:
            self.logger.debug("使用缓存的交易日历")

        # 读取缓存
        cal = self.cache.read_csv_if_exists(path)
        if cal is None or cal.empty:
            self.logger.error("交易日历为空")
            return []

        cal = cal[cal["is_open"] == 1].sort_values("cal_date")
        result = cal["cal_date"].astype(str).tolist()
        self.logger.debug(f"交易日历获取成功，共{len(result)}个交易日")
        return result

    def get_stock_basic(self) -> pd.DataFrame:
        """
        获取股票基础信息，优先使用缓存

        返回:
            包含 ts_code, name, industry, list_date 等字段的DataFrame
        """
        self.logger.debug("获取股票基础信息")

        path = self.cache.cache_path_stock_basic()

        if self.cache.is_cache_expired(path, STOCK_BASIC_TTL_DAYS):
            # 缓存过期，重新获取
            self.logger.info("股票基础信息缓存过期，重新获取...")
            basic = self._safe_api_call(
                self.pro.stock_basic,
                exchange="",
                list_status="L",
                fields="ts_code,name,industry,list_date"
            )
            if basic is not None and not basic.empty:
                self.cache.save_csv(basic, path)
                self.logger.info(f"股票基础信息已保存，共{len(basic)}只股票")
            else:
                self.logger.warning("股票基础信息获取失败")
        else:
            self.logger.debug("使用缓存的股票基础信息")

        result = self.cache.read_csv_if_exists(path)
        if result is None or result.empty:
            self.logger.error("股票基础信息为空")
            return pd.DataFrame()

        self.logger.debug(f"股票基础信息获取成功，共{len(result)}只股票")
        return result

    def get_daily_by_date(self, trade_date: str) -> pd.DataFrame:
        """
        获取某交易日全市场日线数据，优先使用缓存

        参数:
            trade_date: 交易日（YYYYMMDD格式）

        返回:
            包含 ts_code, trade_date, open, high, low, close, amount 等字段的DataFrame
        """
        self.logger.debug(f"获取{trade_date}的日线数据")

        # 验证参数
        DateValidator.validate_date_format(trade_date, date_formats=["%Y%m%d"])

        path = self.cache.cache_path_daily(trade_date)

        if not self.cache.is_cache_expired(path, 365):  # 日线数据缓存1年
            df = self.cache.read_csv_if_exists(path)
            if df is not None and not df.empty:
                self.logger.debug(f"使用缓存的{trade_date}日线数据，共{len(df)}条")
                return df

        # 缓存不存在或过期，获取数据
        self.logger.info(f"从API获取{trade_date}日线数据...")
        df = self._safe_api_call(
            self.pro.daily,
            trade_date=trade_date
        )

        if df is not None and not df.empty:
            # 验证数据
            try:
                required_cols = ['ts_code', 'trade_date', 'open', 'high', 'low', 'close', 'amount']
                df = DataFrameValidator.validate_dataframe(df, required_cols, f"{trade_date}日线数据")
                df = DataFrameValidator.validate_numeric_columns(
                    df, ['open', 'high', 'low', 'close', 'amount'], f"{trade_date}日线数据"
                )
                # 验证OHLC数据
                df = PriceValidator.validate_ohlc(df)

                self.cache.save_csv(df, path)
                self.logger.info(f"{trade_date}日线数据已保存，共{len(df)}条")
            except ValidationError as e:
                self.logger.warning(f"{trade_date}日线数据验证失败: {e}")
                return pd.DataFrame()
        else:
            self.logger.warning(f"{trade_date}日线数据获取失败")

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
        self.logger.debug(f"获取指数日线数据: {ts_code}, {start_date}~{end_date}")

        # 验证参数
        DateValidator.validate_date_range(start_date, end_date)
        if not ts_code or not isinstance(ts_code, str) or len(ts_code) < 6:
            raise ValidationError(f"指数代码无效: {ts_code}")

        df = self._safe_api_call(
            self.pro.index_daily,
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date
        )

        if df is not None and not df.empty:
            # 验证数据
            try:
                required_cols = ['trade_date', 'open', 'high', 'low', 'close', 'amount']
                df = DataFrameValidator.validate_dataframe(df, required_cols, f"{ts_code}指数数据")
                df = DataFrameValidator.validate_numeric_columns(
                    df, ['open', 'high', 'low', 'close', 'amount'], f"{ts_code}指数数据"
                )
                df = PriceValidator.validate_ohlc(df)
                df = df.sort_values("trade_date")
            except ValidationError as e:
                self.logger.warning(f"{ts_code}指数数据验证失败: {e}")
                return pd.DataFrame()

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
        # 验证参数
        if not trade_dates:
            self.logger.warning("trade_dates为空")
            return pd.DataFrame()

        if n < 1:
            raise ValidationError(f"n 必须大于0: {n}")

        if len(trade_dates) < n:
            n = len(trade_dates)
            self.logger.info(f"调整n值为实际可用交易日数: {n}")

        dates_needed = trade_dates[-n:]
        results = []

        for i, date in enumerate(dates_needed):
            try:
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

            except Exception as e:
                self.logger.error(f"获取{date}指数数据失败: {e}")
                continue

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
        # 验证参数
        if not trade_dates:
            self.logger.warning("trade_dates为空")
            return pd.DataFrame()

        if n < 1:
            raise ValidationError(f"n 必须大于0: {n}")

        if len(trade_dates) < n:
            n = len(trade_dates)
            self.logger.info(f"调整n值为实际可用交易日数: {n}")

        dates_needed = trade_dates[-n:]
        results = []

        for date in dates_needed:
            try:
                df = self.get_daily_by_date(date)
                if not df.empty:
                    results.append(df)

                if progress_callback:
                    progress_callback(1)

                if self.rate_limiter:
                    self.rate_limiter.sleep(SLEEP_PER_CALL)
            except Exception as e:
                self.logger.error(f"获取{date}日线数据失败: {e}")
                continue

        if results:
            return pd.concat(results, ignore_index=True)
        return pd.DataFrame()

    def get_daily_multi(self, ts_codes: list[str], start_date: str, end_date: str) -> pd.DataFrame:
        """
        批量获取多只股票的日线数据

        参数:
            ts_codes: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期

        返回:
            日线数据DataFrame
        """
        self.logger.debug(f"批量获取{len(ts_codes)}只股票日线数据: {start_date}~{end_date}")

        DateValidator.validate_date_range(start_date, end_date)

        results = []
        for i, code in enumerate(ts_codes):
            try:
                df = self._safe_api_call(
                    self.pro.daily,
                    ts_code=code,
                    start_date=start_date,
                    end_date=end_date
                )
                if df is not None and not df.empty:
                    results.append(df)

                if self.rate_limiter:
                    self.rate_limiter.sleep(SLEEP_PER_CALL)
            except Exception as e:
                self.logger.error(f"获取{code}日线数据失败: {e}")
                continue

        if results:
            return pd.concat(results, ignore_index=True)
        return pd.DataFrame()

    def get_weekly_data(self, ts_codes: list[str], start_date: str, end_date: str) -> pd.DataFrame:
        """
        批量获取多只股票的周线数据

        参数:
            ts_codes: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期

        返回:
            周线数据DataFrame
        """
        self.logger.debug(f"批量获取{len(ts_codes)}只股票周线数据: {start_date}~{end_date}")

        DateValidator.validate_date_range(start_date, end_date)

        results = []
        for i, code in enumerate(ts_codes):
            try:
                df = self._safe_api_call(
                    self.pro.weekly,
                    ts_code=code,
                    start_date=start_date,
                    end_date=end_date
                )
                if df is not None and not df.empty:
                    results.append(df)

                if self.rate_limiter:
                    self.rate_limiter.sleep(SLEEP_PER_CALL)
            except Exception as e:
                self.logger.error(f"获取{code}周线数据失败: {e}")
                continue

        if results:
            return pd.concat(results, ignore_index=True)
        return pd.DataFrame()

    def get_monthly_data(self, ts_codes: list[str], start_date: str, end_date: str) -> pd.DataFrame:
        """
        批量获取多只股票的月线数据

        参数:
            ts_codes: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期

        返回:
            月线数据DataFrame
        """
        self.logger.debug(f"批量获取{len(ts_codes)}只股票月线数据: {start_date}~{end_date}")

        DateValidator.validate_date_range(start_date, end_date)

        results = []
        for i, code in enumerate(ts_codes):
            try:
                df = self._safe_api_call(
                    self.pro.monthly,
                    ts_code=code,
                    start_date=start_date,
                    end_date=end_date
                )
                if df is not None and not df.empty:
                    results.append(df)

                if self.rate_limiter:
                    self.rate_limiter.sleep(SLEEP_PER_CALL)
            except Exception as e:
                self.logger.error(f"获取{code}月线数据失败: {e}")
                continue

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
        last_error = None

        for attempt in range(max_retries):
            try:
                if self.rate_limiter:
                    self.rate_limiter.wait_if_needed()

                result = api_func(*args, **kwargs)

                if self.rate_limiter:
                    self.rate_limiter.sleep(SLEEP_PER_CALL)

                return result

            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2
                    self.logger.warning(f"API调用失败(尝试{attempt+1}/{max_retries})，{wait_time}秒后重试: {e}")
                    time.sleep(wait_time)
                else:
                    self.logger.error(f"API调用失败(尝试{attempt+1}/{max_retries}): {e}")

        return None
