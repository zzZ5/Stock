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
from config.settings import settings


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

        if self.cache.is_cache_expired(path, settings.TRADE_CAL_TTL_DAYS):
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

        if self.cache.is_cache_expired(path, settings.STOCK_BASIC_TTL_DAYS):
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

    def get_daily_by_date(self, trade_date: str, skip_delay=False) -> pd.DataFrame:
        """
        获取某交易日全市场日线数据，优先使用缓存

        参数:
            trade_date: 交易日（YYYYMMDD格式）
            skip_delay: 是否跳过延迟（用于并发场景）

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

        # 根据参数选择API调用方法
        if skip_delay:
            df = self._safe_api_call_no_delay(
                self.pro.daily,
                trade_date=trade_date
            )
        else:
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

    def get_index_daily(self, ts_code: str, start_date: str, end_date: str,
                       enable_cache=True) -> pd.DataFrame:
        """
        获取指数日线数据（支持缓存）

        参数:
            ts_code: 指数代码（如 "000300.SH"）
            start_date: 开始日期（YYYYMMDD格式）
            end_date: 结束日期（YYYYMMDD格式）
            enable_cache: 是否使用缓存（默认True）

        返回:
            指数日线数据DataFrame
        """
        self.logger.debug(f"获取指数日线数据: {ts_code}, {start_date}~{end_date}")

        # 验证参数
        DateValidator.validate_date_range(start_date, end_date)
        if not ts_code or not isinstance(ts_code, str) or len(ts_code) < 6:
            raise ValidationError(f"指数代码无效: {ts_code}")

        # 尝试从缓存加载（使用范围路径）
        cache_path = self.cache.cache_path_index_range(ts_code, start_date, end_date)

        if enable_cache:
            # 检查缓存是否存在且未过期（指数缓存1天）
            if not self.cache.is_cache_expired(cache_path, 1):
                df_cached = self.cache.read_csv_if_exists(cache_path)
                if df_cached is not None and not df_cached.empty:
                    # 检查缓存的数据是否覆盖请求的日期范围
                    cached_start = df_cached['trade_date'].min()
                    cached_end = df_cached['trade_date'].max()
                    if cached_start <= start_date and cached_end >= end_date:
                        df_result = df_cached[
                            (df_cached['trade_date'] >= start_date) &
                            (df_cached['trade_date'] <= end_date)
                        ].copy()
                        self.logger.debug(f"指数数据：从缓存获取{len(df_result)}条记录")
                        return df_result

        # 缓存未命中或数据不完整，从API获取
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

                # 保存到缓存
                if enable_cache:
                    self.cache.save_csv(df, cache_path)

            except ValidationError as e:
                self.logger.warning(f"{ts_code}指数数据验证失败: {e}")
                return pd.DataFrame()

        return df if df is not None else pd.DataFrame()

    def get_index_window(self, ts_code: str, trade_dates: list[str], n: int,
                        progress_callback=None) -> pd.DataFrame:
        """
        获取指数最近N个交易日数据（使用范围查询优化）

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
        start_date = dates_needed[0]
        end_date = dates_needed[-1]

        self.logger.debug(f"指数窗口: 请求{ts_code}的{len(dates_needed)}天数据 ({start_date}~{end_date})")

        # 使用范围缓存路径（包含整个日期范围）
        cache_path = self.cache.cache_path_index_range(ts_code, start_date, end_date)

        # 先尝试使用缓存
        if not self.cache.is_cache_expired(cache_path, 1):  # 指数数据缓存1天
            df_cached = self.cache.read_csv_if_exists(cache_path)
            if df_cached is not None and not df_cached.empty:
                # 确保trade_date是字符串格式（统一类型）
                df_cached['trade_date'] = df_cached['trade_date'].astype(str)

                # 检查缓存的数据是否包含需要的日期
                cached_dates = set(df_cached['trade_date'].tolist())
                needed_dates = set(dates_needed)

                # 调试信息
                self.logger.debug(f"指数缓存检查: 需要{len(needed_dates)}个日期，缓存有{len(cached_dates)}个日期")
                if not needed_dates.issubset(cached_dates):
                    missing = needed_dates - cached_dates
                    self.logger.warning(f"指数缓存缺少{len(missing)}个日期，前5个: {list(missing)[:5]}")

                if needed_dates.issubset(cached_dates):
                    # 缓存数据完整，直接返回
                    df_result = df_cached[df_cached['trade_date'].isin(dates_needed)]
                    self.logger.info(f"指数数据：从缓存获取{len(df_result)}条记录")
                    if progress_callback:
                        progress_callback(len(dates_needed))
                    return df_result.sort_values("trade_date")

        # 缓存未命中，从API获取
        self.logger.info(f"指数数据：从API获取 {start_date}~{end_date}")

        df = self._safe_api_call(
            self.pro.index_daily,
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date
        )

        if df is not None and not df.empty:
            self.logger.info(f"指数数据：API返回{len(df)}条记录")

            # 确保trade_date是字符串格式
            df['trade_date'] = df['trade_date'].astype(str)

            # 保存到缓存
            self.cache.save_csv(df, cache_path)

            # 过滤出需要的日期
            df_result = df[df['trade_date'].isin(dates_needed)]
            self.logger.info(f"指数数据：过滤后剩余{len(df_result)}条需要的记录")

            if progress_callback:
                progress_callback(len(dates_needed))

            return df_result.sort_values("trade_date")
        else:
            self.logger.error(f"指数数据获取失败: {ts_code}，返回空数据")
            return pd.DataFrame()

    def get_daily_window(self, trade_dates: list[str], n: int,
                        progress_callback=None, use_concurrent=True) -> pd.DataFrame:
        """
        获取全市场最近N个交易日数据（支持并发）

        参数:
            trade_dates: 交易日列表
            n: 需要的交易日数
            progress_callback: 进度回调函数
            use_concurrent: 是否使用并发获取（默认True）

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

        # 使用并发获取以提升速度
        if use_concurrent and len(dates_needed) > 10:
            return self._get_daily_window_concurrent(dates_needed, progress_callback)

        # 单线程模式（用于少量数据或调试）
        results = []
        for date in dates_needed:
            try:
                df = self.get_daily_by_date(date)
                if not df.empty:
                    results.append(df)

                if progress_callback:
                    progress_callback(1)

                if self.rate_limiter:
                    self.rate_limiter.sleep(settings.SLEEP_PER_CALL)
            except Exception as e:
                self.logger.error(f"获取{date}日线数据失败: {e}")
                continue

        if results:
            return pd.concat(results, ignore_index=True)
        return pd.DataFrame()

    def _get_daily_window_concurrent(self, dates_needed: list[str],
                                     progress_callback=None) -> pd.DataFrame:
        """
        并发获取多日日线数据
        """
        import threading
        import queue
        import time

        self.logger.info(f"使用并发模式获取{len(dates_needed)}天的日线数据")

        results = []
        result_queue = queue.Queue()
        completed = [0]
        start_time = time.time()

        # 并发数：控制同时获取的天数
        max_workers = 5

        def fetch_single_date(date):
            """获取单日数据的线程函数"""
            try:
                df = self.get_daily_by_date(date, skip_delay=True)
                if df is not None and not df.empty:
                    result_queue.put(df)
            except Exception as e:
                self.logger.debug(f"获取{date}日线数据失败: {e}")
            finally:
                with threading.Lock():
                    completed[0] += 1
                    if progress_callback:
                        progress_callback(1)

        # 分批处理
        batch_size = max_workers * 2
        for i in range(0, len(dates_needed), batch_size):
            batch = dates_needed[i:i + batch_size]

            # 为当前批次创建线程
            threads = []
            for date in batch:
                thread = threading.Thread(target=fetch_single_date, args=(date,))
                thread.start()
                threads.append(thread)

            # 等待当前批次完成
            for thread in threads:
                thread.join()

            # 批次间短暂延迟，确保速率限制
            if i + batch_size < len(dates_needed):
                time.sleep(0.3)

            # 记录进度
            if completed[0] > 0:
                elapsed = time.time() - start_time
                remaining = len(dates_needed) - completed[0]
                avg_time = elapsed / completed[0]
                eta = remaining * avg_time
                self.logger.debug(
                    f"日线数据: 已完成{completed[0]}/{len(dates_needed)} "
                    f"预计剩余时间: {int(eta)}秒"
                )

        # 收集结果
        while not result_queue.empty():
            results.append(result_queue.get())

        elapsed = time.time() - start_time

        # 显示等待统计信息
        if self.rate_limiter:
            wait_count = self.rate_limiter.get_wait_count()
            if wait_count > 0:
                print(f"\nℹ️  日线数据获取过程中因API限制共等待{wait_count}次")

        self.logger.info(f"并发获取日线数据完成，成功获取{len(results)}/{len(dates_needed)}天，耗时{elapsed:.1f}秒")

        if results:
            return pd.concat(results, ignore_index=True)
        return pd.DataFrame()

    def get_daily_multi(self, ts_codes: list[str], start_date: str, end_date: str,
                       progress_callback=None, use_concurrent=True, enable_cache=True) -> pd.DataFrame:
        """
        批量获取多只股票的日线数据（支持并发和缓存）

        参数:
            ts_codes: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            progress_callback: 进度回调函数
            use_concurrent: 是否使用并发获取（默认True）
            enable_cache: 是否使用缓存（默认True）

        返回:
            日线数据DataFrame
        """
        self.logger.debug(f"批量获取{len(ts_codes)}只股票日线数据: {start_date}~{end_date}")

        DateValidator.validate_date_range(start_date, end_date)

        results = []
        codes_to_fetch = []

        # 第一轮：尝试从缓存加载
        if enable_cache:
            self.logger.debug("日线数据：尝试从缓存加载...")
            for code in ts_codes:
                cache_path = self.cache.cache_path_stock_daily(code, start_date, end_date)

                if not self.cache.is_cache_expired(cache_path, 1):  # 日线缓存1天
                    df = self.cache.read_csv_if_exists(cache_path)
                    if df is not None and not df.empty:
                        results.append(df)
                        if progress_callback:
                            progress_callback(1)
                    else:
                        codes_to_fetch.append(code)
                else:
                    codes_to_fetch.append(code)

            self.logger.info(f"日线数据：缓存命中{len(results)}/{len(ts_codes)}只股票")
        else:
            codes_to_fetch = ts_codes.copy()

        # 第二轮：获取未缓存的数据
        if codes_to_fetch:
            self.logger.debug(f"日线数据：需要从API获取{len(codes_to_fetch)}只股票...")

            # 使用并发获取以提升速度
            if use_concurrent and len(codes_to_fetch) > 50:
                df_api = self._get_data_concurrent(
                    self.pro.daily, codes_to_fetch, start_date, end_date,
                    progress_callback, "日线数据"
                )
                if not df_api.empty:
                    results.append(df_api)

                    # 保存到缓存
                    if enable_cache:
                        for code in codes_to_fetch:
                            df_single = df_api[df_api['ts_code'] == code]
                            if not df_single.empty:
                                cache_path = self.cache.cache_path_stock_daily(code, start_date, end_date)
                                self.cache.save_csv(df_single, cache_path)
            else:
                # 单线程模式（用于少量股票或调试）
                for code in codes_to_fetch:
                    try:
                        df = self._safe_api_call(
                            self.pro.daily,
                            ts_code=code,
                            start_date=start_date,
                            end_date=end_date
                        )
                        if df is not None and not df.empty:
                            results.append(df)
                            # 保存到缓存
                            if enable_cache:
                                cache_path = self.cache.cache_path_stock_daily(code, start_date, end_date)
                                self.cache.save_csv(df, cache_path)

                        if progress_callback:
                            progress_callback(1)

                        if self.rate_limiter:
                            self.rate_limiter.sleep(settings.SLEEP_PER_CALL)
                    except Exception as e:
                        self.logger.error(f"获取{code}日线数据失败: {e}")
                        continue

        if results:
            return pd.concat(results, ignore_index=True)
        return pd.DataFrame()

    def get_weekly_data(self, ts_codes: list[str], start_date: str, end_date: str,
                    progress_callback=None, use_concurrent=True, enable_cache=True) -> pd.DataFrame:
        """
        批量获取多只股票的周线数据（支持缓存）

        参数:
            ts_codes: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            progress_callback: 进度回调函数
            use_concurrent: 是否使用并发获取（默认True）
            enable_cache: 是否使用缓存（默认True）

        返回:
            周线数据DataFrame
        """
        self.logger.debug(f"批量获取{len(ts_codes)}只股票周线数据: {start_date}~{end_date}")

        DateValidator.validate_date_range(start_date, end_date)

        results = []
        codes_to_fetch = []

        # 第一轮：尝试从缓存加载
        if enable_cache:
            self.logger.debug("周线数据：尝试从缓存加载...")
            for code in ts_codes:
                path = self.cache.cache_path_weekly(code, start_date, end_date)
                if not self.cache.is_cache_expired(path, 7):  # 周线缓存7天
                    df = self.cache.read_csv_if_exists(path)
                    if df is not None and not df.empty:
                        results.append(df)
                        if progress_callback:
                            progress_callback(1)
                    else:
                        codes_to_fetch.append(code)
                else:
                    codes_to_fetch.append(code)

            self.logger.info(f"周线数据：缓存命中{len(results)}/{len(ts_codes)}只股票")
        else:
            codes_to_fetch = ts_codes.copy()

        # 第二轮：获取未缓存的数据
        if codes_to_fetch:
            self.logger.debug(f"周线数据：需要从API获取{len(codes_to_fetch)}只股票...")

            # 使用并发获取以提升速度
            if use_concurrent and len(codes_to_fetch) > 50:
                df_api = self._get_data_concurrent(
                    self.pro.weekly, codes_to_fetch, start_date, end_date,
                    progress_callback, "周线数据"
                )
                if not df_api.empty:
                    results.append(df_api)

                    # 保存到缓存
                    if enable_cache:
                        for code in codes_to_fetch:
                            df_single = df_api[df_api['ts_code'] == code]
                            if not df_single.empty:
                                path = self.cache.cache_path_weekly(code, start_date, end_date)
                                self.cache.save_csv(df_single, path)
            else:
                # 单线程模式（用于少量股票或调试）
                for code in codes_to_fetch:
                    try:
                        df = self._safe_api_call(
                            self.pro.weekly,
                            ts_code=code,
                            start_date=start_date,
                            end_date=end_date
                        )
                        if df is not None and not df.empty:
                            results.append(df)
                            # 保存到缓存
                            if enable_cache:
                                path = self.cache.cache_path_weekly(code, start_date, end_date)
                                self.cache.save_csv(df, path)

                        if progress_callback:
                            progress_callback(1)

                        if self.rate_limiter:
                            self.rate_limiter.sleep(settings.SLEEP_PER_CALL)
                    except Exception as e:
                        self.logger.error(f"获取{code}周线数据失败: {e}")
                        continue

        if results:
            return pd.concat(results, ignore_index=True)
        return pd.DataFrame()

    def get_monthly_data(self, ts_codes: list[str], start_date: str, end_date: str,
                     progress_callback=None, use_concurrent=True, enable_cache=True) -> pd.DataFrame:
        """
        批量获取多只股票的月线数据（支持缓存）

        参数:
            ts_codes: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            progress_callback: 进度回调函数
            use_concurrent: 是否使用并发获取（默认True）
            enable_cache: 是否使用缓存（默认True）

        返回:
            月线数据DataFrame
        """
        self.logger.debug(f"批量获取{len(ts_codes)}只股票月线数据: {start_date}~{end_date}")

        DateValidator.validate_date_range(start_date, end_date)

        results = []
        codes_to_fetch = []

        # 第一轮：尝试从缓存加载
        if enable_cache:
            self.logger.debug("月线数据：尝试从缓存加载...")
            for code in ts_codes:
                path = self.cache.cache_path_monthly(code, start_date, end_date)
                if not self.cache.is_cache_expired(path, 30):  # 月线缓存30天
                    df = self.cache.read_csv_if_exists(path)
                    if df is not None and not df.empty:
                        results.append(df)
                        if progress_callback:
                            progress_callback(1)
                    else:
                        codes_to_fetch.append(code)
                else:
                    codes_to_fetch.append(code)

            self.logger.info(f"月线数据：缓存命中{len(results)}/{len(ts_codes)}只股票")
        else:
            codes_to_fetch = ts_codes.copy()

        # 第二轮：获取未缓存的数据
        if codes_to_fetch:
            self.logger.debug(f"月线数据：需要从API获取{len(codes_to_fetch)}只股票...")

            # 使用并发获取以提升速度
            if use_concurrent and len(codes_to_fetch) > 50:
                df_api = self._get_data_concurrent(
                    self.pro.monthly, codes_to_fetch, start_date, end_date,
                    progress_callback, "月线数据"
                )
                if not df_api.empty:
                    results.append(df_api)

                    # 保存到缓存
                    if enable_cache:
                        for code in codes_to_fetch:
                            df_single = df_api[df_api['ts_code'] == code]
                            if not df_single.empty:
                                path = self.cache.cache_path_monthly(code, start_date, end_date)
                                self.cache.save_csv(df_single, path)
            else:
                # 单线程模式（用于少量股票或调试）
                for code in codes_to_fetch:
                    try:
                        df = self._safe_api_call(
                            self.pro.monthly,
                            ts_code=code,
                            start_date=start_date,
                            end_date=end_date
                        )
                        if df is not None and not df.empty:
                            results.append(df)
                            # 保存到缓存
                            if enable_cache:
                                path = self.cache.cache_path_monthly(code, start_date, end_date)
                                self.cache.save_csv(df, path)

                        if progress_callback:
                            progress_callback(1)

                        if self.rate_limiter:
                            self.rate_limiter.sleep(settings.SLEEP_PER_CALL)
                    except Exception as e:
                        self.logger.error(f"获取{code}月线数据失败: {e}")
                        continue

        if results:
            return pd.concat(results, ignore_index=True)
        return pd.DataFrame()

    def _get_data_concurrent(self, api_func, ts_codes: list[str], start_date: str, end_date: str,
                          progress_callback, data_type: str) -> pd.DataFrame:
        """
        使用并发方式获取数据（适用于大量股票）
        严格遵守每分钟200次API调用的限制

        参数:
            api_func: API函数（pro.weekly 或 pro.monthly）
            ts_codes: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            progress_callback: 进度回调函数
            data_type: 数据类型名称（用于日志）

        返回:
            DataFrame
        """
        import threading
        import queue
        import time

        self.logger.info(f"使用并发模式获取{len(ts_codes)}只股票的{data_type}")

        results = []
        result_queue = queue.Queue()
        completed = [0]
        start_time = time.time()

        # 根据限流器计算并发数：确保不会超过速率限制
        # 如果每分钟200次，考虑安全裕度，设置并发数为3-5
        max_workers = 4 if self.rate_limiter and self.rate_limiter.max_calls >= 200 else 3

        # 批次大小：每批处理max_workers只股票
        batch_size = max_workers * 2

        def fetch_single_code(code):
            """获取单只股票数据的线程函数"""
            try:
                df = self._safe_api_call_no_delay(
                    api_func,
                    ts_code=code,
                    start_date=start_date,
                    end_date=end_date
                )
                if df is not None and not df.empty:
                    result_queue.put(df)
            except Exception as e:
                self.logger.debug(f"获取{code}{data_type}失败: {e}")
            finally:
                with threading.Lock():
                    completed[0] += 1
                    if progress_callback:
                        progress_callback(1)

        # 分批处理以控制并发
        for i in range(0, len(ts_codes), batch_size):
            batch = ts_codes[i:i + batch_size]

            # 为当前批次创建线程
            threads = []
            for code in batch:
                thread = threading.Thread(target=fetch_single_code, args=(code,))
                thread.start()
                threads.append(thread)

            # 等待当前批次完成
            for thread in threads:
                thread.join()

            # 如果还有下一批，短暂休息以确保不超过速率限制
            if i + batch_size < len(ts_codes):
                time.sleep(0.5)

            # 记录进度和预计时间
            elapsed = time.time() - start_time
            if completed[0] > 0:
                remaining = len(ts_codes) - completed[0]
                avg_time = elapsed / completed[0]
                eta = remaining * avg_time
                self.logger.debug(
                    f"已完成{completed[0]}/{len(ts_codes)} ({completed[0]/len(ts_codes)*100:.1f}%) "
                    f"预计剩余时间: {int(eta)}秒"
                )

        # 收集结果
        while not result_queue.empty():
            results.append(result_queue.get())

        elapsed = time.time() - start_time

        # 显示等待统计信息
        if self.rate_limiter:
            wait_count = self.rate_limiter.get_wait_count()
            if wait_count > 0:
                print(f"\nℹ️  {data_type}获取过程中因API限制共等待{wait_count}次")

        self.logger.info(f"并发获取{data_type}完成，成功获取{len(results)}/{len(ts_codes)}只股票，耗时{elapsed:.1f}秒")

        if results:
            return pd.concat(results, ignore_index=True)
        return pd.DataFrame()

    def _safe_api_call_no_delay(self, api_func, *args, **kwargs) -> pd.DataFrame | None:
        """
        安全的API调用（不带延迟），用于并发场景
        """
        max_retries = 3
        last_error = None

        for attempt in range(max_retries):
            try:
                if self.rate_limiter:
                    self.rate_limiter.wait_if_needed()

                result = api_func(*args, **kwargs)
                return result

            except Exception as e:
                last_error = e
                # IP限制错误不重试，直接返回None
                if "IP数量超限" in str(e):
                    self.logger.debug(f"API调用IP受限，跳过该请求: {e}")
                    return None
                elif attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2
                    self.logger.warning(f"API调用失败(尝试{attempt+1}/{max_retries})，{wait_time}秒后重试: {e}")
                    time.sleep(wait_time)
                else:
                    self.logger.error(f"API调用失败(尝试{attempt+1}/{max_retries}): {e}")

        return None

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
                    self.rate_limiter.sleep(settings.SLEEP_PER_CALL)

                return result

            except Exception as e:
                last_error = e
                # IP限制错误不重试，直接返回None
                if "IP数量超限" in str(e):
                    self.logger.debug(f"API调用IP受限，跳过该请求: {e}")
                    return None
                elif attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2
                    self.logger.warning(f"API调用失败(尝试{attempt+1}/{max_retries})，{wait_time}秒后重试: {e}")
                    time.sleep(wait_time)
                else:
                    self.logger.error(f"API调用失败(尝试{attempt+1}/{max_retries}): {e}")

        return None
