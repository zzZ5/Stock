"""
趋势雷达选股系统 - 增强的工具函数模块
包含线程安全限流器、并发工具、进度追踪等
"""
import os
import time
import threading
import queue
from pathlib import Path
from datetime import datetime
from typing import List, Callable, Any
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

from core.logger import get_logger


def ensure_dir(path):
    """确保目录存在"""
    path = Path(path)
    if path.is_file():
        path = path.parent
    path.mkdir(parents=True, exist_ok=True)


def days_since_mtime(path: str) -> int:
    """计算文件自修改以来的天数"""
    path = Path(path)
    if not path.exists():
        return float('inf')

    mtime = path.stat().st_mtime
    mtime_dt = datetime.fromtimestamp(mtime)
    days = (datetime.now() - mtime_dt).days
    return days


class RateLimiter:
    """线程安全的API调用限流器"""

    def __init__(self, max_calls_per_minute: int = 200):
        """
        初始化限流器

        参数:
            max_calls_per_minute: 每分钟最大调用次数
        """
        self.max_calls = max_calls_per_minute
        self.calls = []
        self.lock = threading.Lock()
        self.logger = get_logger()

    def wait_if_needed(self):
        """如果需要，等待以遵守速率限制"""
        with self.lock:
            now = time.time()

            # 清理超过1分钟的调用记录
            self.calls = [t for t in self.calls if now - t < 60]

            # 如果达到限制，等待
            if len(self.calls) >= self.max_calls:
                sleep_time = 60 - (now - self.calls[0]) + 0.1
                if sleep_time > 0:
                    self.logger.debug(f"达到速率限制，等待 {sleep_time:.2f} 秒")
                    time.sleep(sleep_time)

            # 记录这次调用
            self.calls.append(time.time())

    def sleep(self, seconds: float):
        """额外等待"""
        time.sleep(seconds)

    def reset(self):
        """重置限流器"""
        with self.lock:
            self.calls.clear()
            self.logger.debug("限流器已重置")


class ConcurrentRateLimiter:
    """支持并发的速率限制器 - 基于令牌桶算法"""

    def __init__(self, max_rate: float = 200.0, capacity: int = 200):
        """
        初始化令牌桶限流器

        参数:
            max_rate: 每秒最大速率
            capacity: 桶容量
        """
        self.max_rate = max_rate / 60.0  # 转换为每秒速率
        self.capacity = capacity
        self.tokens = capacity
        self.last_refill = time.time()
        self.lock = threading.Lock()
        self.logger = get_logger()

    def _refill(self):
        """补充令牌"""
        now = time.time()
        elapsed = now - self.last_refill
        tokens_to_add = elapsed * self.max_rate

        self.tokens = min(self.capacity, self.tokens + tokens_to_add)
        self.last_refill = now

    def acquire(self, timeout: float = 60.0) -> bool:
        """
        获取令牌

        参数:
            timeout: 超时时间(秒)

        返回:
            是否成功获取令牌
        """
        start_time = time.time()

        while time.time() - start_time < timeout:
            with self.lock:
                self._refill()

                if self.tokens >= 1:
                    self.tokens -= 1
                    return True

            # 短暂等待
            time.sleep(0.01)

        return False

    def wait_if_needed(self):
        """等待直到可以获取令牌"""
        if not self.acquire(timeout=60.0):
            raise Exception("限流器超时: 无法在60秒内获取令牌")


class ProgressTracker:
    """进度追踪器"""

    def __init__(self, total: int, desc: str = "Progress", 
                 verbose: bool = True):
        """
        初始化进度追踪器

        参数:
            total: 总数量
            desc: 描述信息
            verbose: 是否显示进度
        """
        self.total = total
        self.current = 0
        self.desc = desc
        self.start_time = time.time()
        self.verbose = verbose
        self.lock = threading.Lock()
        self.logger = get_logger()

    def update(self, n: int = 1):
        """
        更新进度

        参数:
            n: 增加的数量
        """
        with self.lock:
            self.current += n

        if self.verbose and self.current % 10 == 0:
            self._print_progress()

    def _print_progress(self):
        """打印进度"""
        percent = self.current / self.total * 100 if self.total > 0 else 0
        elapsed = time.time() - self.start_time

        if self.current > 0:
            eta = elapsed / self.current * (self.total - self.current)
            eta_str = f"{int(eta)}s"
            rate = self.current / elapsed
        else:
            eta_str = "?"
            rate = 0

        print(f"\r[{self.desc}] {self.current}/{self.total} ({percent:.1f}%) "
              f"Rate: {rate:.2f}/s ETA: {eta_str}", end="", flush=True)

    def finish(self):
        """完成进度"""
        elapsed = time.time() - self.start_time
        if self.verbose:
            print(f"\r[{self.desc}] 完成! 耗时: {elapsed:.1f}s")
            self.logger.info(f"{self.desc} 完成, 耗时: {elapsed:.1f}s")


class ThreadPool:
    """线程池工具"""

    def __init__(self, max_workers: int = 4, 
                 rate_limiter: RateLimiter = None):
        """
        初始化线程池

        参数:
            max_workers: 最大工作线程数
            rate_limiter: 速率限制器
        """
        self.max_workers = max_workers
        self.rate_limiter = rate_limiter
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.logger = get_logger()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown(wait=True)

    def submit(self, func: Callable, *args, **kwargs) -> Any:
        """
        提交任务到线程池

        参数:
            func: 要执行的函数
            *args: 位置参数
            **kwargs: 关键字参数

        返回:
            Future对象
        """
        # 如果有速率限制器，先等待
        if self.rate_limiter:
            self.rate_limiter.wait_if_needed()

        return self.executor.submit(func, *args, **kwargs)

    def map(self, func: Callable, items: List[Any]) -> List[Any]:
        """
        并发执行函数对列表中的每个元素

        参数:
            func: 要执行的函数
            items: 元素列表

        返回:
            结果列表
        """
        futures = []
        for item in items:
            future = self.submit(func, item)
            futures.append(future)

        results = []
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                self.logger.error(f"任务执行失败: {e}")
                results.append(None)

        return results

    def shutdown(self, wait: bool = True):
        """
        关闭线程池

        参数:
            wait: 是否等待所有任务完成
        """
        self.executor.shutdown(wait=wait)
        self.logger.info("线程池已关闭")


class ProcessPool:
    """进程池工具 - CPU密集型任务"""

    def __init__(self, max_workers: int = None):
        """
        初始化进程池

        参数:
            max_workers: 最大工作进程数，None则自动检测
        """
        self.max_workers = max_workers
        self.executor = None
        self.logger = get_logger()

    def __enter__(self):
        self.executor = ProcessPoolExecutor(max_workers=self.max_workers)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.executor:
            self.executor.shutdown(wait=True)
            self.logger.info("进程池已关闭")

    def map(self, func: Callable, items: List[Any]) -> List[Any]:
        """
        并发执行函数对列表中的每个元素

        参数:
            func: 要执行的函数
            items: 元素列表

        返回:
            结果列表
        """
        if not self.executor:
            raise Exception("进程池未初始化，请使用with语句")

        results = list(self.executor.map(func, items))
        return results


class BatchProcessor:
    """批量处理器"""

    def __init__(self, batch_size: int = 100, 
                 max_workers: int = 4,
                 progress_bar: bool = True):
        """
        初始化批量处理器

        参数:
            batch_size: 每批处理的大小
            max_workers: 最大并发数
            progress_bar: 是否显示进度条
        """
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.progress_bar = progress_bar
        self.logger = get_logger()

    def process(self, items: List[Any], 
               func: Callable,
               batch_func: Callable = None) -> List[Any]:
        """
        批量处理项目

        参数:
            items: 要处理的项目列表
            func: 单个项目的处理函数
            batch_func: 批量处理函数(可选)

        返回:
            结果列表
        """
        total_items = len(items)
        progress = ProgressTracker(total_items, "Batch Processing", 
                               verbose=self.progress_bar)

        results = []

        # 使用线程池处理
        with ThreadPool(max_workers=self.max_workers) as pool:
            futures = []

            for item in items:
                future = pool.submit(func, item)
                futures.append(future)

            for i, future in enumerate(as_completed(futures), 1):
                try:
                    result = future.result()
                    results.append(result)
                    progress.update(1)
                except Exception as e:
                    self.logger.error(f"处理项目失败: {e}")
                    results.append(None)
                    progress.update(1)

        progress.finish()
        return results

    def process_batches(self, items: List[Any],
                       batch_func: Callable) -> List[Any]:
        """
        按批处理项目

        参数:
            items: 要处理的项目列表
            batch_func: 批量处理函数

        返回:
            结果列表
        """
        total_items = len(items)
        batches = [items[i:i + self.batch_size] 
                  for i in range(0, total_items, self.batch_size)]

        progress = ProgressTracker(len(batches), "Batch Processing",
                               verbose=self.progress_bar)

        results = []

        with ThreadPool(max_workers=self.max_workers) as pool:
            futures = []

            for batch in batches:
                future = pool.submit(batch_func, batch)
                futures.append(future)

            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.extend(result)
                except Exception as e:
                    self.logger.error(f"批量处理失败: {e}")

                progress.update(1)

        progress.finish()
        return results


def retry_on_failure(max_retries: int = 3, 
                    delay: float = 1.0,
                    backoff: float = 2.0,
                    exceptions: tuple = (Exception,)):
    """
    重试装饰器

    参数:
        max_retries: 最大重试次数
        delay: 初始延迟(秒)
        backoff: 退避因子
        exceptions: 要捕获的异常类型

    返回:
        装饰器函数

    使用示例:
        @retry_on_failure(max_retries=3, delay=1.0)
        def fetch_data():
            # 获取数据
            ...
    """
    def decorator(func):
        logger = get_logger()

        def wrapper(*args, **kwargs):
            retries = 0
            current_delay = delay

            while retries <= max_retries:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    retries += 1
                    if retries > max_retries:
                        logger.error(f"函数 {func.__name__} 重试 {max_retries} 次后仍失败: {e}")
                        raise

                    logger.warning(f"函数 {func.__name__} 失败, {current_delay}秒后重试 ({retries}/{max_retries}): {e}")
                    time.sleep(current_delay)
                    current_delay *= backoff

        return wrapper
    return decorator
