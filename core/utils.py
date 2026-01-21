"""
趋势雷达选股系统 - 工具函数模块
包含通用工具函数、进度追踪、限流器等
"""
import os
import time
from pathlib import Path
from datetime import datetime


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
    """API调用限流器（线程安全）"""

    def __init__(self, max_calls_per_minute: int = 200):
        """
        初始化限流器

        参数:
            max_calls_per_minute: 每分钟最大调用次数
        """
        self.max_calls = max_calls_per_minute
        self.calls = []
        import threading
        self.lock = threading.Lock()
        self.wait_count = 0  # 等待次数
        self.waiting = False  # 是否正在等待
        self.wait_start_time = 0  # 开始等待的时间
        self.wait_seconds = 0  # 需要等待的秒数

    def wait_if_needed(self, show_warning=True):
        """
        如果需要，等待以遵守速率限制（线程安全）

        参数:
            show_warning: 是否显示等待提示
        """
        now = time.time()

        with self.lock:
            # 清理超过1分钟的调用记录
            self.calls = [t for t in self.calls if now - t < 60]

            # 如果达到限制，需要等待
            if len(self.calls) >= self.max_calls:
                # 等待时间：从最早一次调用算起，到60秒的时间差
                sleep_time = 60 - (now - self.calls[0]) + 0.1
                if sleep_time > 0:
                    self.wait_count += 1
                    self.waiting = True
                    self.wait_start_time = now
                    self.wait_seconds = int(sleep_time)

        # 在锁外等待，避免阻塞其他线程
        actual_sleep_time = 0
        with self.lock:
            if len(self.calls) >= self.max_calls:
                sleep_time = 60 - (now - self.calls[0]) + 0.1
                if sleep_time > 0:
                    self.lock.release()
                    time.sleep(sleep_time)
                    actual_sleep_time = time.time() - (now - (60 - sleep_time + 0.1))
                    self.lock.acquire()

        # 记录这次调用
        with self.lock:
            self.calls.append(time.time())
            self.waiting = False

    def get_wait_count(self) -> int:
        """获取等待次数"""
        with self.lock:
            return self.wait_count

    def get_wait_status(self) -> tuple:
        """获取等待状态：(是否等待, 剩余秒数, 总秒数)"""
        with self.lock:
            if self.waiting and self.wait_start_time > 0:
                elapsed = time.time() - self.wait_start_time
                remaining = max(0, self.wait_seconds - int(elapsed))
                return (True, remaining, self.wait_seconds)
            return (False, 0, 0)

    def sleep(self, seconds: float):
        """额外等待"""
        time.sleep(seconds)


class ProgressTracker:
    """进度追踪器"""

    def __init__(self, total: int, desc: str = "Progress", rate_limiter=None):
        """
        初始化进度追踪器

        参数:
            total: 总数量
            desc: 描述信息
            rate_limiter: 限流器实例（可选）
        """
        self.total = total
        self.current = 0
        self.desc = desc
        self.start_time = time.time()
        self.rate_limiter = rate_limiter

    def update(self, n: int = 1):
        """
        更新进度

        参数:
            n: 增加的数量
        """
        self.current += n
        percent = self.current / self.total * 100 if self.total > 0 else 0
        elapsed = time.time() - self.start_time

        if self.current > 0:
            eta = elapsed / self.current * (self.total - self.current)
            eta_str = f"{int(eta)}s"
        else:
            eta_str = "?"

        # 获取API等待状态
        status_parts = [f"ETA: {eta_str}"]
        if self.rate_limiter:
            is_waiting, remaining, total_wait = self.rate_limiter.get_wait_status()
            if is_waiting:
                status_parts.append(f"API限流: {remaining}/{total_wait}s")

        print(f"\r[{self.desc}] {self.current}/{self.total} ({percent:.1f}%) "
              f"{', '.join(status_parts)}", end="", flush=True)

    def finish(self):
        """完成进度"""
        elapsed = time.time() - self.start_time
        print(f"\r[{self.desc}] 完成! 耗时: {elapsed:.1f}s")
