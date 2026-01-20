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
    """API调用限流器"""

    def __init__(self, max_calls_per_minute: int = 200):
        """
        初始化限流器

        参数:
            max_calls_per_minute: 每分钟最大调用次数
        """
        self.max_calls = max_calls_per_minute
        self.calls = []
        self.lock = None

    def wait_if_needed(self):
        """如果需要，等待以遵守速率限制"""
        now = time.time()

        # 清理超过1分钟的调用记录
        self.calls = [t for t in self.calls if now - t < 60]

        # 如果达到限制，等待
        if len(self.calls) >= self.max_calls:
            sleep_time = 60 - (now - self.calls[0]) + 0.1
            if sleep_time > 0:
                time.sleep(sleep_time)

        # 记录这次调用
        self.calls.append(time.time())

    def sleep(self, seconds: float):
        """额外等待"""
        time.sleep(seconds)


class ProgressTracker:
    """进度追踪器"""

    def __init__(self, total: int, desc: str = "Progress"):
        """
        初始化进度追踪器

        参数:
            total: 总数量
            desc: 描述信息
        """
        self.total = total
        self.current = 0
        self.desc = desc
        self.start_time = time.time()

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

        print(f"\r[{self.desc}] {self.current}/{self.total} ({percent:.1f}%) "
              f"ETA: {eta_str}", end="", flush=True)

    def finish(self):
        """完成进度"""
        elapsed = time.time() - self.start_time
        print(f"\r[{self.desc}] 完成! 耗时: {elapsed:.1f}s")
