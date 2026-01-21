"""
趋势雷达选股系统 - 缓存管理模块
整合原版和优化版功能，支持LRU缓存、压缩存储、线程安全
"""
import os
import pickle
import gzip
import hashlib
import threading
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any
from collections import OrderedDict
from functools import wraps

from core.logger import get_logger

logger = get_logger(__name__)


class LRUCache:
    """LRU (最近最少使用) 缓存"""

    def __init__(self, capacity: int = 100):
        """初始化LRU缓存"""
        self.capacity = capacity
        self.cache: OrderedDict = OrderedDict()
        self.lock = threading.Lock()

    def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
                return self.cache[key]
            return None

    def put(self, key: str, value: Any):
        """存入缓存"""
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
            self.cache[key] = value
            if len(self.cache) > self.capacity:
                self.cache.popitem(last=False)

    def clear(self):
        """清空缓存"""
        with self.lock:
            self.cache.clear()

    def size(self) -> int:
        """获取当前缓存大小"""
        with self.lock:
            return len(self.cache)


class CacheManager:
    """缓存管理器 - 支持内存缓存、LRU策略、压缩存储"""

    def __init__(self, cache_dir: str = "cache", 
                 memory_cache_size: int = 100,
                 enable_compression: bool = True):
        """
        初始化缓存管理器

        参数:
            cache_dir: 缓存目录路径
            memory_cache_size: 内存缓存大小
            enable_compression: 是否启用gzip压缩
        """
        self.cache_dir = Path(cache_dir)
        self.memory_cache = LRUCache(capacity=memory_cache_size)
        self.enable_compression = enable_compression
        self._ensure_cache_dirs()

    def _ensure_cache_dirs(self):
        """确保所有缓存目录存在"""
        subdirs = ["trade_cal", "stock_basic", "daily", "index", "weekly", "monthly"]
        for subdir in subdirs:
            (self.cache_dir / subdir).mkdir(parents=True, exist_ok=True)
        # 确保daily下的stocks子目录存在
        (self.cache_dir / "daily" / "stocks").mkdir(parents=True, exist_ok=True)

    def _get_cache_key(self, *args) -> str:
        """生成缓存键"""
        key_str = "|".join(str(arg) for arg in args)
        return hashlib.md5(key_str.encode()).hexdigest()

    def cache_path_trade_cal(self) -> str:
        """获取交易日历缓存路径"""
        return str(self.cache_dir / "trade_cal" / "trade_cal.csv")

    def cache_path_stock_basic(self) -> str:
        """获取股票基础信息缓存路径"""
        return str(self.cache_dir / "stock_basic" / "stock_basic.csv")

    def cache_path_daily(self, trade_date: str) -> str:
        """获取日线数据缓存路径"""
        filename = f"{trade_date}.csv"
        return str(self.cache_dir / "daily" / filename)

    def cache_path_index(self, trade_date: str, ts_code: str) -> str:
        """获取指数数据缓存路径（单日）"""
        filename = f"{trade_date}_{ts_code.replace('.', '_')}.csv"
        return str(self.cache_dir / "index" / filename)

    def cache_path_index_range(self, ts_code: str, start_date: str, end_date: str) -> str:
        """获取指数数据范围缓存路径"""
        filename = f"{ts_code.replace('.', '_')}_{start_date}_{end_date}.csv"
        return str(self.cache_dir / "index" / filename)

    def cache_path_weekly(self, ts_code: str, start_date: str, end_date: str) -> str:
        """获取周线数据缓存路径"""
        filename = f"{ts_code.replace('.', '_')}_{start_date}_{end_date}.csv"
        return str(self.cache_dir / "weekly" / filename)

    def cache_path_monthly(self, ts_code: str, start_date: str, end_date: str) -> str:
        """获取月线数据缓存路径"""
        filename = f"{ts_code.replace('.', '_')}_{start_date}_{end_date}.csv"
        return str(self.cache_dir / "monthly" / filename)

    def save_feather(self, df: pd.DataFrame, path: str):
        """保存DataFrame到Feather文件（比CSV快很多）"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_feather(path)

    def read_feather_if_exists(self, path: str) -> Optional[pd.DataFrame]:
        """读取Feather文件，如果文件不存在返回None"""
        if not os.path.exists(path):
            return None

        try:
            return pd.read_feather(path)
        except Exception as e:
            logger.warning(f"读取Feather缓存文件失败: {path}, 错误: {e}")
            return None

    def cache_path_stock_daily(self, ts_code: str, start_date: str, end_date: str) -> str:
        """获取单只股票日线数据缓存路径"""
        filename = f"{ts_code.replace('.', '_')}_{start_date}_{end_date}.csv"
        return str(self.cache_dir / "daily" / "stocks" / filename)

    def is_cache_expired(self, path: str, ttl_days: int) -> bool:
        """检查缓存是否过期"""
        if not os.path.exists(path):
            return True

        file_time = datetime.fromtimestamp(os.path.getmtime(path))
        expired_time = datetime.now() - timedelta(days=ttl_days)
        return file_time < expired_time

    def save_csv(self, df: pd.DataFrame, path: str):
        """保存DataFrame到CSV文件"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False, encoding='utf-8-sig')

    def read_csv_if_exists(self, path: str) -> Optional[pd.DataFrame]:
        """读取CSV文件，如果文件不存在返回None"""
        if not os.path.exists(path):
            return None

        try:
            return pd.read_csv(path)
        except Exception as e:
            logger.warning(f"读取缓存文件失败: {path}, 错误: {e}")
            return None

    def get_from_memory(self, key: str) -> Optional[Any]:
        """从内存缓存获取数据"""
        return self.memory_cache.get(key)

    def put_to_memory(self, key: str, value: Any):
        """将数据存入内存缓存"""
        self.memory_cache.put(key, value)

    def compress_data(self, data: Any) -> bytes:
        """压缩数据"""
        return gzip.compress(pickle.dumps(data))

    def decompress_data(self, compressed: bytes) -> Any:
        """解压数据"""
        return pickle.loads(gzip.decompress(compressed))

    def save_compressed(self, data: Any, path: str):
        """保存压缩数据"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        compressed = self.compress_data(data) if self.enable_compression else pickle.dumps(data)
        with open(path, 'wb') as f:
            f.write(compressed)

    def load_compressed(self, path: str) -> Optional[Any]:
        """加载压缩数据"""
        if not os.path.exists(path):
            return None

        try:
            with open(path, 'rb') as f:
                compressed = f.read()
            return self.decompress_data(compressed) if self.enable_compression else pickle.loads(compressed)
        except Exception as e:
            logger.warning(f"加载压缩数据失败: {path}, 错误: {e}")
            return None

    def clear_cache(self, cache_type: str = None):
        """清理缓存"""
        if cache_type:
            target_dir = self.cache_dir / cache_type
            if target_dir.exists():
                for file in target_dir.glob("*.csv"):
                    file.unlink()
                logger.info(f"已清理 {cache_type} 缓存")
        else:
            for subdir in self.cache_dir.iterdir():
                if subdir.is_dir():
                    for file in subdir.glob("*.csv"):
                        file.unlink()
            self.memory_cache.clear()
            logger.info("已清理全部缓存")
