"""
趋势雷达选股系统 - 优化后的缓存管理模块
支持内存缓存、LRU策略、压缩存储、线程安全
"""
import os
import pickle
import gzip
import hashlib
import threading
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from collections import OrderedDict
from functools import wraps

from core.logger import get_logger


class LRUCache:
    """LRU (最近最少使用) 缓存"""

    def __init__(self, capacity: int = 100):
        """
        初始化LRU缓存

        参数:
            capacity: 最大缓存条目数
        """
        self.capacity = capacity
        self.cache: OrderedDict = OrderedDict()
        self.lock = threading.Lock()

    def get(self, key: str) -> Optional[Any]:
        """
        获取缓存值

        参数:
            key: 缓存键

        返回:
            缓存值或None
        """
        with self.lock:
            if key in self.cache:
                # 移动到末尾(最近使用)
                self.cache.move_to_end(key)
                return self.cache[key]
            return None

    def put(self, key: str, value: Any):
        """
        存入缓存

        参数:
            key: 缓存键
            value: 缓存值
        """
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
            self.cache[key] = value

            # 超过容量,移除最久未使用的条目
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

    def keys(self) -> list:
        """获取所有键"""
        with self.lock:
            return list(self.cache.keys())


class CacheManager:
    """增强的缓存管理器 - 支持内存缓存、压缩存储、LRU策略"""

    def __init__(self, cache_dir: str = "cache", 
                 memory_cache_size: int = 100,
                 enable_compression: bool = True):
        """
        初始化缓存管理器

        参数:
            cache_dir: 缓存目录路径
            memory_cache_size: 内存缓存大小(LRU)
            enable_compression: 是否启用gzip压缩
        """
        self.cache_dir = Path(cache_dir)
        self.memory_cache = LRUCache(capacity=memory_cache_size)
        self.enable_compression = enable_compression
        self.logger = get_logger()
        self._ensure_cache_dirs()
        self.cache_stats = {
            'memory_hits': 0,
            'disk_hits': 0,
            'misses': 0,
            'writes': 0
        }

    def _ensure_cache_dirs(self):
        """确保所有缓存目录存在"""
        subdirs = [
            "trade_cal",
            "stock_basic",
            "daily",
            "index",
            "indicators",
            "backtest"
        ]
        for subdir in subdirs:
            (self.cache_dir / subdir).mkdir(parents=True, exist_ok=True)

    def _generate_key(self, *args) -> str:
        """
        生成缓存键

        参数:
            *args: 用于生成键的参数

        返回:
            MD5哈希键
        """
        key_str = "|".join(str(arg) for arg in args)
        return hashlib.md5(key_str.encode()).hexdigest()

    def _get_cache_path(self, cache_type: str, key: str) -> Path:
        """
        获取缓存文件路径

        参数:
            cache_type: 缓存类型
            key: 缓存键

        返回:
            缓存文件路径
        """
        filename = f"{key}.pkl{'z' if self.enable_compression else ''}"
        return self.cache_dir / cache_type / filename

    def _serialize(self, data: Any) -> bytes:
        """
        序列化数据

        参数:
            data: 要序列化的数据

        返回:
            序列化后的字节
        """
        pickled = pickle.dumps(data)
        if self.enable_compression:
            pickled = gzip.compress(pickled)
        return pickled

    def _deserialize(self, data: bytes) -> Any:
        """
        反序列化数据

        参数:
            data: 序列化的数据

        返回:
            反序列化后的对象
        """
        if self.enable_compression:
            data = gzip.decompress(data)
        return pickle.loads(data)

    def get(self, cache_type: str, *args, ttl_days: int = 7) -> Optional[Any]:
        """
        获取缓存数据(优先从内存缓存)

        参数:
            cache_type: 缓存类型
            *args: 缓存键参数
            ttl_days: 有效天数

        返回:
            缓存数据或None
        """
        key = self._generate_key(*args)

        # 1. 尝试从内存缓存获取
        cached_data = self.memory_cache.get(key)
        if cached_data is not None:
            data, timestamp = cached_data
            if (datetime.now() - timestamp).days < ttl_days:
                self.cache_stats['memory_hits'] += 1
                self.logger.debug(f"内存缓存命中: {cache_type}/{key[:8]}...")
                return data

        # 2. 尝试从磁盘缓存获取
        cache_path = self._get_cache_path(cache_type, key)
        if cache_path.exists():
            file_time = datetime.fromtimestamp(cache_path.stat().st_mtime)
            expired_time = datetime.now() - timedelta(days=ttl_days)

            if file_time >= expired_time:
                try:
                    with open(cache_path, 'rb') as f:
                        data = self._deserialize(f.read())
                    # 存入内存缓存
                    self.memory_cache.put(key, (data, datetime.now()))
                    self.cache_stats['disk_hits'] += 1
                    self.logger.debug(f"磁盘缓存命中: {cache_type}/{key[:8]}...")
                    return data
                except Exception as e:
                    self.logger.warning(f"读取磁盘缓存失败: {cache_path}, {e}")
            else:
                # 删除过期缓存
                cache_path.unlink()
                self.logger.debug(f"删除过期缓存: {cache_path}")

        self.cache_stats['misses'] += 1
        return None

    def put(self, cache_type: str, data: Any, *args):
        """
        存储缓存数据(同时写入内存和磁盘)

        参数:
            cache_type: 缓存类型
            data: 要缓存的数据
            *args: 缓存键参数
        """
        key = self._generate_key(*args)

        # 1. 存入内存缓存
        self.memory_cache.put(key, (data, datetime.now()))

        # 2. 写入磁盘缓存
        cache_path = self._get_cache_path(cache_type, key)
        try:
            # 确保目录存在
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            
            serialized = self._serialize(data)
            with open(cache_path, 'wb') as f:
                f.write(serialized)
            self.cache_stats['writes'] += 1
            self.logger.debug(f"写入缓存: {cache_type}/{key[:8]}...")
        except Exception as e:
            self.logger.error(f"写入缓存失败: {cache_path}, {e}")

    def get_csv(self, path: str, ttl_days: int = 7) -> Optional[pd.DataFrame]:
        """
        读取CSV缓存(兼容旧接口)

        参数:
            path: CSV文件路径
            ttl_days: 有效天数

        返回:
            DataFrame或None
        """
        # 检查文件是否存在
        if not os.path.exists(path):
            return None

        # 检查是否过期
        file_time = datetime.fromtimestamp(os.path.getmtime(path))
        expired_time = datetime.now() - timedelta(days=ttl_days)

        if file_time < expired_time:
            return None

        try:
            return pd.read_csv(path)
        except Exception as e:
            self.logger.warning(f"读取CSV缓存失败: {path}, {e}")
            return None

    def save_csv(self, df: pd.DataFrame, path: str):
        """
        保存DataFrame到CSV(兼容旧接口)

        参数:
            df: 要保存的DataFrame
            path: 保存路径
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False, encoding='utf-8-sig')

    def is_cache_expired(self, path: str, ttl_days: int) -> bool:
        """
        检查缓存是否过期(兼容旧接口)

        参数:
            path: 缓存文件路径
            ttl_days: 有效天数

        返回:
            True表示过期或不存在，False表示有效
        """
        if not os.path.exists(path):
            return True

        file_time = datetime.fromtimestamp(os.path.getmtime(path))
        expired_time = datetime.now() - timedelta(days=ttl_days)

        return file_time < expired_time

    def cache_path_trade_cal(self) -> str:
        """获取交易日历缓存路径(兼容旧接口)"""
        return str(self.cache_dir / "trade_cal" / "trade_cal.csv")

    def cache_path_stock_basic(self) -> str:
        """获取股票基础信息缓存路径(兼容旧接口)"""
        return str(self.cache_dir / "stock_basic" / "stock_basic.csv")

    def cache_path_daily(self, trade_date: str) -> str:
        """
        获取日线数据缓存路径(兼容旧接口)

        参数:
            trade_date: 交易日（YYYYMMDD格式）

        返回:
            缓存文件路径
        """
        filename = f"{trade_date}.csv"
        return str(self.cache_dir / "daily" / filename)

    def cache_path_index(self, trade_date: str, ts_code: str) -> str:
        """
        获取指数数据缓存路径(兼容旧接口)

        参数:
            trade_date: 交易日（YYYYMMDD格式）
            ts_code: 指数代码

        返回:
            缓存文件路径
        """
        filename = f"{trade_date}_{ts_code.replace('.', '_')}.csv"
        return str(self.cache_dir / "index" / filename)

    def clear(self, cache_type: str = None):
        """
        清理缓存

        参数:
            cache_type: 缓存类型，如果为None则清理全部缓存
        """
        # 清空内存缓存
        self.memory_cache.clear()

        # 清理磁盘缓存
        if cache_type:
            target_dir = self.cache_dir / cache_type
            if target_dir.exists():
                for file in target_dir.glob("*"):
                    if file.is_file():
                        file.unlink()
                self.logger.info(f"已清理 {cache_type} 缓存")
        else:
            for subdir in self.cache_dir.iterdir():
                if subdir.is_dir():
                    for file in subdir.glob("*"):
                        if file.is_file():
                            file.unlink()
            self.logger.info("已清理全部缓存")

        # 重置统计
        self.cache_stats = {
            'memory_hits': 0,
            'disk_hits': 0,
            'misses': 0,
            'writes': 0
        }

    def get_stats(self) -> Dict[str, int]:
        """
        获取缓存统计信息

        返回:
            统计字典
        """
        total_requests = (self.cache_stats['memory_hits'] + 
                       self.cache_stats['disk_hits'] + 
                       self.cache_stats['misses'])
        
        hit_rate = 0
        if total_requests > 0:
            hit_rate = ((self.cache_stats['memory_hits'] + 
                       self.cache_stats['disk_hits']) / total_requests * 100)

        return {
            'memory_hits': self.cache_stats['memory_hits'],
            'disk_hits': self.cache_stats['disk_hits'],
            'misses': self.cache_stats['misses'],
            'writes': self.cache_stats['writes'],
            'memory_cache_size': self.memory_cache.size(),
            'hit_rate': f"{hit_rate:.2f}%"
        }

    def print_stats(self):
        """打印缓存统计信息"""
        stats = self.get_stats()
        self.logger.info("=" * 50)
        self.logger.info("缓存统计信息")
        self.logger.info("=" * 50)
        for key, value in stats.items():
            self.logger.info(f"{key}: {value}")
        self.logger.info("=" * 50)


def cached(ttl_days: int = 7, cache_type: str = 'default'):
    """
    缓存装饰器

    参数:
        ttl_days: 缓存有效期(天)
        cache_type: 缓存类型

    返回:
        装饰器函数

    使用示例:
        @cached(ttl_days=1, cache_type='trade_cal')
        def get_trade_cal():
            # 获取交易日历
            ...
    """
    def decorator(func):
        cache_manager = CacheManager()

        @wraps(func)
        def wrapper(*args, **kwargs):
            # 生成缓存键
            key = (func.__name__, args, tuple(sorted(kwargs.items())))

            # 尝试从缓存获取
            result = cache_manager.get(cache_type, *key, ttl_days=ttl_days)
            if result is not None:
                return result

            # 调用函数并缓存结果
            result = func(*args, **kwargs)
            cache_manager.put(cache_type, result, *key)

            return result

        return wrapper
    return decorator
