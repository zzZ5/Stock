"""
趋势雷达选股系统 - 缓存管理模块
负责数据的缓存、读取和过期检查
"""
import os
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path


class CacheManager:
    """缓存管理器"""

    def __init__(self, cache_dir: str = "cache"):
        """
        初始化缓存管理器

        参数:
            cache_dir: 缓存目录路径
        """
        self.cache_dir = Path(cache_dir)
        self._ensure_cache_dirs()

    def _ensure_cache_dirs(self):
        """确保所有缓存目录存在"""
        subdirs = [
            "trade_cal",
            "stock_basic",
            "daily",
            "index"
        ]
        for subdir in subdirs:
            (self.cache_dir / subdir).mkdir(parents=True, exist_ok=True)

    def cache_path_trade_cal(self) -> str:
        """获取交易日历缓存路径"""
        return str(self.cache_dir / "trade_cal" / "trade_cal.csv")

    def cache_path_stock_basic(self) -> str:
        """获取股票基础信息缓存路径"""
        return str(self.cache_dir / "stock_basic" / "stock_basic.csv")

    def cache_path_daily(self, trade_date: str) -> str:
        """
        获取日线数据缓存路径

        参数:
            trade_date: 交易日（YYYYMMDD格式）

        返回:
            缓存文件路径
        """
        filename = f"{trade_date}.csv"
        return str(self.cache_dir / "daily" / filename)

    def cache_path_index(self, trade_date: str, ts_code: str) -> str:
        """
        获取指数数据缓存路径

        参数:
            trade_date: 交易日（YYYYMMDD格式）
            ts_code: 指数代码

        返回:
            缓存文件路径
        """
        filename = f"{trade_date}_{ts_code.replace('.', '_')}.csv"
        return str(self.cache_dir / "index" / filename)

    def is_cache_expired(self, path: str, ttl_days: int) -> bool:
        """
        检查缓存是否过期

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

    def save_csv(self, df: pd.DataFrame, path: str):
        """
        保存DataFrame到CSV文件

        参数:
            df: 要保存的DataFrame
            path: 保存路径
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False, encoding='utf-8-sig')

    def read_csv_if_exists(self, path: str) -> pd.DataFrame | None:
        """
        读取CSV文件，如果文件不存在返回None

        参数:
            path: 文件路径

        返回:
            DataFrame或None
        """
        if not os.path.exists(path):
            return None

        try:
            return pd.read_csv(path)
        except Exception as e:
            print(f"[警告] 读取缓存文件失败: {path}, 错误: {e}")
            return None

    def clear_cache(self, cache_type: str = None):
        """
        清理缓存

        参数:
            cache_type: 缓存类型（trade_cal, stock_basic, daily, index），
                       如果为None则清理全部缓存
        """
        if cache_type:
            target_dir = self.cache_dir / cache_type
            if target_dir.exists():
                for file in target_dir.glob("*.csv"):
                    file.unlink()
                print(f"[缓存] 已清理 {cache_type} 缓存")
        else:
            for subdir in self.cache_dir.iterdir():
                if subdir.is_dir():
                    for file in subdir.glob("*.csv"):
                        file.unlink()
            print(f"[缓存] 已清理全部缓存")
