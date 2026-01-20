"""
趋势雷达选股系统 - 日志模块
提供统一的日志配置和管理
"""
import logging
import os
from logging.handlers import RotatingFileHandler
from datetime import datetime
from typing import Optional


class Logger:
    """日志管理器"""

    _loggers = {}
    _initialized = False

    @classmethod
    def setup_logging(cls,
                    log_level: str = "INFO",
                    log_dir: str = "./logs",
                    console_output: bool = True,
                    file_output: bool = True,
                    max_file_size: int = 10 * 1024 * 1024,  # 10MB
                    backup_count: int = 5):
        """
        初始化全局日志配置

        参数:
            log_level: 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_dir: 日志文件目录
            console_output: 是否输出到控制台
            file_output: 是否输出到文件
            max_file_size: 单个日志文件最大大小（字节）
            backup_count: 保留的日志文件备份数量
        """
        if cls._initialized:
            return

        # 创建日志目录
        if file_output:
            os.makedirs(log_dir, exist_ok=True)

        # 设置根日志级别
        level = getattr(logging, log_level.upper(), logging.INFO)
        logging.basicConfig(level=level, force=True)

        # 移除默认处理器
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        # 日志格式
        formatter = logging.Formatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # 控制台输出
        if console_output:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(level)
            console_handler.setFormatter(formatter)
            root_logger.addHandler(console_handler)

        # 文件输出
        if file_output:
            # 主日志文件（所有级别）
            main_log_file = os.path.join(log_dir, f"stock_system_{datetime.now().strftime('%Y%m%d')}.log")
            file_handler = RotatingFileHandler(
                main_log_file,
                maxBytes=max_file_size,
                backupCount=backup_count,
                encoding='utf-8'
            )
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)

            # 错误日志文件（只记录ERROR及以上）
            error_log_file = os.path.join(log_dir, f"stock_system_error_{datetime.now().strftime('%Y%m%d')}.log")
            error_handler = RotatingFileHandler(
                error_log_file,
                maxBytes=max_file_size,
                backupCount=backup_count,
                encoding='utf-8'
            )
            error_handler.setLevel(logging.ERROR)
            error_handler.setFormatter(formatter)
            root_logger.addHandler(error_handler)

        cls._initialized = True
        logging.info("日志系统初始化完成")

    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """
        获取指定名称的logger实例

        参数:
            name: logger名称（通常使用 __name__）

        返回:
            Logger实例
        """
        if name not in cls._loggers:
            cls._loggers[name] = logging.getLogger(name)
        return cls._loggers[name]


# 便捷函数
def get_logger(name: str = __name__) -> logging.Logger:
    """
    获取logger实例的便捷函数

    参数:
        name: logger名称

    返回:
        Logger实例
    """
    return Logger.get_logger(name)


# 各模块专用logger
def get_datafetcher_logger() -> logging.Logger:
    """获取DataFetcher专用logger"""
    return Logger.get_logger('datafetcher')


def get_strategy_logger() -> logging.Logger:
    """获取Strategy专用logger"""
    return Logger.get_logger('strategy')


def get_backtest_logger() -> logging.Logger:
    """获取Backtest专用logger"""
    return Logger.get_logger('backtest')


def get_optimizer_logger() -> logging.Logger:
    """获取Optimizer专用logger"""
    return Logger.get_logger('optimizer')


def get_indicator_logger() -> logging.Logger:
    """获取Indicator专用logger"""
    return Logger.get_logger('indicator')


def get_reporter_logger() -> logging.Logger:
    """获取Reporter专用logger"""
    return Logger.get_logger('reporter')


if __name__ == "__main__":
    # 测试日志系统
    Logger.setup_logging(log_level="DEBUG")

    # 测试不同级别的logger
    logger = get_logger("test")
    logger.debug("这是DEBUG信息")
    logger.info("这是INFO信息")
    logger.warning("这是WARNING信息")
    logger.error("这是ERROR信息")
    logger.critical("这是CRITICAL信息")

    # 测试模块专用logger
    df_logger = get_datafetcher_logger()
    df_logger.info("DataFetcher测试")

    st_logger = get_strategy_logger()
    st_logger.info("Strategy测试")
