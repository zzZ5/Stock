"""
配置管理模块
支持从 YAML/JSON 文件加载配置
"""

from config.config_loader import (
    ConfigLoader,
    ConfigValidator,
    ConfigLoadError,
    load_config,
    generate_config_template,
)


class Config:
    """全局配置管理类"""
    
    _instance = None
    _config = None
    _config_file = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def load(cls, config_file: str = None, validate: bool = True) -> None:
        """
        加载配置文件
        
        Args:
            config_file: 配置文件路径，如果为 None 则使用默认配置
            validate: 是否验证配置项
        """
        cls._config_file = config_file
        cls._config = load_config(config_file, validate)
    
    @classmethod
    def reload(cls, validate: bool = True) -> None:
        """
        重新加载配置文件
        """
        if cls._config_file is None:
            cls._config = load_config(None, validate)
        else:
            cls._config = load_config(cls._config_file, validate)
    
    @classmethod
    def get(cls, key: str, default=None):
        """
        获取配置项
        
        Args:
            key: 配置键名
            default: 默认值（如果配置项不存在）
        
        Returns:
            配置值
        """
        if cls._config is None:
            cls.load()
        
        return cls._config.get(key, default)
    
    @classmethod
    def get_all(cls) -> dict:
        """
        获取所有配置项
        
        Returns:
            配置字典
        """
        if cls._config is None:
            cls.load()
        
        return cls._config.copy()
    
    @classmethod
    def set(cls, key: str, value) -> None:
        """
        设置配置项（仅内存生效，不保存到文件）
        
        Args:
            key: 配置键名
            value: 配置值
        """
        if cls._config is None:
            cls.load()
        
        cls._config[key] = value
    
    @classmethod
    def is_loaded(cls) -> bool:
        """
        检查配置是否已加载
        
        Returns:
            是否已加载
        """
        return cls._config is not None


# 创建全局配置实例
config = Config()


# =========================
# 向后兼容：导出所有配置常量
# =========================

def _init_default_config():
    """初始化默认配置"""
    global _config_initialized
    if not _config_initialized:
        Config.load()
        _config_initialized = True


_config_initialized = False

# 选股策略参数
def get_TOP_N(): return Config.get('TOP_N', 20)
def get_BREAKOUT_N(): return Config.get('BREAKOUT_N', 60)
def get_MULTI_TIMEFRAME_MODE(): return Config.get('MULTI_TIMEFRAME_MODE', True)
def get_WEEKLY_BREAKOUT_N(): return Config.get('WEEKLY_BREAKOUT_N', 12)
def get_MONTHLY_BREAKOUT_N(): return Config.get('MONTHLY_BREAKOUT_N', 6)
def get_MA_FAST(): return Config.get('MA_FAST', 20)
def get_MA_SLOW(): return Config.get('MA_SLOW', 60)
def get_VOL_LOOKBACK(): return Config.get('VOL_LOOKBACK', 20)
def get_VOL_CONFIRM_MULT(): return Config.get('VOL_CONFIRM_MULT', 1.2)
def get_RSI_MAX(): return Config.get('RSI_MAX', 80)
def get_MAX_LOSS_PCT(): return Config.get('MAX_LOSS_PCT', 0.10)
def get_ATR_N(): return Config.get('ATR_N', 14)
def get_ATR_MULT(): return Config.get('ATR_MULT', 2.5)
def get_MIN_LIST_DAYS(): return Config.get('MIN_LIST_DAYS', 120)
def get_MIN_PRICE(): return Config.get('MIN_PRICE', 2.0)
def get_MIN_AVG_AMOUNT_20D(): return Config.get('MIN_AVG_AMOUNT_20D', 30000)
def get_EXCLUDE_ONE_WORD_LIMITUP(): return Config.get('EXCLUDE_ONE_WORD_LIMITUP', True)

# 缓存与限流参数
def get_CACHE_DIR(): return Config.get('CACHE_DIR', './tushare_cache')
def get_SLEEP_PER_CALL(): return Config.get('SLEEP_PER_CALL', 0.25)
def get_MAX_CALLS_PER_MINUTE(): return Config.get('MAX_CALLS_PER_MINUTE', 200)
def get_STOCK_BASIC_TTL_DAYS(): return Config.get('STOCK_BASIC_TTL_DAYS', 7)
def get_TRADE_CAL_TTL_DAYS(): return Config.get('TRADE_CAL_TTL_DAYS', 30)

# 报告输出参数
def get_SAVE_REPORT(): return Config.get('SAVE_REPORT', True)
def get_REPORT_DIR(): return Config.get('REPORT_DIR', './reports')

# 回测参数
def get_DEFAULT_HOLDING_DAYS(): return Config.get('DEFAULT_HOLDING_DAYS', 10)

# 指数参数
def get_INDEX_CODE(): return Config.get('INDEX_CODE', '000300.SH')

# 日志配置
def get_LOG_LEVEL(): return Config.get('LOG_LEVEL', 'INFO')
def get_LOG_DIR(): return Config.get('LOG_DIR', './logs')
def get_LOG_CONSOLE_OUTPUT(): return Config.get('LOG_CONSOLE_OUTPUT', True)
def get_LOG_FILE_OUTPUT(): return Config.get('LOG_FILE_OUTPUT', True)
def get_LOG_MAX_FILE_SIZE(): return Config.get('LOG_MAX_FILE_SIZE', 10 * 1024 * 1024)
def get_LOG_BACKUP_COUNT(): return Config.get('LOG_BACKUP_COUNT', 5)


# 导出函数
__all__ = [
    'Config',
    'config',
    'ConfigLoader',
    'ConfigValidator',
    'ConfigLoadError',
    'load_config',
    'generate_config_template',
    # Getter 函数（向后兼容）
    'get_TOP_N',
    'get_BREAKOUT_N',
    'get_MULTI_TIMEFRAME_MODE',
    'get_WEEKLY_BREAKOUT_N',
    'get_MONTHLY_BREAKOUT_N',
    'get_MA_FAST',
    'get_MA_SLOW',
    'get_VOL_LOOKBACK',
    'get_VOL_CONFIRM_MULT',
    'get_RSI_MAX',
    'get_MAX_LOSS_PCT',
    'get_ATR_N',
    'get_ATR_MULT',
    'get_MIN_LIST_DAYS',
    'get_MIN_PRICE',
    'get_MIN_AVG_AMOUNT_20D',
    'get_EXCLUDE_ONE_WORD_LIMITUP',
    'get_CACHE_DIR',
    'get_SLEEP_PER_CALL',
    'get_MAX_CALLS_PER_MINUTE',
    'get_STOCK_BASIC_TTL_DAYS',
    'get_TRADE_CAL_TTL_DAYS',
    'get_SAVE_REPORT',
    'get_REPORT_DIR',
    'get_DEFAULT_HOLDING_DAYS',
    'get_INDEX_CODE',
    'get_LOG_LEVEL',
    'get_LOG_DIR',
    'get_LOG_CONSOLE_OUTPUT',
    'get_LOG_FILE_OUTPUT',
    'get_LOG_MAX_FILE_SIZE',
    'get_LOG_BACKUP_COUNT',
]
