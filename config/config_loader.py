"""
配置文件加载器
支持 YAML 和 JSON 格式的配置文件
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


class ConfigLoadError(Exception):
    """配置加载异常"""
    pass


class ConfigValidator:
    """配置验证器"""
    
    # 选股策略参数验证规则
    STRATEGY_VALIDATORS = {
        'TOP_N': lambda x: isinstance(x, int) and x > 0,
        'BREAKOUT_N': lambda x: isinstance(x, int) and x > 0,
        'MULTI_TIMEFRAME_MODE': lambda x: isinstance(x, bool),
        'WEEKLY_BREAKOUT_N': lambda x: isinstance(x, int) and x > 0,
        'MONTHLY_BREAKOUT_N': lambda x: isinstance(x, int) and x > 0,
        'MA_FAST': lambda x: isinstance(x, int) and x > 0,
        'MA_SLOW': lambda x: isinstance(x, int) and x > 0,
        'VOL_LOOKBACK': lambda x: isinstance(x, int) and x > 0,
        'VOL_CONFIRM_MULT': lambda x: isinstance(x, (int, float)) and x > 0,
        'RSI_MAX': lambda x: isinstance(x, (int, float)) and 0 < x <= 100,
        'MAX_LOSS_PCT': lambda x: isinstance(x, (int, float)) and 0 < x <= 1,
        'ATR_N': lambda x: isinstance(x, int) and x > 0,
        'ATR_MULT': lambda x: isinstance(x, (int, float)) and x > 0,
        'MIN_LIST_DAYS': lambda x: isinstance(x, int) and x > 0,
        'MIN_PRICE': lambda x: isinstance(x, (int, float)) and x > 0,
        'MIN_AVG_AMOUNT_20D': lambda x: isinstance(x, (int, float)) and x >= 0,
        'EXCLUDE_ONE_WORD_LIMITUP': lambda x: isinstance(x, bool),
    }
    
    # 缓存参数验证规则
    CACHE_VALIDATORS = {
        'CACHE_DIR': lambda x: isinstance(x, str) and len(x) > 0,
        'SLEEP_PER_CALL': lambda x: isinstance(x, (int, float)) and x >= 0,
        'MAX_CALLS_PER_MINUTE': lambda x: isinstance(x, int) and x > 0,
        'STOCK_BASIC_TTL_DAYS': lambda x: isinstance(x, int) and x > 0,
        'TRADE_CAL_TTL_DAYS': lambda x: isinstance(x, int) and x > 0,
    }
    
    # 报告参数验证规则
    REPORT_VALIDATORS = {
        'SAVE_REPORT': lambda x: isinstance(x, bool),
        'REPORT_DIR': lambda x: isinstance(x, str) and len(x) > 0,
    }
    
    # 回测参数验证规则
    BACKTEST_VALIDATORS = {
        'DEFAULT_HOLDING_DAYS': lambda x: isinstance(x, int) and x > 0,
    }
    
    # 指数参数验证规则
    INDEX_VALIDATORS = {
        'INDEX_CODE': lambda x: isinstance(x, str) and len(x) > 0,
    }
    
    # 日志参数验证规则
    LOG_VALIDATORS = {
        'LOG_LEVEL': lambda x: isinstance(x, str) and x in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        'LOG_DIR': lambda x: isinstance(x, str) and len(x) > 0,
        'LOG_CONSOLE_OUTPUT': lambda x: isinstance(x, bool),
        'LOG_FILE_OUTPUT': lambda x: isinstance(x, bool),
        'LOG_MAX_FILE_SIZE': lambda x: isinstance(x, int) and x > 0,
        'LOG_BACKUP_COUNT': lambda x: isinstance(x, int) and x >= 0,
    }
    
    @classmethod
    def validate_all(cls, config: Dict[str, Any]) -> None:
        """验证所有配置项"""
        all_validators = {}
        all_validators.update(cls.STRATEGY_VALIDATORS)
        all_validators.update(cls.CACHE_VALIDATORS)
        all_validators.update(cls.REPORT_VALIDATORS)
        all_validators.update(cls.BACKTEST_VALIDATORS)
        all_validators.update(cls.INDEX_VALIDATORS)
        all_validators.update(cls.LOG_VALIDATORS)
        
        errors = []
        for key, validator in all_validators.items():
            if key in config:
                try:
                    if not validator(config[key]):
                        errors.append(f"配置项 '{key}' 的值 '{config[key]}' 无效")
                except Exception as e:
                    errors.append(f"配置项 '{key}' 验证失败: {str(e)}")
        
        if errors:
            raise ConfigLoadError("配置验证失败:\n" + "\n".join(errors))


class ConfigLoader:
    """配置文件加载器"""
    
    # 默认配置模板
    DEFAULT_CONFIG = {
        # 选股策略参数
        'TOP_N': 20,
        'BREAKOUT_N': 60,
        'MULTI_TIMEFRAME_MODE': True,
        'WEEKLY_BREAKOUT_N': 12,
        'MONTHLY_BREAKOUT_N': 6,
        'MA_FAST': 20,
        'MA_SLOW': 60,
        'VOL_LOOKBACK': 20,
        'VOL_CONFIRM_MULT': 1.2,
        'RSI_MAX': 80,
        'MAX_LOSS_PCT': 0.10,
        'ATR_N': 14,
        'ATR_MULT': 2.5,
        'MIN_LIST_DAYS': 120,
        'MIN_PRICE': 2.0,
        'MIN_AVG_AMOUNT_20D': 30000,
        'EXCLUDE_ONE_WORD_LIMITUP': True,
        
        # 缓存与限流参数
        'CACHE_DIR': './tushare_cache',
        'SLEEP_PER_CALL': 0.25,
        'MAX_CALLS_PER_MINUTE': 200,
        'STOCK_BASIC_TTL_DAYS': 7,
        'TRADE_CAL_TTL_DAYS': 30,
        
        # 报告输出参数
        'SAVE_REPORT': True,
        'REPORT_DIR': './reports',
        
        # 回测参数
        'DEFAULT_HOLDING_DAYS': 10,
        
        # 指数参数
        'INDEX_CODE': '000300.SH',
        
        # 日志配置
        'LOG_LEVEL': 'INFO',
        'LOG_DIR': './logs',
        'LOG_CONSOLE_OUTPUT': True,
        'LOG_FILE_OUTPUT': True,
        'LOG_MAX_FILE_SIZE': 10 * 1024 * 1024,
        'LOG_BACKUP_COUNT': 5,
    }
    
    @staticmethod
    def _load_yaml(file_path: Path) -> Dict[str, Any]:
        """加载 YAML 配置文件"""
        if not YAML_AVAILABLE:
            raise ConfigLoadError("未安装 PyYAML 库，请运行: pip install pyyaml")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            raise ConfigLoadError(f"YAML 文件解析失败: {str(e)}")
        except Exception as e:
            raise ConfigLoadError(f"读取 YAML 文件失败: {str(e)}")
    
    @staticmethod
    def _load_json(file_path: Path) -> Dict[str, Any]:
        """加载 JSON 配置文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            raise ConfigLoadError(f"JSON 文件解析失败: {str(e)}")
        except Exception as e:
            raise ConfigLoadError(f"读取 JSON 文件失败: {str(e)}")
    
    @staticmethod
    def load(file_path: Optional[str] = None, validate: bool = True) -> Dict[str, Any]:
        """
        加载配置文件
        
        Args:
            file_path: 配置文件路径（YAML 或 JSON），如果为 None 则使用默认配置
            validate: 是否验证配置项
        
        Returns:
            配置字典
        """
        if file_path is None:
            config = ConfigLoader.DEFAULT_CONFIG.copy()
            if validate:
                ConfigValidator.validate_all(config)
            return config
        
        path = Path(file_path)
        
        if not path.exists():
            raise ConfigLoadError(f"配置文件不存在: {file_path}")
        
        if not path.is_file():
            raise ConfigLoadError(f"配置路径不是文件: {file_path}")
        
        # 根据文件扩展名选择加载方式
        suffix = path.suffix.lower()
        
        if suffix in ['.yaml', '.yml']:
            loaded_config = ConfigLoader._load_yaml(path)
        elif suffix == '.json':
            loaded_config = ConfigLoader._load_json(path)
        else:
            raise ConfigLoadError(f"不支持的配置文件格式: {suffix}")
        
        # 合并默认配置和加载的配置
        config = ConfigLoader.DEFAULT_CONFIG.copy()
        config.update(loaded_config)
        
        if validate:
            ConfigValidator.validate_all(config)
        
        return config
    
    @staticmethod
    def save_template(file_path: str, format: str = 'yaml') -> None:
        """
        保存配置模板文件
        
        Args:
            file_path: 保存路径
            format: 文件格式 ('yaml' 或 'json')
        """
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if format.lower() == 'yaml':
            if not YAML_AVAILABLE:
                raise ConfigLoadError("未安装 PyYAML 库，请运行: pip install pyyaml")
            
            with open(path, 'w', encoding='utf-8') as f:
                yaml.safe_dump(ConfigLoader.DEFAULT_CONFIG, f, 
                             allow_unicode=True, 
                             sort_keys=False,
                             default_flow_style=False)
        
        elif format.lower() == 'json':
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(ConfigLoader.DEFAULT_CONFIG, f, 
                         ensure_ascii=False, 
                         indent=2)
        
        else:
            raise ConfigLoadError(f"不支持的格式: {format}")


def load_config(file_path: Optional[str] = None, validate: bool = True) -> Dict[str, Any]:
    """
    加载配置的便捷函数
    
    Args:
        file_path: 配置文件路径，如果为 None 则使用默认配置
        validate: 是否验证配置项
    
    Returns:
        配置字典
    """
    return ConfigLoader.load(file_path, validate)


def generate_config_template(file_path: str, format: str = 'yaml') -> None:
    """
    生成配置模板的便捷函数
    
    Args:
        file_path: 保存路径
        format: 文件格式 ('yaml' 或 'json')
    """
    ConfigLoader.save_template(file_path, format)
