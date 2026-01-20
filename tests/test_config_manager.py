"""
配置管理测试
"""

import json
import os
import pytest
import tempfile
from pathlib import Path

from app_config import Config, config


class TestConfig:
    """全局配置管理类测试"""
    
    def test_config_singleton(self):
        """测试 Config 单例模式"""
        config1 = Config()
        config2 = Config()
        assert config1 is config2
    
    def test_load_default_config(self):
        """测试加载默认配置"""
        Config._config = None  # 重置
        Config.load()
        
        assert Config.is_loaded()
        assert Config.get('TOP_N') == 20
        assert Config.get('BREAKOUT_N') == 60
    
    def test_load_json_file(self):
        """测试从 JSON 文件加载配置"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({'TOP_N': 30, 'MA_FAST': 30}, f)
            temp_path = f.name
        
        try:
            Config._config = None  # 重置
            Config.load(temp_path)
            
            assert Config.get('TOP_N') == 30
            assert Config.get('MA_FAST') == 30
        finally:
            os.unlink(temp_path)
    
    def test_load_yaml_file(self):
        """测试从 YAML 文件加载配置"""
        pytest.importorskip('yaml')
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write('TOP_N: 25\n')
            f.write('LOG_LEVEL: DEBUG\n')
            temp_path = f.name
        
        try:
            Config._config = None  # 重置
            Config.load(temp_path)
            
            assert Config.get('TOP_N') == 25
            assert Config.get('LOG_LEVEL') == 'DEBUG'
        finally:
            os.unlink(temp_path)
    
    def test_reload_config(self):
        """测试重新加载配置"""
        Config._config = None
        Config.load()
        
        original_top_n = Config.get('TOP_N')
        
        # 修改内存中的配置
        Config.set('TOP_N', 99)
        assert Config.get('TOP_N') == 99
        
        # 重新加载
        Config.reload()
        assert Config.get('TOP_N') == original_top_n
    
    def test_reload_from_file(self):
        """测试从文件重新加载配置"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({'TOP_N': 40}, f)
            temp_path = f.name
        
        try:
            Config._config = None
            Config.load(temp_path)
            assert Config.get('TOP_N') == 40
            
            # 修改文件
            with open(temp_path, 'w') as f:
                json.dump({'TOP_N': 50}, f)
            
            Config.reload()
            assert Config.get('TOP_N') == 50
        finally:
            os.unlink(temp_path)
    
    def test_get_with_default(self):
        """测试获取配置（带默认值）"""
        Config._config = None
        Config.load()
        
        # 存在的配置项
        assert Config.get('TOP_N') == 20
        
        # 不存在的配置项（使用默认值）
        assert Config.get('NONEXISTENT_KEY', 100) == 100
        assert Config.get('ANOTHER_KEY', 'default') == 'default'
    
    def test_get_all(self):
        """测试获取所有配置"""
        Config._config = None
        Config.load()
        
        all_config = Config.get_all()
        
        assert isinstance(all_config, dict)
        assert 'TOP_N' in all_config
        assert 'LOG_LEVEL' in all_config
    
    def test_set_config(self):
        """测试设置配置项"""
        Config._config = None
        Config.load()
        
        # 设置新值
        Config.set('TOP_N', 50)
        assert Config.get('TOP_N') == 50
        
        # 设置新键
        Config.set('NEW_KEY', 'test_value')
        assert Config.get('NEW_KEY') == 'test_value'
    
    def test_is_loaded(self):
        """测试配置是否已加载"""
        Config._config = None
        assert not Config.is_loaded()
        
        Config.load()
        assert Config.is_loaded()
    
    def test_config_instance(self):
        """测试全局配置实例"""
        Config._config = None
        config.load()
        
        assert Config.is_loaded()
        assert Config.get('TOP_N') == 20


class TestConfigGetters:
    """配置 getter 函数测试（向后兼容）"""
    
    def test_strategy_getters(self):
        """测试策略参数 getter 函数"""
        Config._config = None
        Config.load()
        
        assert Config.get('TOP_N') == 20
        assert Config.get('BREAKOUT_N') == 60
        assert Config.get('MA_FAST') == 20
        assert Config.get('MA_SLOW') == 60
        assert Config.get('VOL_LOOKBACK') == 20
        assert Config.get('VOL_CONFIRM_MULT') == 1.2
        assert Config.get('RSI_MAX') == 80
        assert Config.get('MAX_LOSS_PCT') == 0.10
        assert Config.get('ATR_N') == 14
        assert Config.get('ATR_MULT') == 2.5
        assert Config.get('MIN_LIST_DAYS') == 120
        assert Config.get('MIN_PRICE') == 2.0
        assert Config.get('MIN_AVG_AMOUNT_20D') == 30000
        assert Config.get('EXCLUDE_ONE_WORD_LIMITUP') is True
    
    def test_cache_getters(self):
        """测试缓存参数 getter 函数"""
        Config._config = None
        Config.load()
        
        assert Config.get('CACHE_DIR') == './tushare_cache'
        assert Config.get('SLEEP_PER_CALL') == 0.25
        assert Config.get('MAX_CALLS_PER_MINUTE') == 200
        assert Config.get('STOCK_BASIC_TTL_DAYS') == 7
        assert Config.get('TRADE_CAL_TTL_DAYS') == 30
    
    def test_report_getters(self):
        """测试报告参数 getter 函数"""
        Config._config = None
        Config.load()
        
        assert Config.get('SAVE_REPORT') is True
        assert Config.get('REPORT_DIR') == './reports'
    
    def test_backtest_getters(self):
        """测试回测参数 getter 函数"""
        Config._config = None
        Config.load()
        
        assert Config.get('DEFAULT_HOLDING_DAYS') == 10
    
    def test_index_getters(self):
        """测试指数参数 getter 函数"""
        Config._config = None
        Config.load()
        
        assert Config.get('INDEX_CODE') == '000300.SH'
    
    def test_log_getters(self):
        """测试日志参数 getter 函数"""
        Config._config = None
        Config.load()
        
        assert Config.get('LOG_LEVEL') == 'INFO'
        assert Config.get('LOG_DIR') == './logs'
        assert Config.get('LOG_CONSOLE_OUTPUT') is True
        assert Config.get('LOG_FILE_OUTPUT') is True
        assert Config.get('LOG_MAX_FILE_SIZE') == 10 * 1024 * 1024
        assert Config.get('LOG_BACKUP_COUNT') == 5


class TestConfigCustomization:
    """配置自定义测试"""
    
    def test_custom_strategy_params(self):
        """测试自定义策略参数"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({
                'TOP_N': 50,
                'BREAKOUT_N': 30,
                'RSI_MAX': 75,
            }, f)
            temp_path = f.name
        
        try:
            Config._config = None
            Config.load(temp_path)
            
            assert Config.get('TOP_N') == 50
            assert Config.get('BREAKOUT_N') == 30
            assert Config.get('RSI_MAX') == 75
            
            # 未自定义的参数使用默认值
            assert Config.get('MA_FAST') == 20
        finally:
            os.unlink(temp_path)
    
    def test_custom_cache_params(self):
        """测试自定义缓存参数"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({
                'CACHE_DIR': './custom_cache',
                'MAX_CALLS_PER_MINUTE': 300,
            }, f)
            temp_path = f.name
        
        try:
            Config._config = None
            Config.load(temp_path)
            
            assert Config.get('CACHE_DIR') == './custom_cache'
            assert Config.get('MAX_CALLS_PER_MINUTE') == 300
        finally:
            os.unlink(temp_path)
    
    def test_custom_log_params(self):
        """测试自定义日志参数"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({
                'LOG_LEVEL': 'DEBUG',
                'LOG_DIR': './custom_logs',
                'LOG_CONSOLE_OUTPUT': False,
            }, f)
            temp_path = f.name
        
        try:
            Config._config = None
            Config.load(temp_path)
            
            assert Config.get('LOG_LEVEL') == 'DEBUG'
            assert Config.get('LOG_DIR') == './custom_logs'
            assert Config.get('LOG_CONSOLE_OUTPUT') is False
        finally:
            os.unlink(temp_path)
