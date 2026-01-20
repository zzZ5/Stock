"""
配置文件加载器测试
"""

import json
import os
import pytest
import tempfile
from pathlib import Path

from config.config_loader import (
    ConfigLoader,
    ConfigValidator,
    ConfigLoadError,
    load_config,
    generate_config_template,
)


class TestConfigValidator:
    """配置验证器测试"""
    
    def test_validate_valid_strategy_config(self):
        """测试验证有效的策略配置"""
        valid_config = {
            'TOP_N': 20,
            'BREAKOUT_N': 60,
            'MA_FAST': 20,
            'MA_SLOW': 60,
            'MAX_LOSS_PCT': 0.10,
        }
        # 不应该抛出异常
        ConfigValidator.validate_all(valid_config)
    
    def test_validate_invalid_top_n(self):
        """测试验证无效的 TOP_N"""
        invalid_config = {
            'TOP_N': -1,
        }
        with pytest.raises(ConfigLoadError):
            ConfigValidator.validate_all(invalid_config)
    
    def test_validate_invalid_rsi_max(self):
        """测试验证无效的 RSI_MAX"""
        invalid_config = {
            'RSI_MAX': 150,  # 超过100
        }
        with pytest.raises(ConfigLoadError):
            ConfigValidator.validate_all(invalid_config)
    
    def test_validate_invalid_max_loss_pct(self):
        """测试验证无效的 MAX_LOSS_PCT"""
        invalid_config = {
            'MAX_LOSS_PCT': 1.5,  # 超过1
        }
        with pytest.raises(ConfigLoadError):
            ConfigValidator.validate_all(invalid_config)
    
    def test_validate_invalid_log_level(self):
        """测试验证无效的 LOG_LEVEL"""
        invalid_config = {
            'LOG_LEVEL': 'INVALID',
        }
        with pytest.raises(ConfigLoadError):
            ConfigValidator.validate_all(invalid_config)
    
    def test_validate_empty_config(self):
        """测试验证空配置"""
        empty_config = {}
        # 空配置应该通过验证（只验证存在的配置项）
        ConfigValidator.validate_all(empty_config)


class TestConfigLoader:
    """配置加载器测试"""
    
    def test_load_default_config(self):
        """测试加载默认配置"""
        config = ConfigLoader.load()
        
        assert 'TOP_N' in config
        assert 'BREAKOUT_N' in config
        assert config['TOP_N'] == 20
        assert config['BREAKOUT_N'] == 60
    
    def test_load_json_config(self):
        """测试加载 JSON 配置文件"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({'TOP_N': 30, 'MA_FAST': 30}, f)
            temp_path = f.name
        
        try:
            config = ConfigLoader.load(temp_path)
            assert config['TOP_N'] == 30
            assert config['MA_FAST'] == 30
            # 其他配置项应该是默认值
            assert config['BREAKOUT_N'] == 60
        finally:
            os.unlink(temp_path)
    
    def test_load_yaml_config(self):
        """测试加载 YAML 配置文件"""
        pytest.importorskip('yaml')
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write('TOP_N: 25\n')
            f.write('MA_SLOW: 120\n')
            temp_path = f.name
        
        try:
            config = ConfigLoader.load(temp_path)
            assert config['TOP_N'] == 25
            assert config['MA_SLOW'] == 120
            # 其他配置项应该是默认值
            assert config['BREAKOUT_N'] == 60
        finally:
            os.unlink(temp_path)
    
    def test_load_nonexistent_file(self):
        """测试加载不存在的文件"""
        with pytest.raises(ConfigLoadError, match="配置文件不存在"):
            ConfigLoader.load('nonexistent_config.yaml')
    
    def test_load_unsupported_format(self):
        """测试加载不支持的格式"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write('test')
            temp_path = f.name
        
        try:
            with pytest.raises(ConfigLoadError, match="不支持的配置文件格式"):
                ConfigLoader.load(temp_path)
        finally:
            os.unlink(temp_path)
    
    def test_load_with_validation_disabled(self):
        """测试禁用验证加载配置"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({'TOP_N': -1}, f)  # 无效值
            temp_path = f.name
        
        try:
            # 禁用验证应该能加载
            config = ConfigLoader.load(temp_path, validate=False)
            assert config['TOP_N'] == -1
        finally:
            os.unlink(temp_path)
    
    def test_load_with_validation_enabled(self):
        """测试启用验证加载配置"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({'TOP_N': -1}, f)  # 无效值
            temp_path = f.name
        
        try:
            with pytest.raises(ConfigLoadError, match="配置验证失败"):
                ConfigLoader.load(temp_path, validate=True)
        finally:
            os.unlink(temp_path)
    
    def test_save_yaml_template(self):
        """测试保存 YAML 模板"""
        pytest.importorskip('yaml')
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            temp_path = f.name
        
        try:
            ConfigLoader.save_template(temp_path, format='yaml')
            assert Path(temp_path).exists()
            
            # 验证保存的文件可以重新加载
            config = ConfigLoader.load(temp_path)
            assert 'TOP_N' in config
        finally:
            os.unlink(temp_path)
    
    def test_save_json_template(self):
        """测试保存 JSON 模板"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            ConfigLoader.save_template(temp_path, format='json')
            assert Path(temp_path).exists()
            
            # 验证保存的文件可以重新加载
            config = ConfigLoader.load(temp_path)
            assert 'TOP_N' in config
        finally:
            os.unlink(temp_path)
    
    def test_save_unsupported_format(self):
        """测试保存不支持的格式"""
        with pytest.raises(ConfigLoadError, match="不支持的格式"):
            ConfigLoader.save_template('test.txt', format='xml')
    
    def test_config_merging(self):
        """测试配置合并"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({'TOP_N': 30}, f)
            temp_path = f.name
        
        try:
            config = ConfigLoader.load(temp_path)
            # 自定义值
            assert config['TOP_N'] == 30
            # 默认值
            assert config['BREAKOUT_N'] == 60
            assert config['MA_FAST'] == 20
        finally:
            os.unlink(temp_path)
    
    def test_invalid_json(self):
        """测试无效的 JSON 文件"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write('invalid json {{{')
            temp_path = f.name
        
        try:
            with pytest.raises(ConfigLoadError, match="JSON 文件解析失败"):
                ConfigLoader.load(temp_path)
        finally:
            os.unlink(temp_path)
    
    def test_invalid_yaml(self):
        """测试无效的 YAML 文件"""
        pytest.importorskip('yaml')
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write('invalid:\n  yaml: {\n    unclosed')
            temp_path = f.name
        
        try:
            with pytest.raises(ConfigLoadError, match="YAML 文件解析失败"):
                ConfigLoader.load(temp_path)
        finally:
            os.unlink(temp_path)


class TestConvenienceFunctions:
    """便捷函数测试"""
    
    def test_load_config_default(self):
        """测试加载默认配置的便捷函数"""
        config = load_config()
        assert 'TOP_N' in config
        assert config['TOP_N'] == 20
    
    def test_generate_config_template_yaml(self):
        """测试生成 YAML 模板的便捷函数"""
        pytest.importorskip('yaml')
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            temp_path = f.name
        
        try:
            generate_config_template(temp_path, format='yaml')
            assert Path(temp_path).exists()
        finally:
            os.unlink(temp_path)
    
    def test_generate_config_template_json(self):
        """测试生成 JSON 模板的便捷函数"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            generate_config_template(temp_path, format='json')
            assert Path(temp_path).exists()
        finally:
            os.unlink(temp_path)


class TestDefaultConfig:
    """默认配置测试"""
    
    def test_default_config_completeness(self):
        """测试默认配置的完整性"""
        config = ConfigLoader.load()
        
        required_keys = [
            'TOP_N', 'BREAKOUT_N', 'MA_FAST', 'MA_SLOW',
            'VOL_LOOKBACK', 'VOL_CONFIRM_MULT', 'RSI_MAX',
            'MAX_LOSS_PCT', 'ATR_N', 'ATR_MULT',
            'MIN_LIST_DAYS', 'MIN_PRICE', 'MIN_AVG_AMOUNT_20D',
            'EXCLUDE_ONE_WORD_LIMITUP',
            'CACHE_DIR', 'SLEEP_PER_CALL', 'MAX_CALLS_PER_MINUTE',
            'STOCK_BASIC_TTL_DAYS', 'TRADE_CAL_TTL_DAYS',
            'SAVE_REPORT', 'REPORT_DIR',
            'DEFAULT_HOLDING_DAYS', 'INDEX_CODE',
            'LOG_LEVEL', 'LOG_DIR', 'LOG_CONSOLE_OUTPUT',
            'LOG_FILE_OUTPUT', 'LOG_MAX_FILE_SIZE', 'LOG_BACKUP_COUNT',
        ]
        
        for key in required_keys:
            assert key in config, f"默认配置缺少键: {key}"
    
    def test_default_config_types(self):
        """测试默认配置的类型"""
        config = ConfigLoader.load()
        
        assert isinstance(config['TOP_N'], int)
        assert isinstance(config['BREAKOUT_N'], int)
        assert isinstance(config['MAX_LOSS_PCT'], float)
        assert isinstance(config['EXCLUDE_ONE_WORD_LIMITUP'], bool)
        assert isinstance(config['CACHE_DIR'], str)


class TestProjectConfigFiles:
    """项目配置文件测试"""
    
    def test_yaml_config_exists(self):
        """测试项目 YAML 配置文件是否存在"""
        config_path = Path(__file__).parent.parent / 'config.yaml'
        assert config_path.exists(), "项目配置文件 config.yaml 不存在"
    
    def test_json_config_exists(self):
        """测试项目 JSON 配置文件是否存在"""
        config_path = Path(__file__).parent.parent / 'config.json'
        assert config_path.exists(), "项目配置文件 config.json 不存在"
    
    def test_load_project_yaml_config(self):
        """测试加载项目 YAML 配置文件"""
        pytest.importorskip('yaml')
        
        config_path = Path(__file__).parent.parent / 'config.yaml'
        config = ConfigLoader.load(str(config_path))
        
        assert 'TOP_N' in config
        assert 'LOG_LEVEL' in config
    
    def test_load_project_json_config(self):
        """测试加载项目 JSON 配置文件"""
        config_path = Path(__file__).parent.parent / 'config.json'
        config = ConfigLoader.load(str(config_path))
        
        assert 'TOP_N' in config
        assert 'LOG_LEVEL' in config
