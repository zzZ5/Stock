"""
趋势雷达选股系统 - 配置测试
测试配置参数的合法性
"""
import pytest
import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config.settings as config


class TestConfigParameters:
    """测试配置参数"""
    
    def test_top_n_positive(self):
        """测试TOP_N应该为正数"""
        assert config.TOP_N > 0
        assert isinstance(config.TOP_N, int)
    
    def test_breakout_n_positive(self):
        """测试BREAKOUT_N应该为正数"""
        assert config.BREAKOUT_N > 0
        assert isinstance(config.BREAKOUT_N, int)
    
    def test_ma_periods_valid(self):
        """测试均线周期应该合理"""
        assert config.MA_FAST > 0
        assert config.MA_SLOW > 0
        assert config.MA_SLOW > config.MA_FAST  # 慢线周期应该大于快线
        assert isinstance(config.MA_FAST, int)
        assert isinstance(config.MA_SLOW, int)
    
    def test_vol_lookback_positive(self):
        """测试VOL_LOOKBACK应该为正数"""
        assert config.VOL_LOOKBACK > 0
        assert isinstance(config.VOL_LOOKBACK, int)
    
    def test_vol_confirm_mult_positive(self):
        """测试量能确认倍数应该为正数"""
        assert config.VOL_CONFIRM_MULT > 1.0
        assert isinstance(config.VOL_CONFIRM_MULT, (int, float))
    
    def test_rsi_max_range(self):
        """测试RSI_MAX应该在合理范围内"""
        assert 50 <= config.RSI_MAX <= 100
        assert isinstance(config.RSI_MAX, (int, float))
    
    def test_loss_pct_negative(self):
        """测试止损比例应该为正数"""
        assert config.MAX_LOSS_PCT > 0
        assert config.MAX_LOSS_PCT <= 0.5  # 最大止损不应超过50%
        assert isinstance(config.MAX_LOSS_PCT, (int, float))
    
    def test_atr_params_valid(self):
        """测试ATR参数应该合理"""
        assert config.ATR_N > 0
        assert config.ATR_MULT > 0
        assert isinstance(config.ATR_N, int)
        assert isinstance(config.ATR_MULT, (int, float))
    
    def test_min_list_days_positive(self):
        """测试MIN_LIST_DAYS应该为正数"""
        assert config.MIN_LIST_DAYS >= 60  # 至少上市60天
        assert isinstance(config.MIN_LIST_DAYS, int)
    
    def test_min_price_positive(self):
        """测试MIN_PRICE应该为正数"""
        assert config.MIN_PRICE > 0
        assert isinstance(config.MIN_PRICE, (int, float))
    
    def test_min_avg_amount_positive(self):
        """测试MIN_AVG_AMOUNT_20D应该为正数"""
        assert config.MIN_AVG_AMOUNT_20D > 0
        assert isinstance(config.MIN_AVG_AMOUNT_20D, (int, float))
    
    def test_exclude_one_word_limitup_bool(self):
        """测试EXCLUDE_ONE_WORD_LIMITUP应该是布尔值"""
        assert isinstance(config.EXCLUDE_ONE_WORD_LIMITUP, bool)
    
    def test_log_level_valid(self):
        """测试日志级别应该有效"""
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        assert config.LOG_LEVEL in valid_levels
        assert isinstance(config.LOG_LEVEL, str)
    
    def test_log_dir_string(self):
        """测试日志目录应该是字符串"""
        assert isinstance(config.LOG_DIR, str)
        assert len(config.LOG_DIR) > 0
    
    def test_log_output_bools(self):
        """测试日志输出配置应该是布尔值"""
        assert isinstance(config.LOG_CONSOLE_OUTPUT, bool)
        assert isinstance(config.LOG_FILE_OUTPUT, bool)
    
    def test_log_file_size_positive(self):
        """测试日志文件大小应该是正数"""
        assert config.LOG_MAX_FILE_SIZE > 0
        assert isinstance(config.LOG_MAX_FILE_SIZE, int)
    
    def test_log_backup_count_positive(self):
        """测试日志备份数量应该是正数"""
        assert config.LOG_BACKUP_COUNT >= 0
        assert isinstance(config.LOG_BACKUP_COUNT, int)
    
    def test_default_holding_days_positive(self):
        """测试DEFAULT_HOLDING_DAYS应该是正数"""
        assert config.DEFAULT_HOLDING_DAYS > 0
        assert isinstance(config.DEFAULT_HOLDING_DAYS, int)
    
    def test_index_code_valid(self):
        """测试指数代码格式应该正确"""
        assert isinstance(config.INDEX_CODE, str)
        assert len(config.INDEX_CODE) > 0
        # 格式应该类似 "000300.SH"
        assert '.' in config.INDEX_CODE


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
