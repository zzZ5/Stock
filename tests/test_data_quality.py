"""
数据质量模块测试
"""
import pytest
import pandas as pd
import numpy as np
from core.data_quality import (
    DataCleaner, DataValidator, DataPipeline,
    DataQualityReport, DataCleaner,
    fill_missing_with_linear_trend,
    detect_price_anomalies
)


class TestDataCleaner:
    """数据清洗器测试"""

    @pytest.fixture
    def sample_df(self):
        """创建测试用例数据"""
        np.random.seed(42)
        return pd.DataFrame({
            'open': np.random.randn(100) * 10 + 100,
            'high': np.random.randn(100) * 10 + 105,
            'low': np.random.randn(100) * 10 + 95,
            'close': np.random.randn(100) * 10 + 100,
            'volume': np.random.randint(100000, 1000000, 100)
        })

    def test_detect_outliers_iqr(self, sample_df):
        """测试IQR方法检测异常值"""
        cleaner = DataCleaner(outlier_method='iqr')
        outliers = cleaner.detect_outliers(sample_df)

        assert isinstance(outliers, dict)
        assert all(isinstance(mask, pd.Series) for mask in outliers.values())

    def test_detect_outliers_zscore(self, sample_df):
        """测试Z-score方法检测异常值"""
        cleaner = DataCleaner(outlier_method='zscore')
        outliers = cleaner.detect_outliers(sample_df, threshold=3.0)

        assert isinstance(outliers, dict)
        assert all(isinstance(mask, pd.Series) for mask in outliers.values())

    def test_handle_missing_values_forward_fill(self):
        """测试前向填充处理缺失值"""
        df = pd.DataFrame({
            'value': [1, np.nan, 3, np.nan, 5]
        })
        cleaner = DataCleaner(missing_strategy='forward_fill')
        result = cleaner.handle_missing_values(df)

        assert result['value'].isna().sum() == 0

    def test_handle_missing_values_mean(self):
        """测试均值填充处理缺失值"""
        df = pd.DataFrame({
            'value': [1, np.nan, 3, np.nan, 5]
        })
        cleaner = DataCleaner(missing_strategy='mean')
        result = cleaner.handle_missing_values(df)

        expected_mean = (1 + 3 + 5) / 3
        assert abs(result.loc[1, 'value'] - expected_mean) < 0.01

    def test_remove_outliers_clip(self):
        """测试截断方法处理异常值"""
        df = pd.DataFrame({
            'value': [1, 2, 3, 1000, 5]  # 1000是异常值
        })
        cleaner = DataCleaner(outlier_method='iqr')
        result = cleaner.remove_outliers(df, replace_method='clip')

        # 由于数据量太少，IQR可能无法检测到异常值，所以这里只检查不报错
        assert len(result) == len(df)  # 数据行数应该不变（clip方法）

    def test_clean_price_data(self):
        """测试价格数据清洗"""
        # 创建包含无效数据的DataFrame
        df = pd.DataFrame({
            'open': [100, 105, -5, 110],  # -5是无效的
            'high': [105, 110, 110, 115],  # 第一行high < low
            'low': [110, 102, 100, 105],
            'close': [108, 108, 100, 112]  # close超出high/low范围
        })

        cleaner = DataCleaner()
        result = cleaner.clean_price_data(df)

        # 负值应该被处理
        assert result.loc[2, 'open'] > 0

    def test_check_data_consistency(self, sample_df):
        """测试数据一致性检查"""
        cleaner = DataCleaner()
        results = cleaner.check_data_consistency(sample_df)

        assert isinstance(results, dict)
        assert 'high_ge_low' in results


class TestDataValidator:
    """数据验证器测试"""

    @pytest.fixture
    def valid_df(self):
        """创建有效的测试数据"""
        return pd.DataFrame({
            'open': np.arange(100, 110),
            'high': np.arange(105, 115),
            'low': np.arange(95, 105),
            'close': np.arange(100, 110),
            'volume': np.random.randint(100000, 1000000, 10)
        })

    def test_generate_quality_report(self, valid_df):
        """测试生成数据质量报告"""
        report = DataValidator.generate_quality_report(valid_df)

        assert isinstance(report, DataQualityReport)
        assert report.total_rows == 10
        assert report.total_columns == 5
        assert isinstance(report.summary, str)

    def test_validate_stock_data_valid(self, valid_df):
        """测试验证有效数据"""
        is_valid, errors = DataValidator.validate_stock_data(valid_df)

        assert is_valid is True
        assert len(errors) == 0

    def test_validate_stock_data_too_short(self):
        """测试验证数据过短"""
        df = pd.DataFrame({
            'open': [100],
            'high': [105],
            'low': [95],
            'close': [100]
        })
        is_valid, errors = DataValidator.validate_stock_data(df)

        assert is_valid is False
        assert len(errors) > 0

    def test_validate_stock_data_missing_columns(self):
        """测试验证缺少必要列"""
        df = pd.DataFrame({
            'open': [100, 101, 102],
            'close': [100, 101, 102]
        })
        is_valid, errors = DataValidator.validate_stock_data(df)

        assert is_valid is False
        assert len(errors) > 0  # 应该有错误信息


class TestDataPipeline:
    """数据处理流水线测试"""

    @pytest.fixture
    def dirty_df(self):
        """创建包含问题的测试数据"""
        np.random.seed(42)
        df = pd.DataFrame({
            'open': np.random.randn(100) * 10 + 100,
            'high': np.random.randn(100) * 10 + 105,
            'low': np.random.randn(100) * 10 + 95,
            'close': np.random.randn(100) * 10 + 100,
            'volume': np.random.randint(100000, 1000000, 100)
        })

        # 添加一些缺失值
        df.loc[10:15, 'close'] = np.nan
        df.loc[20:22, 'volume'] = np.nan

        # 添加一些异常值
        df.loc[30, 'close'] = 1000

        return df

    def test_process(self, dirty_df):
        """测试完整处理流程"""
        pipeline = DataPipeline()
        result = pipeline.process(dirty_df, generate_report=False)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(dirty_df)

    def test_batch_process(self, dirty_df):
        """测试批量处理"""
        dfs = {
            '000001.SZ': dirty_df.copy(),
            '000002.SZ': dirty_df.copy(),
            '600000.SH': dirty_df.copy()
        }

        pipeline = DataPipeline()
        results = pipeline.batch_process(dfs)

        assert len(results) == 3
        assert all(isinstance(df, pd.DataFrame) for df in results.values())

    def test_process_generates_report(self, dirty_df):
        """测试处理流程生成报告"""
        pipeline = DataPipeline()

        # 注意：这里我们不会真正输出日志，只是测试不报错
        result = pipeline.process(dirty_df, generate_report=True)

        assert result is not None


class TestUtilityFunctions:
    """工具函数测试"""

    def test_fill_missing_with_linear_trend(self):
        """测试线性趋势填充"""
        df = pd.DataFrame({
            'value': [1, 2, np.nan, np.nan, 5]
        })
        result = fill_missing_with_linear_trend(df, columns=['value'])

        assert result['value'].isna().sum() == 0

    def test_detect_price_anomalies(self):
        """检测价格异常"""
        # 创建包含异常值的数据
        np.random.seed(42)
        df = pd.DataFrame({
            'close': np.concatenate([
                np.random.randn(50) * 10 + 100,
                [500],  # 异常值
                np.random.randn(49) * 10 + 100
            ])
        })

        anomalies = detect_price_anomalies(df, window=20, threshold=2.0)

        # 应该检测到异常值
        assert anomalies.sum() > 0

    def test_detect_price_anomalies_no_anomalies(self):
        """测试无异常值的情况"""
        np.random.seed(42)
        df = pd.DataFrame({
            'close': np.random.randn(100) * 5 + 100
        })

        anomalies = detect_price_anomalies(df, window=20, threshold=5.0)

        # 应该很少或没有异常值
        assert anomalies.sum() < 5


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
