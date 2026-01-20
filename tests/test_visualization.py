"""
趋势雷达选股系统 - 可视化模块测试
测试各种图表绘制功能
"""
import pytest
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tempfile

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt

from visualization.plotter import (
    Plotter,
    plot_stock_candlestick,
    plot_stock_indicators,
    plot_backtest_results,
    plot_drawdown_chart,
    plot_monthly_returns,
    plot_parameter_heatmap,
    plot_parameter_sensitivity
)


@pytest.fixture
def sample_stock_data():
    """创建示例股票数据"""
    np.random.seed(42)
    n_days = 120
    dates = pd.date_range(start='2023-01-01', periods=n_days, freq='D')

    # 生成随机价格数据
    close = np.cumsum(np.random.randn(n_days) * 0.02 + 0.001) + 100
    open_price = close + np.random.randn(n_days) * 0.5
    high = np.maximum(open_price, close) + np.random.rand(n_days) * 0.5
    low = np.minimum(open_price, close) - np.random.rand(n_days) * 0.5
    volume = np.random.randint(1000000, 10000000, n_days)

    df = pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    }, index=dates)

    return df


@pytest.fixture
def sample_backtest_results():
    """创建示例回测结果"""
    np.random.seed(42)
    n_days = 252

    # 净值曲线
    dates = pd.date_range(start='2023-01-01', periods=n_days, freq='D')
    equity = 100000 * (1 + np.cumsum(np.random.randn(n_days) * 0.005))
    benchmark_equity = 100000 * (1 + np.cumsum(np.random.randn(n_days) * 0.003))

    # 回撤
    drawdown = -np.abs(np.cumsum(np.random.randn(n_days) * 0.02))
    drawdown = np.clip(drawdown, -30, 0)

    equity_curve = pd.DataFrame({
        'equity': equity,
        'benchmark_equity': benchmark_equity,
        'drawdown': drawdown
    }, index=dates)

    # 月度收益
    months = pd.date_range(start='2023-01-01', periods=12, freq='M')
    monthly_returns = pd.DataFrame({
        'returns': np.random.randn(12) * 5
    }, index=months)

    # 交易记录
    trades = pd.DataFrame({
        'entry_date': pd.date_range(start='2023-01-01', periods=10, freq='15D'),
        'exit_date': pd.date_range(start='2023-01-15', periods=10, freq='15D'),
        'pnl_pct': np.random.randn(10) * 3
    })

    return {
        'equity_curve': equity_curve,
        'monthly_returns': monthly_returns,
        'trades': trades
    }


@pytest.fixture
def sample_optimization_results():
    """创建示例参数优化结果"""
    param1_values = [20, 40, 60, 80]
    param2_values = [10, 20, 30]

    results = []
    for p1 in param1_values:
        for p2 in param2_values:
            results.append({
                'BREAKOUT_N': p1,
                'MA_FAST': p2,
                'total_return': np.random.randn() * 10 + 15,
                'sharpe_ratio': np.random.randn() * 0.5 + 1.2
            })

    return pd.DataFrame(results)


class TestPlotter:
    """测试Plotter类"""

    def test_setup_style(self):
        """测试设置绘图风格"""
        Plotter.setup_style()
        assert plt.style.available is not None


class TestStockCandlestick:
    """测试K线图绘制"""

    def test_plot_stock_candlestick_basic(self, sample_stock_data):
        """测试基本K线图"""
        fig = plot_stock_candlestick(sample_stock_data)

        assert fig is not None
        assert len(fig.axes) >= 2
        plt.close(fig)

    def test_plot_stock_candlestick_with_indicators(self, sample_stock_data):
        """测试带指标的K线图"""
        fig = plot_stock_candlestick(
            sample_stock_data,
            indicators=['ma', 'bollinger', 'volume', 'macd']
        )

        assert fig is not None
        assert len(fig.axes) >= 3
        plt.close(fig)

    def test_plot_stock_candlestick_missing_columns(self):
        """测试缺少必需列的情况"""
        df = pd.DataFrame({
            'open': [100],
            'high': [105],
            'close': [103]
        })

        with pytest.raises(ValueError, match="缺少必需列"):
            plot_stock_candlestick(df)

    def test_plot_stock_candlestick_save(self, sample_stock_data):
        """测试保存K线图"""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            save_path = f.name

        try:
            fig = plot_stock_candlestick(sample_stock_data, save_path=save_path)
            assert os.path.exists(save_path)
            os.remove(save_path)
            plt.close(fig)
        except:
            if os.path.exists(save_path):
                os.remove(save_path)
            raise


class TestStockIndicators:
    """测试技术指标图绘制"""

    def test_plot_stock_indicators_basic(self, sample_stock_data):
        """测试基本指标图"""
        fig = plot_stock_indicators(sample_stock_data)

        assert fig is not None
        assert len(fig.axes) >= 2
        plt.close(fig)

    def test_plot_stock_indicators_custom(self, sample_stock_data):
        """测试自定义指标图"""
        fig = plot_stock_indicators(
            sample_stock_data,
            indicators=['rsi', 'atr'],
            figsize=(14, 6)
        )

        assert fig is not None
        assert len(fig.axes) == 2
        plt.close(fig)

    def test_plot_stock_indicators_all(self, sample_stock_data):
        """测试所有指标图"""
        fig = plot_stock_indicators(
            sample_stock_data,
            indicators=['rsi', 'kdj', 'cci', 'atr']
        )

        assert fig is not None
        assert len(fig.axes) == 4
        plt.close(fig)

    def test_plot_stock_indicators_save(self, sample_stock_data):
        """测试保存指标图"""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            save_path = f.name

        try:
            fig = plot_stock_indicators(sample_stock_data, save_path=save_path)
            assert os.path.exists(save_path)
            os.remove(save_path)
            plt.close(fig)
        except:
            if os.path.exists(save_path):
                os.remove(save_path)
            raise


class TestBacktestResults:
    """测试回测结果绘制"""

    def test_plot_backtest_results_basic(self, sample_backtest_results):
        """测试基本回测结果图"""
        fig = plot_backtest_results(sample_backtest_results)

        assert fig is not None
        assert len(fig.axes) >= 3
        plt.close(fig)

    def test_plot_backtest_results_with_trades(self, sample_backtest_results):
        """测试带交易记录的回测结果图"""
        fig = plot_backtest_results(sample_backtest_results)

        assert fig is not None
        plt.close(fig)

    def test_plot_backtest_results_save(self, sample_backtest_results):
        """测试保存回测结果图"""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            save_path = f.name

        try:
            fig = plot_backtest_results(sample_backtest_results, save_path=save_path)
            assert os.path.exists(save_path)
            os.remove(save_path)
            plt.close(fig)
        except:
            if os.path.exists(save_path):
                os.remove(save_path)
            raise


class TestDrawdownChart:
    """测试回撤图"""

    def test_plot_drawdown_chart_basic(self, sample_backtest_results):
        """测试基本回撤图"""
        equity_curve = sample_backtest_results['equity_curve']
        fig = plot_drawdown_chart(equity_curve)

        assert fig is not None
        assert len(fig.axes) == 2
        plt.close(fig)

    def test_plot_drawdown_chart_save(self, sample_backtest_results):
        """测试保存回撤图"""
        equity_curve = sample_backtest_results['equity_curve']

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            save_path = f.name

        try:
            fig = plot_drawdown_chart(equity_curve, save_path=save_path)
            assert os.path.exists(save_path)
            os.remove(save_path)
            plt.close(fig)
        except:
            if os.path.exists(save_path):
                os.remove(save_path)
            raise


class TestMonthlyReturns:
    """测试月度收益图"""

    def test_plot_monthly_returns_basic(self, sample_backtest_results):
        """测试基本月度收益图"""
        monthly_returns = sample_backtest_results['monthly_returns']
        fig = plot_monthly_returns(monthly_returns)

        assert fig is not None
        assert len(fig.axes) >= 2
        plt.close(fig)

    def test_plot_monthly_returns_save(self, sample_backtest_results):
        """测试保存月度收益图"""
        monthly_returns = sample_backtest_results['monthly_returns']

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            save_path = f.name

        try:
            fig = plot_monthly_returns(monthly_returns, save_path=save_path)
            assert os.path.exists(save_path)
            os.remove(save_path)
            plt.close(fig)
        except:
            if os.path.exists(save_path):
                os.remove(save_path)
            raise


class TestParameterHeatmap:
    """测试参数热力图"""

    def test_plot_parameter_heatmap_basic(self, sample_optimization_results):
        """测试基本参数热力图"""
        fig = plot_parameter_heatmap(
            sample_optimization_results,
            'BREAKOUT_N',
            'MA_FAST',
            'total_return'
        )

        assert fig is not None
        assert len(fig.axes) >= 1
        plt.close(fig)

    def test_plot_parameter_heatmap_different_metric(self, sample_optimization_results):
        """测试不同指标的参数热力图"""
        fig = plot_parameter_heatmap(
            sample_optimization_results,
            'BREAKOUT_N',
            'MA_FAST',
            'sharpe_ratio'
        )

        assert fig is not None
        plt.close(fig)

    def test_plot_parameter_heatmap_save(self, sample_optimization_results):
        """测试保存参数热力图"""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            save_path = f.name

        try:
            fig = plot_parameter_heatmap(
                sample_optimization_results,
                'BREAKOUT_N',
                'MA_FAST',
                'total_return',
                save_path=save_path
            )
            assert os.path.exists(save_path)
            os.remove(save_path)
            plt.close(fig)
        except:
            if os.path.exists(save_path):
                os.remove(save_path)
            raise


class TestParameterSensitivity:
    """测试参数敏感性分析"""

    def test_plot_parameter_sensitivity_basic(self, sample_optimization_results):
        """测试基本参数敏感性图"""
        fig = plot_parameter_sensitivity(
            sample_optimization_results,
            ['BREAKOUT_N', 'MA_FAST'],
            'total_return'
        )

        assert fig is not None
        assert len(fig.axes) == 2
        plt.close(fig)

    def test_plot_parameter_sensitivity_single_param(self, sample_optimization_results):
        """测试单个参数的敏感性图"""
        fig = plot_parameter_sensitivity(
            sample_optimization_results,
            ['BREAKOUT_N'],
            'total_return'
        )

        assert fig is not None
        assert len(fig.axes) == 1
        plt.close(fig)

    def test_plot_parameter_sensitivity_save(self, sample_optimization_results):
        """测试保存参数敏感性图"""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            save_path = f.name

        try:
            fig = plot_parameter_sensitivity(
                sample_optimization_results,
                ['BREAKOUT_N', 'MA_FAST'],
                'total_return',
                save_path=save_path
            )
            assert os.path.exists(save_path)
            os.remove(save_path)
            plt.close(fig)
        except:
            if os.path.exists(save_path):
                os.remove(save_path)
            raise


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
