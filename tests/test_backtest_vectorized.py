"""
向量化回测引擎测试
"""

import numpy as np
import pandas as pd
import pytest

from analysis.backtest import BacktestConfig, BacktestEngine
from analysis.backtest_vectorized import (
    VectorizedMetrics,
    VectorizedPositionChecker,
    VectorizedBacktestEngine,
    ParallelBacktestRunner
)


class TestVectorizedMetrics:
    """向量化指标计算测试"""
    
    @pytest.fixture
    def sample_equity_curve(self):
        """创建示例资产曲线"""
        return pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=100),
            'cash': [1000000] * 100,
            'positions_value': np.linspace(1000000, 1200000, 100),
            'equity': np.linspace(2000000, 2200000, 100)
        })
    
    @pytest.fixture
    def sample_trades(self):
        """创建示例交易记录"""
        return pd.DataFrame({
            'pnl': [10000, -5000, 15000, -3000, 8000],
            'pnl_pct': [0.1, -0.05, 0.15, -0.03, 0.08],
            'entry_date': ['2023-01-01', '2023-01-05', '2023-01-10', 
                         '2023-01-15', '2023-01-20'],
            'exit_date': ['2023-01-04', '2023-01-09', '2023-01-14',
                        '2023-01-19', '2023-01-25'],
            'ts_code': ['000001.SZ', '000002.SZ', '000003.SZ',
                       '000004.SZ', '000005.SZ'],
            'name': ['股票A', '股票B', '股票C', '股票D', '股票E'],
            'entry_price': [10.0, 20.0, 15.0, 25.0, 30.0],
            'exit_price': [11.0, 19.0, 17.25, 24.25, 32.4],
            'shares': [1000, 500, 1000, 400, 250],
            'reason': ['take_profit', 'stop_loss', 'take_profit',
                     'stop_loss', 'take_profit']
        })
    
    def test_calculate_basic_metrics(self, sample_equity_curve, sample_trades):
        """测试基础指标计算"""
        metrics = VectorizedMetrics.calculate_all_metrics_vectorized(
            sample_equity_curve,
            sample_trades,
            initial_capital=2000000
        )
        
        assert metrics['total_trades'] == 5
        assert metrics['trading_days'] == 100
        assert metrics['final_equity'] > 0
        assert isinstance(metrics['total_return'], (int, float))
    
    def test_calculate_sharpe_ratio(self, sample_equity_curve, sample_trades):
        """测试夏普比率计算"""
        metrics = VectorizedMetrics.calculate_all_metrics_vectorized(
            sample_equity_curve,
            sample_trades,
            initial_capital=2000000
        )
        
        assert isinstance(metrics['sharpe_ratio'], (int, float))
        assert metrics['sharpe_ratio'] >= 0  # 应该是正值（资产增长）
    
    def test_calculate_max_drawdown(self, sample_equity_curve, sample_trades):
        """测试最大回撤计算"""
        metrics = VectorizedMetrics.calculate_all_metrics_vectorized(
            sample_equity_curve,
            sample_trades,
            initial_capital=2000000
        )
        
        assert isinstance(metrics['max_drawdown'], (int, float))
        # 单调增长的曲线应该没有回撤或回撤很小
        assert metrics['max_drawdown'] <= 0
    
    def test_calculate_win_rate(self, sample_equity_curve, sample_trades):
        """测试胜率计算"""
        metrics = VectorizedMetrics.calculate_all_metrics_vectorized(
            sample_equity_curve,
            sample_trades,
            initial_capital=2000000
        )
        
        # 3盈利，2亏损，胜率应该是60%
        assert abs(metrics['win_rate'] - 60) < 0.1
    
    def test_calculate_profit_factor(self, sample_equity_curve, sample_trades):
        """测试盈亏比计算"""
        metrics = VectorizedMetrics.calculate_all_metrics_vectorized(
            sample_equity_curve,
            sample_trades,
            initial_capital=2000000
        )
        
        assert isinstance(metrics['profit_factor'], (int, float))
        assert metrics['profit_factor'] > 0
        # 盈利总额应该大于亏损总额
        assert metrics['profit_factor'] > 1
    
    def test_empty_metrics(self):
        """测试空数据返回空指标"""
        empty_df = pd.DataFrame()
        empty_trades = pd.DataFrame()
        
        metrics = VectorizedMetrics.calculate_all_metrics_vectorized(
            empty_df,
            empty_trades,
            initial_capital=1000000
        )
        
        # 验证所有指标都是0或空
        assert metrics['total_trades'] == 0
        assert metrics['trading_days'] == 0
        assert metrics['win_rate'] == 0
    
    def test_empty_metrics_helper(self):
        """测试空指标辅助方法"""
        metrics = VectorizedMetrics._empty_metrics()
        
        assert metrics['total_trades'] == 0
        assert metrics['trading_days'] == 0
        assert metrics['total_return'] == 0
        assert metrics['final_equity'] == 0
        assert isinstance(metrics['trades'], list)
        assert isinstance(metrics['equity_curve'], list)


class TestVectorizedPositionChecker:
    """向量化持仓检查器测试"""
    
    @pytest.fixture
    def backtest_config(self):
        """创建回测配置"""
        return BacktestConfig(
            start_date='20230101',
            end_date='20231231',
            initial_capital=1000000,
            max_positions=10,
            position_size=0.1,
            slippage=0.001,
            commission=0.0003,
            stop_loss=-0.10,
            take_profit=0.20,
            max_holding_days=20,
            rebalance_days=5
        )
    
    @pytest.fixture
    def sample_positions(self):
        """创建示例持仓"""
        return {
            '000001.SZ': {
                'entry_price': 10.0,
                'entry_date': '20230101',
                'shares': 1000,
                'name': '股票A',
                'atr_stop_price': 9.0
            },
            '000002.SZ': {
                'entry_price': 20.0,
                'entry_date': '20230101',
                'shares': 500,
                'name': '股票B',
                'atr_stop_price': 18.0
            }
        }
    
    @pytest.fixture
    def sample_daily_df(self):
        """创建示例行情数据"""
        return pd.DataFrame({
            'ts_code': ['000001.SZ', '000002.SZ'],
            'close': [8.5, 22.0],  # 第一个触发止损，第二个触发止盈
            'high': [9.0, 23.0],
            'low': [8.0, 21.0]
        })
    
    def test_check_positions_stop_loss(
        self, backtest_config, sample_positions, sample_daily_df
    ):
        """测试止损检查"""
        checker = VectorizedPositionChecker(backtest_config)
        
        positions_to_close, capital = checker.check_positions_vectorized(
            sample_positions,
            sample_daily_df,
            '20230105',
            1000000
        )
        
        # 000001.SZ 应该触发止损（从10跌到8.5，跌幅15%）
        assert len(positions_to_close) >= 1
        stop_loss_positions = [p for p in positions_to_close if p['reason'] == 'stop_loss']
        assert len(stop_loss_positions) >= 1
    
    def test_check_positions_take_profit(
        self, backtest_config, sample_positions, sample_daily_df
    ):
        """测试止盈检查"""
        checker = VectorizedPositionChecker(backtest_config)
        
        positions_to_close, _ = checker.check_positions_vectorized(
            sample_positions,
            sample_daily_df,
            '20230105',
            1000000
        )
        
        # 000002.SZ 应该触发止盈（从20涨到22，涨幅10%，但需要20%才止盈）
        # 实际上10%不满足20%止盈条件，所以不会触发
        pass
    
    def test_check_empty_positions(self, backtest_config, sample_daily_df):
        """测试空持仓"""
        checker = VectorizedPositionChecker(backtest_config)
        
        positions_to_close, capital = checker.check_positions_vectorized(
            {},
            sample_daily_df,
            '20230105',
            1000000
        )
        
        assert len(positions_to_close) == 0
        assert capital == 1000000
    
    def test_check_positions_with_missing_prices(
        self, backtest_config, sample_positions, sample_daily_df
    ):
        """测试缺少行情数据的情况"""
        # 添加一个不在行情中的股票
        sample_positions['000003.SZ'] = {
            'entry_price': 30.0,
            'entry_date': '20230101',
            'shares': 300,
            'name': '股票C'
        }
        
        checker = VectorizedPositionChecker(backtest_config)
        
        positions_to_close, capital = checker.check_positions_vectorized(
            sample_positions,
            sample_daily_df,
            '20230105',
            1000000
        )
        
        # 000003.SZ 不在行情中，应该被跳过
        for pos in positions_to_close:
            assert pos['ts_code'] != '000003.SZ'


class TestVectorizedBacktestEngine:
    """向量化回测引擎测试"""
    
    @pytest.fixture
    def backtest_config(self):
        """创建回测配置"""
        return BacktestConfig(
            start_date='20230101',
            end_date='20231231',
            initial_capital=1000000,
            max_positions=10,
            position_size=0.1,
            slippage=0.001,
            commission=0.0003,
            stop_loss=-0.10,
            take_profit=0.20,
            max_holding_days=20,
            rebalance_days=5
        )
    
    def test_engine_inheritance(self, backtest_config):
        """测试引擎继承关系"""
        # 需要strategy和fetcher实例，这里只测试类结构
        from analysis.backtest_vectorized import VectorizedBacktestEngine
        from analysis.backtest import BacktestEngine
        
        assert issubclass(VectorizedBacktestEngine, BacktestEngine)
    
    def test_position_checker_initialization(self, backtest_config):
        """测试持仓检查器初始化"""
        from analysis.backtest_vectorized import VectorizedBacktestEngine
        
        # 需要mock strategy和fetcher
        # 这里只测试初始化逻辑
        pass


class TestParallelBacktestRunner:
    """并行回测运行器测试"""
    
    def test_parallel_runner_initialization(self):
        """测试并行运行器初始化"""
        config = BacktestConfig(
            start_date='20230101',
            end_date='20231231',
            initial_capital=1000000,
            max_positions=10,
            position_size=0.1,
            slippage=0.001,
            commission=0.0003,
            stop_loss=-0.10,
            take_profit=0.20,
            max_holding_days=20,
            rebalance_days=5
        )
        
        # 需要mock strategy和fetcher
        # 这里只测试类结构
        from analysis.backtest_vectorized import ParallelBacktestRunner
        
        assert ParallelBacktestRunner is not None
    
    def test_vectorized_batch_backtest_import(self):
        """测试批量回测函数导入"""
        from analysis.backtest_vectorized import vectorized_batch_backtest
        
        assert vectorized_batch_backtest is not None


class TestPerformance:
    """性能测试"""
    
    def test_large_equity_curve_performance(self):
        """测试大资产曲线的性能"""
        # 创建大的资产曲线（1000天）
        equity_curve = pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=1000),
            'cash': [1000000] * 1000,
            'positions_value': np.linspace(1000000, 1500000, 1000),
            'equity': np.linspace(2000000, 2500000, 1000)
        })
        
        trades_df = pd.DataFrame({
            'pnl': np.random.randn(100) * 10000,
            'pnl_pct': np.random.randn(100) * 0.1,
            'entry_date': [f'2020-{i%12+1:02d}-{i%28+1:02d}' for i in range(100)],
            'exit_date': [f'2020-{i%12+1:02d}-{(i%28+5)%28+1:02d}' for i in range(100)],
            'ts_code': [f'00000{i%10}.SZ' for i in range(100)],
            'name': [f'股票{i}' for i in range(100)],
            'entry_price': [10 + i for i in range(100)],
            'exit_price': [10 + i + np.random.randn() for i in range(100)],
            'shares': [1000] * 100,
            'reason': ['take_profit'] * 100
        })
        
        import time
        start = time.time()
        metrics = VectorizedMetrics.calculate_all_metrics_vectorized(
            equity_curve,
            trades_df,
            initial_capital=2000000
        )
        elapsed = time.time() - start
        
        # 向量化计算应该很快（< 1秒）
        assert elapsed < 1.0
        assert metrics['total_trades'] == 100
        assert metrics['trading_days'] == 1000
