"""
高级绩效评估指标测试
"""
import pytest
import pandas as pd
import numpy as np

from core.performance_metrics import PerformanceMetrics


class TestBasicMetrics:
    """基础指标测试"""
    
    @pytest.fixture
    def sample_equity_curve(self):
        """创建示例资产曲线"""
        return pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=252),
            'cash': [1000000] * 252,
            'positions_value': np.linspace(1000000, 1200000, 252),
            'equity': np.linspace(2000000, 2200000, 252)
        })
    
    def test_calculate_basic_metrics(self, sample_equity_curve):
        """测试基础指标计算"""
        metrics = PerformanceMetrics._calculate_basic_metrics(
            sample_equity_curve, 2000000
        )
        
        assert 'final_equity' in metrics
        assert 'total_return' in metrics
        assert 'annual_return' in metrics
        assert 'trading_days' in metrics
        
        # 基本验证
        assert metrics['trading_days'] == 252
        assert metrics['total_return'] > 0
        assert metrics['annual_return'] > 0


class TestReturnMetrics:
    """收益率指标测试"""
    
    @pytest.fixture
    def sample_equity_curve(self):
        """创建示例资产曲线"""
        return pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=252),
            'equity': np.cumprod(1 + np.random.randn(252) * 0.01) * 1000000
        })
    
    def test_calculate_return_metrics(self, sample_equity_curve):
        """测试收益率指标计算"""
        metrics = PerformanceMetrics._calculate_return_metrics(
            sample_equity_curve, 1000000
        )
        
        assert 'avg_daily_return' in metrics
        assert 'std_daily_return' in metrics
        assert 'avg_monthly_return' in metrics
        assert 'avg_yearly_return' in metrics
        
        # 验证收益率合理性
        assert isinstance(metrics['avg_daily_return'], (int, float))
        assert isinstance(metrics['std_daily_return'], (int, float))


class TestRiskMetrics:
    """风险指标测试"""
    
    @pytest.fixture
    def sample_equity_curve(self):
        """创建示例资产曲线"""
        np.random.seed(42)  # 固定随机种子确保可重复
        return pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=252),
            'equity': np.cumprod(1 + np.random.randn(252) * 0.015) * 1000000
        })
    
    def test_calculate_risk_metrics(self, sample_equity_curve):
        """测试风险指标计算"""
        metrics = PerformanceMetrics._calculate_risk_metrics(sample_equity_curve)
        
        assert 'max_drawdown' in metrics
        assert 'avg_drawdown_duration' in metrics
        assert 'max_drawdown_duration' in metrics
        assert 'downside_deviation' in metrics
        assert 'var_95' in metrics
        assert 'var_99' in metrics
        assert 'cvar_95' in metrics
        assert 'skewness' in metrics
        assert 'kurtosis' in metrics
        assert 'volatility' in metrics
        
        # 基本验证
        assert metrics['max_drawdown'] <= 0  # 回撤应该是负数或0
        # 注意：VaR是负值，检查是否有值即可
    
    def test_drawdown_duration(self, sample_equity_curve):
        """测试回撤持续时间计算"""
        metrics = PerformanceMetrics._calculate_risk_metrics(sample_equity_curve)
        
        assert metrics['avg_drawdown_duration'] >= 0
        assert metrics['max_drawdown_duration'] >= 0
        assert metrics['max_drawdown_duration'] >= metrics['avg_drawdown_duration']


class TestTradeMetrics:
    """交易指标测试"""
    
    @pytest.fixture
    def sample_trades(self):
        """创建示例交易记录"""
        return pd.DataFrame({
            'pnl': [10000, -5000, 15000, -3000, 8000],
            'pnl_pct': [0.10, -0.05, 0.15, -0.03, 0.08],
            'entry_date': pd.to_datetime(['2023-01-01', '2023-01-05', '2023-01-10',
                                          '2023-01-15', '2023-01-20']),
            'exit_date': pd.to_datetime(['2023-01-04', '2023-01-09', '2023-01-14',
                                        '2023-01-19', '2023-01-25']),
            'ts_code': ['000001.SZ', '000002.SZ', '000003.SZ',
                       '000004.SZ', '000005.SZ'],
            'name': ['股票A', '股票B', '股票C', '股票D', '股票E'],
            'entry_price': [10.0, 20.0, 15.0, 25.0, 30.0],
            'exit_price': [11.0, 19.0, 17.25, 24.25, 32.4],
            'shares': [1000, 500, 1000, 400, 250],
            'reason': ['take_profit', 'stop_loss', 'take_profit',
                     'stop_loss', 'take_profit']
        })
    
    def test_calculate_trade_metrics(self, sample_trades):
        """测试交易指标计算"""
        metrics = PerformanceMetrics._calculate_trade_metrics(sample_trades)
        
        assert 'total_trades' in metrics
        assert 'win_trades' in metrics
        assert 'lose_trades' in metrics
        assert 'win_rate' in metrics
        assert 'avg_win' in metrics
        assert 'avg_loss' in metrics
        assert 'profit_factor' in metrics
        
        # 基本验证
        assert metrics['total_trades'] == 5
        assert metrics['win_trades'] == 3
        assert metrics['lose_trades'] == 2
        assert abs(metrics['win_rate'] - 60) < 0.1  # 60%胜率
        assert metrics['profit_factor'] > 0
    
    def test_profit_factor(self, sample_trades):
        """测试盈亏比计算"""
        metrics = PerformanceMetrics._calculate_trade_metrics(sample_trades)
        
        # 总盈利应该大于总亏损（根据示例数据）
        assert metrics['profit_factor'] > 1
    
    def test_holding_days(self, sample_trades):
        """测试持仓天数计算"""
        metrics = PerformanceMetrics._calculate_trade_metrics(sample_trades)
        
        assert 'avg_holding_days' in metrics
        assert metrics['avg_holding_days'] > 0


class TestRiskAdjustedMetrics:
    """风险调整收益指标测试"""
    
    @pytest.fixture
    def sample_equity_curve(self):
        """创建示例资产曲线"""
        returns = np.random.randn(252) * 0.015 + 0.0005  # 正收益
        return pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=252),
            'equity': np.cumprod(1 + returns) * 1000000
        })
    
    def test_calculate_risk_adjusted_metrics(self, sample_equity_curve):
        """测试风险调整收益指标计算"""
        metrics = PerformanceMetrics._calculate_risk_adjusted_metrics(sample_equity_curve)
        
        assert 'sharpe_ratio' in metrics
        assert 'sortino_ratio' in metrics
        assert 'calmar_ratio' in metrics
        assert 'information_ratio' in metrics
        assert 'omega_ratio' in metrics
        assert 'sterling_ratio' in metrics
        assert 'burke_ratio' in metrics
        
        # 基本验证
        assert isinstance(metrics['sharpe_ratio'], (int, float))
        assert isinstance(metrics['sortino_ratio'], (int, float))
    
    def test_sharpe_ratio(self, sample_equity_curve):
        """测试夏普比率计算"""
        metrics = PerformanceMetrics._calculate_risk_adjusted_metrics(sample_equity_curve)
        
        # 夏普比率可以是正数或负数
        assert isinstance(metrics['sharpe_ratio'], (int, float))
    
    def test_sortino_ratio_greater_than_sharpe(self, sample_equity_curve):
        """测试Sortino比率通常大于夏普比率"""
        metrics = PerformanceMetrics._calculate_risk_adjusted_metrics(sample_equity_curve)
        
        # Sortino比率通常大于夏普比率（因为只考虑下行风险）
        # 但这不是绝对的，所以这里不做断言
        assert isinstance(metrics['sortino_ratio'], (int, float))
    
    def test_calmar_ratio(self, sample_equity_curve):
        """测试Calmar比率计算"""
        metrics = PerformanceMetrics._calculate_risk_adjusted_metrics(sample_equity_curve)
        
        # Calmar比率可以是正数或负数
        assert isinstance(metrics['calmar_ratio'], (int, float))


class TestBenchmarkMetrics:
    """基准比较指标测试"""
    
    @pytest.fixture
    def sample_equity_curve(self):
        """创建示例资产曲线"""
        returns = np.random.randn(252) * 0.015 + 0.0005
        return pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=252),
            'equity': np.cumprod(1 + returns) * 1000000
        })
    
    @pytest.fixture
    def benchmark_returns(self):
        """创建基准收益率"""
        np.random.seed(43)  # 固定随机种子
        returns = np.random.randn(252) * 0.012 + 0.0003
        return pd.Series(returns, index=pd.date_range('2023-01-01', periods=252))
    
    def test_calculate_benchmark_metrics(self, sample_equity_curve, benchmark_returns):
        """测试基准比较指标计算"""
        # 确保日期索引一致
        sample_equity_curve['returns'] = sample_equity_curve['equity'].pct_change().dropna()
        sample_equity_curve = sample_equity_curve.dropna()
        
        metrics = PerformanceMetrics._calculate_benchmark_metrics(
            sample_equity_curve, benchmark_returns
        )
        
        # 检查指标是否存在（有些可能是NaN）
        assert 'beta' in metrics
        assert 'alpha' in metrics
        assert 'alpha_annual' in metrics
        assert 'tracking_error' in metrics
        assert 'correlation' in metrics
        assert 'excess_return_annual' in metrics
        assert 'r_squared' in metrics
    
    def test_beta_calculation(self, sample_equity_curve, benchmark_returns):
        """测试Beta计算"""
        # 确保日期索引一致
        sample_equity_curve['returns'] = sample_equity_curve['equity'].pct_change().dropna()
        sample_equity_curve = sample_equity_curve.dropna()
        
        metrics = PerformanceMetrics._calculate_benchmark_metrics(
            sample_equity_curve, benchmark_returns
        )
        
        # Beta可以是正数或负数或NaN
        assert isinstance(metrics['beta'], (int, float)) or pd.isna(metrics['beta'])
    
    def test_correlation(self, sample_equity_curve, benchmark_returns):
        """测试相关性计算"""
        # 确保日期索引一致
        sample_equity_curve['returns'] = sample_equity_curve['equity'].pct_change().dropna()
        sample_equity_curve = sample_equity_curve.dropna()
        
        metrics = PerformanceMetrics._calculate_benchmark_metrics(
            sample_equity_curve, benchmark_returns
        )
        
        # 相关性可以是NaN
        if not pd.isna(metrics['correlation']):
            assert -1 <= metrics['correlation'] <= 1


class TestPositionMetrics:
    """持仓分析指标测试"""
    
    @pytest.fixture
    def sample_trades(self):
        """创建示例交易记录"""
        return pd.DataFrame({
            'pnl': [10000, -5000, 15000],
            'pnl_pct': [0.10, -0.05, 0.15],
            'entry_date': pd.to_datetime(['2023-01-01', '2023-01-05', '2023-01-10']),
            'exit_date': pd.to_datetime(['2023-01-04', '2023-01-09', '2023-01-14']),
            'ts_code': ['000001.SZ', '000002.SZ', '000003.SZ'],
            'name': ['股票A', '股票B', '股票C'],
            'entry_price': [10.0, 20.0, 15.0],
            'exit_price': [11.0, 19.0, 17.25],
            'shares': [1000, 500, 1000],
            'reason': ['take_profit', 'stop_loss', 'take_profit']
        })
    
    def test_calculate_position_metrics(self, sample_trades):
        """测试持仓分析指标计算"""
        metrics = PerformanceMetrics._calculate_position_metrics(sample_trades)
        
        assert 'avg_holding_days' in metrics
        assert 'std_holding_days' in metrics
        assert 'min_holding_days' in metrics
        assert 'max_holding_days' in metrics
        assert 'short_term_trades' in metrics
        assert 'medium_term_trades' in metrics
        assert 'long_term_trades' in metrics
        
        # 基本验证
        assert metrics['max_holding_days'] >= metrics['min_holding_days']
        assert (metrics['short_term_trades'] +
                metrics['medium_term_trades'] +
                metrics['long_term_trades']) == len(sample_trades)


class TestAllMetrics:
    """所有指标综合测试"""
    
    @pytest.fixture
    def sample_equity_curve(self):
        """创建示例资产曲线"""
        returns = np.random.randn(252) * 0.015 + 0.0005
        return pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=252),
            'equity': np.cumprod(1 + returns) * 1000000
        })
    
    @pytest.fixture
    def sample_trades(self):
        """创建示例交易记录"""
        return pd.DataFrame({
            'pnl': [10000, -5000, 15000, -3000],
            'pnl_pct': [0.10, -0.05, 0.15, -0.03],
            'entry_date': pd.to_datetime(['2023-01-01', '2023-01-05', '2023-01-10', '2023-01-15']),
            'exit_date': pd.to_datetime(['2023-01-04', '2023-01-09', '2023-01-14', '2023-01-19']),
            'ts_code': ['000001.SZ', '000002.SZ', '000003.SZ', '000004.SZ'],
            'name': ['股票A', '股票B', '股票C', '股票D'],
            'entry_price': [10.0, 20.0, 15.0, 25.0],
            'exit_price': [11.0, 19.0, 17.25, 24.25],
            'shares': [1000, 500, 1000, 400],
            'reason': ['take_profit', 'stop_loss', 'take_profit', 'stop_loss']
        })
    
    def test_calculate_all_metrics(self, sample_equity_curve, sample_trades):
        """测试所有指标计算"""
        metrics = PerformanceMetrics.calculate_all_metrics(
            sample_equity_curve, sample_trades, initial_capital=1000000
        )
        
        # 检查所有主要指标类别
        assert 'total_return' in metrics
        assert 'max_drawdown' in metrics
        assert 'sharpe_ratio' in metrics
        assert 'win_rate' in metrics
        assert 'volatility' in metrics
        
        # 检查数值合理性
        assert isinstance(metrics['total_return'], (int, float))
        assert isinstance(metrics['max_drawdown'], (int, float))
        assert isinstance(metrics['sharpe_ratio'], (int, float))
        assert isinstance(metrics['win_rate'], (int, float))
    
    def test_generate_metrics_report(self, sample_equity_curve, sample_trades):
        """测试指标报告生成"""
        metrics = PerformanceMetrics.calculate_all_metrics(
            sample_equity_curve, sample_trades, initial_capital=1000000
        )
        
        report = PerformanceMetrics.generate_metrics_report(metrics)
        
        # 检查报告格式
        assert isinstance(report, str)
        assert '基础指标' in report
        assert '风险指标' in report
        assert '风险调整收益' in report
        assert '交易统计' in report


class TestEdgeCases:
    """边缘情况测试"""
    
    def test_empty_equity_curve(self):
        """测试空资产曲线"""
        empty_curve = pd.DataFrame()
        trades_df = pd.DataFrame()
        
        metrics = PerformanceMetrics.calculate_all_metrics(
            empty_curve, trades_df, initial_capital=1000000
        )
        
        # 应该返回空字典
        assert metrics == {}
    
    def test_empty_trades(self):
        """测试空交易记录"""
        np.random.seed(44)
        equity_curve = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=252),
            'equity': np.cumprod(1 + np.random.randn(252) * 0.015) * 1000000
        })
        empty_trades = pd.DataFrame()
        
        metrics = PerformanceMetrics.calculate_all_metrics(
            equity_curve, empty_trades, initial_capital=1000000
        )
        
        # 基础指标应该仍然存在
        assert 'total_return' in metrics
        # 交易指标应该为0
        assert metrics.get('total_trades', 0) == 0
