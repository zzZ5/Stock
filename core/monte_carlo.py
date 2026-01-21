"""
蒙特卡洛模拟模块
用于回测结果的稳健性分析和压力测试
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import stats
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from tqdm import tqdm


class MonteCarloSimulator:
    """蒙特卡洛模拟器"""
    
    def __init__(
        self,
        equity_curve: pd.DataFrame,
        trades_df: pd.DataFrame,
        initial_capital: float,
        n_simulations: int = 1000,
        confidence_level: float = 0.95
    ):
        """
        初始化蒙特卡洛模拟器
        
        参数:
            equity_curve: 资产曲线DataFrame
            trades_df: 交易记录DataFrame
            initial_capital: 初始资金
            n_simulations: 模拟次数
            confidence_level: 置信水平
        """
        self.equity_curve = equity_curve
        self.trades_df = trades_df
        self.initial_capital = initial_capital
        self.n_simulations = n_simulations
        self.confidence_level = confidence_level
        
        # 计算每日收益率
        self.daily_returns = self.equity_curve['equity'].pct_change().dropna()
        
        # 计算基本统计量
        self.mean_return = self.daily_returns.mean()
        self.std_return = self.daily_returns.std()
        self.skewness = stats.skew(self.daily_returns)
        self.kurtosis = stats.kurtosis(self.daily_returns)
        
    def geometric_brownian_motion(
        self,
        n_days: int,
        drift: Optional[float] = None,
        volatility: Optional[float] = None
    ) -> np.ndarray:
        """
        几何布朗运动模拟
        
        参数:
            n_days: 模拟天数
            drift: 漂移率（默认使用历史均值）
            volatility: 波动率（默认使用历史标准差）
        
        返回:
            模拟的收益率序列
        """
        drift = drift or self.mean_return
        volatility = volatility or self.std_return
        
        # 生成随机冲击
        shocks = np.random.normal(0, 1, n_days)
        
        # 计算收益率路径
        returns = drift + volatility * shocks
        
        return returns
    
    def bootstrap_returns(self, n_days: int) -> np.ndarray:
        """
        自助法重采样收益率
        
        参数:
            n_days: 模拟天数
        
        返回:
            重采样的收益率序列
        """
        # 随机重采样历史收益率
        returns = np.random.choice(self.daily_returns.values, size=n_days, replace=True)
        
        return returns
    
    def run_simulation(
        self,
        method: str = 'gbm',
        n_days: Optional[int] = None,
        drift: Optional[float] = None,
        volatility: Optional[float] = None,
        preserve_correlation: bool = False
    ) -> Dict:
        """
        运行单次模拟
        
        参数:
            method: 模拟方法 ('gbm', 'bootstrap', 'skew_normal')
            n_days: 模拟天数
            drift: 漂移率
            volatility: 波动率
            preserve_correlation: 是否保持收益率相关性
        
        返回:
            模拟结果字典
        """
        n_days = n_days or len(self.daily_returns)
        
        # 生成收益率序列
        if method == 'gbm':
            returns = self.geometric_brownian_motion(n_days, drift, volatility)
        
        elif method == 'bootstrap':
            returns = self.bootstrap_returns(n_days)
        
        elif method == 'skew_normal':
            # 考虑偏度和峰度的正态分布
            a, loc, scale = stats.skewnorm.fit(self.daily_returns.values)
            returns = stats.skewnorm.rvs(a, loc=loc, scale=scale, size=n_days)
        
        else:
            returns = self.geometric_brownian_motion(n_days, drift, volatility)
        
        # 计算资产曲线
        cumulative_returns = (1 + returns).cumprod()
        equity = self.initial_capital * cumulative_returns
        equity_series = pd.Series(equity)
        
        # 计算回撤
        running_max = equity_series.expanding().max()
        drawdown = (equity_series - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # 计算其他指标
        final_equity = equity_series.iloc[-1]
        total_return = (final_equity - self.initial_capital) / self.initial_capital
        annual_return = (1 + total_return) ** (252 / n_days) - 1 if n_days > 0 else 0
        annual_volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = (annual_return - 0.03) / annual_volatility if annual_volatility > 0 else 0
        
        return {
            'returns': returns,
            'equity': equity,
            'drawdown': drawdown,
            'final_equity': final_equity,
            'total_return': total_return * 100,
            'annual_return': annual_return * 100,
            'max_drawdown': max_drawdown * 100,
            'volatility': annual_volatility * 100,
            'sharpe_ratio': sharpe_ratio
        }
    
    def run_batch_simulation(
        self,
        n_simulations: Optional[int] = None,
        method: str = 'gbm',
        n_workers: Optional[int] = None,
        show_progress: bool = True
    ) -> pd.DataFrame:
        """
        批量运行蒙特卡洛模拟
        
        参数:
            n_simulations: 模拟次数
            method: 模拟方法
            n_workers: 并行工作数
            show_progress: 是否显示进度
        
        返回:
            包含所有模拟结果的DataFrame
        """
        n_simulations = n_simulations or self.n_simulations
        n_workers = n_workers or min(4, mp.cpu_count())
        
        results = []
        
        if n_workers > 1:
            # 并行模拟 - 批量提交减少开销
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                # 使用 map 提交，减少通信开销
                futures = [
                    executor.submit(self.run_simulation, method=method, n_days=len(self.daily_returns))
                    for _ in range(n_simulations)
                ]
                
                iterator = tqdm(as_completed(futures), total=n_simulations, disable=not show_progress)
                for future in iterator:
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        logger.warning(f"模拟失败: {e}")
                        continue
        else:
            # 串行模拟
            for i in tqdm(range(n_simulations), disable=not show_progress):
                result = self.run_simulation(method=method)
                results.append(result)
        
        # 转换为DataFrame
        df = pd.DataFrame(results)
        
        return df
    
    def calculate_risk_metrics(self, simulation_results: pd.DataFrame) -> Dict:
        """
        计算风险指标
        
        参数:
            simulation_results: 模拟结果DataFrame
        
        返回:
            风险指标字典
        """
        # VaR和CVaR（在置信水平下）
        alpha = 1 - self.confidence_level
        var = np.percentile(simulation_results['total_return'], alpha * 100)
        cvar = simulation_results[simulation_results['total_return'] <= var]['total_return'].mean()
        
        # 收益率分布统计
        return_mean = simulation_results['total_return'].mean()
        return_std = simulation_results['total_return'].std()
        return_median = simulation_results['total_return'].median()
        return_5th = np.percentile(simulation_results['total_return'], 5)
        return_95th = np.percentile(simulation_results['total_return'], 95)
        
        # 回撤统计
        max_drawdown_mean = simulation_results['max_drawdown'].mean()
        max_drawdown_worst = simulation_results['max_drawdown'].min()
        max_drawdown_5th = np.percentile(simulation_results['max_drawdown'], 5)
        max_drawdown_95th = np.percentile(simulation_results['max_drawdown'], 95)
        
        # 夏普比率统计
        sharpe_mean = simulation_results['sharpe_ratio'].mean()
        sharpe_std = simulation_results['sharpe_ratio'].std()
        
        # 概率统计
        prob_positive = (simulation_results['total_return'] > 0).mean()
        prob_positive_10pct = (simulation_results['total_return'] > 10).mean()
        prob_negative = (simulation_results['total_return'] < 0).mean()
        prob_loss_20pct = (simulation_results['total_return'] < -20).mean()
        
        return {
            'var_95': var,
            'cvar_95': cvar,
            'return_mean': return_mean,
            'return_std': return_std,
            'return_median': return_median,
            'return_5th_percentile': return_5th,
            'return_95th_percentile': return_95th,
            'max_drawdown_mean': max_drawdown_mean,
            'max_drawdown_worst': max_drawdown_worst,
            'max_drawdown_5th_percentile': max_drawdown_5th,
            'max_drawdown_95th_percentile': max_drawdown_95th,
            'sharpe_mean': sharpe_mean,
            'sharpe_std': sharpe_std,
            'prob_positive': prob_positive * 100,
            'prob_positive_10pct': prob_positive_10pct * 100,
            'prob_negative': prob_negative * 100,
            'prob_loss_20pct': prob_loss_20pct * 100
        }
    
    def print_risk_report(self, simulation_results: pd.DataFrame):
        """
        打印风险报告
        
        参数:
            simulation_results: 模拟结果DataFrame
        """
        risk_metrics = self.calculate_risk_metrics(simulation_results)
        
        print(f"\n{'='*70}")
        print(f"蒙特卡洛模拟风险分析报告")
        print(f"{'='*70}\n")
        
        print(f"模拟次数: {len(simulation_results)}")
        print(f"置信水平: {self.confidence_level * 100}%\n")
        
        print(f"--- 收益率分布 ---")
        print(f"平均收益率:     {risk_metrics['return_mean']:>10.2f}%")
        print(f"收益率中位数:   {risk_metrics['return_median']:>10.2f}%")
        print(f"收益率标准差:   {risk_metrics['return_std']:>10.2f}%")
        print(f"5%分位数:       {risk_metrics['return_5th_percentile']:>10.2f}%")
        print(f"95%分位数:      {risk_metrics['return_95th_percentile']:>10.2f}%\n")
        
        print(f"--- 风险指标 ---")
        print(f"VaR (95%):      {risk_metrics['var_95']:>10.2f}%")
        print(f"CVaR (95%):     {risk_metrics['cvar_95']:>10.2f}%\n")
        
        print(f"--- 回撤分析 ---")
        print(f"平均最大回撤:   {risk_metrics['max_drawdown_mean']:>10.2f}%")
        print(f"最差最大回撤:   {risk_metrics['max_drawdown_worst']:>10.2f}%")
        print(f"5%分位数回撤:   {risk_metrics['max_drawdown_5th_percentile']:>10.2f}%")
        print(f"95%分位数回撤:  {risk_metrics['max_drawdown_95th_percentile']:>10.2f}%\n")
        
        print(f"--- 夏普比率 ---")
        print(f"平均夏普比率:   {risk_metrics['sharpe_mean']:>10.2f}")
        print(f"夏普比率标准差: {risk_metrics['sharpe_std']:>10.2f}\n")
        
        print(f"--- 概率统计 ---")
        print(f"盈利概率:       {risk_metrics['prob_positive']:>10.2f}%")
        print(f"盈利>10%概率:   {risk_metrics['prob_positive_10pct']:>10.2f}%")
        print(f"亏损概率:       {risk_metrics['prob_negative']:>10.2f}%")
        print(f"亏损>20%概率:   {risk_metrics['prob_loss_20pct']:>10.2f}%\n")
        
        print(f"{'='*70}\n")


class StressTester:
    """压力测试器"""
    
    def __init__(self, backtest_results: Dict):
        """
        初始化压力测试器
        
        参数:
            backtest_results: 回测结果字典
        """
        self.backtest_results = backtest_results
    
    def test_market_crash(
        self,
        crash_dates: List[str],
        crash_magnitude: float = -0.20
    ) -> Dict:
        """
        市场暴跌压力测试
        
        参数:
            crash_dates: 假设的暴跌日期列表
            crash_magnitude: 暴跌幅度（-20%表示下跌20%）
        
        返回:
            压力测试结果
        """
        original_equity = self.backtest_results['equity_curve']
        
        # 复制资产曲线
        stressed_equity = original_equity.copy()
        
        # 在指定日期应用暴跌
        for date in crash_dates:
            idx = stressed_equity[stressed_equity['date'] == date].index
            if len(idx) > 0:
                idx = idx[0]
                # 应用暴跌
                stressed_equity.loc[idx, 'equity'] *= (1 + crash_magnitude)
                # 后续所有日期也受影响
                stressed_equity.loc[idx:, 'equity'] *= (1 + crash_magnitude)
        
        # 计算新的指标
        final_equity = stressed_equity['equity'].iloc[-1]
        total_return = (final_equity - self.backtest_results['final_equity']) / self.backtest_results['final_equity']
        
        # 计算回撤
        equity_values = stressed_equity['equity'].values
        cumulative_returns = equity_values / self.backtest_results['final_equity']
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        return {
            'scenario': 'market_crash',
            'crash_dates': crash_dates,
            'crash_magnitude': crash_magnitude * 100,
            'original_final_equity': self.backtest_results['final_equity'],
            'stressed_final_equity': final_equity,
            'stressed_total_return': total_return * 100,
            'stressed_max_drawdown': max_drawdown * 100,
            'impact_on_return': (total_return * 100 - self.backtest_results['total_return']),
            'impact_on_drawdown': (max_drawdown * 100 - self.backtest_results['max_drawdown'])
        }
    
    def test_volatility_spike(
        self,
        volatility_multiplier: float = 2.0,
        duration: int = 20
    ) -> Dict:
        """
        波动率飙升压力测试
        
        参数:
            volatility_multiplier: 波动率倍数
            duration: 波动率飙升持续天数
        
        返回:
            压力测试结果
        """
        original_equity = self.backtest_results['equity_curve']
        
        # 复制资产曲线
        stressed_equity = original_equity.copy()
        
        # 计算原始收益率
        equity_values = stressed_equity['equity'].values
        original_returns = np.diff(equity_values) / equity_values[:-1]
        
        # 选择波动率飙升区间（随机或指定）
        start_idx = np.random.randint(0, len(original_returns) - duration)
        
        # 在指定区间放大波动
        stressed_returns = original_returns.copy()
        stressed_returns[start_idx:start_idx + duration] *= volatility_multiplier
        
        # 重新计算资产曲线
        stressed_equity_values = np.zeros_like(equity_values)
        stressed_equity_values[0] = equity_values[0]
        
        for i in range(1, len(stressed_equity_values)):
            stressed_equity_values[i] = stressed_equity_values[i-1] * (1 + stressed_returns[i-1])
        
        stressed_equity['equity'] = stressed_equity_values
        
        # 计算新指标
        final_equity = stressed_equity_values[-1]
        total_return = (final_equity - self.backtest_results['final_equity']) / self.backtest_results['final_equity']
        
        # 计算回撤
        cumulative_returns = stressed_equity_values / self.backtest_results['final_equity']
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        return {
            'scenario': 'volatility_spike',
            'volatility_multiplier': volatility_multiplier,
            'duration': duration,
            'original_final_equity': self.backtest_results['final_equity'],
            'stressed_final_equity': final_equity,
            'stressed_total_return': total_return * 100,
            'stressed_max_drawdown': max_drawdown * 100,
            'impact_on_return': (total_return * 100 - self.backtest_results['total_return']),
            'impact_on_drawdown': (max_drawdown * 100 - self.backtest_results['max_drawdown'])
        }
    
    def test_correlation_crisis(
        self,
        correlation_increase: float = 0.5,
        duration: int = 30
    ) -> Dict:
        """
        相关性危机压力测试（所有资产同时下跌）
        
        参数:
            correlation_increase: 相关性增加幅度
            duration: 持续天数
        
        返回:
            压力测试结果
        """
        # 简化处理：假设持仓股票同时下跌
        decline_pct = -0.1 * (1 + correlation_increase)  # 10%下跌，根据相关性调整
        
        original_equity = self.backtest_results['equity_curve']
        
        # 复制资产曲线
        stressed_equity = original_equity.copy()
        
        # 选择危机区间（随机）
        n_days = len(stressed_equity)
        start_idx = np.random.randint(0, n_days - duration)
        
        # 在指定区间应用下跌
        stressed_equity.loc[start_idx:start_idx + duration, 'equity'] *= (1 + decline_pct)
        
        # 重新计算后续资产曲线
        for i in range(start_idx + duration, n_days):
            stressed_equity.loc[i, 'equity'] = stressed_equity.loc[i, 'equity'] * (1 + decline_pct)
        
        # 计算新指标
        final_equity = stressed_equity['equity'].iloc[-1]
        total_return = (final_equity - self.backtest_results['final_equity']) / self.backtest_results['final_equity']
        
        # 计算回撤
        equity_values = stressed_equity['equity'].values
        cumulative_returns = equity_values / self.backtest_results['final_equity']
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        return {
            'scenario': 'correlation_crisis',
            'correlation_increase': correlation_increase,
            'decline_pct': decline_pct * 100,
            'duration': duration,
            'original_final_equity': self.backtest_results['final_equity'],
            'stressed_final_equity': final_equity,
            'stressed_total_return': total_return * 100,
            'stressed_max_drawdown': max_drawdown * 100,
            'impact_on_return': (total_return * 100 - self.backtest_results['total_return']),
            'impact_on_drawdown': (max_drawdown * 100 - self.backtest_results['max_drawdown'])
        }
    
    def run_all_stress_tests(self, n_crashes: int = 3) -> pd.DataFrame:
        """
        运行所有压力测试
        
        参数:
            n_crashes: 市场暴跌测试次数
        
        返回:
            包含所有压力测试结果的DataFrame
        """
        results = []
        
        # 市场暴跌测试
        for i in range(n_crashes):
            crash_dates = ['20230115', '20230220', '20230315'][i:i+1]
            result = self.test_market_crash(crash_dates)
            results.append(result)
        
        # 波动率飙升测试
        for multiplier in [1.5, 2.0, 3.0]:
            result = self.test_volatility_spike(volatility_multiplier=multiplier)
            results.append(result)
        
        # 相关性危机测试
        for correlation in [0.3, 0.5, 0.7]:
            result = self.test_correlation_crisis(correlation_increase=correlation)
            results.append(result)
        
        return pd.DataFrame(results)
    
    def print_stress_test_report(self, stress_results: pd.DataFrame):
        """
        打印压力测试报告
        
        参数:
            stress_results: 压力测试结果DataFrame
        """
        print(f"\n{'='*70}")
        print(f"压力测试报告")
        print(f"{'='*70}\n")
        
        for idx, row in stress_results.iterrows():
            print(f"场景 {idx + 1}: {row['scenario']}")
            print(f"  参数: ", end="")
            
            if row['scenario'] == 'market_crash':
                print(f"暴跌幅度 {row['crash_magnitude']:.1f}%")
            elif row['scenario'] == 'volatility_spike':
                print(f"波动率倍数 {row['volatility_multiplier']:.1f}x")
            elif row['scenario'] == 'correlation_crisis':
                print(f"相关性增加 {row['correlation_increase']:.1f}")
            
            print(f"  原始期末资金: {row['original_final_equity']:,.0f} 元")
            print(f"  压力期末资金: {row['stressed_final_equity']:,.0f} 元")
            print(f"  压力收益率:   {row['stressed_total_return']:>10.2f}%")
            print(f"  压力最大回撤: {row['stressed_max_drawdown']:>10.2f}%")
            print(f"  收益影响:     {row['impact_on_return']:>10.2f}%")
            print(f"  回撤影响:     {row['impact_on_drawdown']:>10.2f}%")
            print()
        
        print(f"{'='*70}\n")
