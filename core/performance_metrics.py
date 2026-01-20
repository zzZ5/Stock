"""
高级绩效评估指标
包含更多维度的绩效评估指标
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from scipy import stats


class PerformanceMetrics:
    """高级绩效评估指标计算器"""
    
    @staticmethod
    def calculate_all_metrics(
        equity_curve: pd.DataFrame,
        trades_df: pd.DataFrame,
        benchmark_returns: Optional[pd.Series] = None,
        initial_capital: float = 1000000
    ) -> Dict:
        """
        计算所有高级绩效指标
        
        参数:
            equity_curve: 资产曲线DataFrame
            trades_df: 交易记录DataFrame
            benchmark_returns: 基准收益率序列（可选）
            initial_capital: 初始资金
        
        返回:
            包含所有指标的字典
        """
        if equity_curve.empty:
            return {}
        
        # 基础指标
        basic_metrics = PerformanceMetrics._calculate_basic_metrics(
            equity_curve, initial_capital
        )
        
        # 收益率相关指标
        return_metrics = PerformanceMetrics._calculate_return_metrics(
            equity_curve, initial_capital
        )
        
        # 风险指标
        risk_metrics = PerformanceMetrics._calculate_risk_metrics(equity_curve)
        
        # 交易统计
        trade_metrics = PerformanceMetrics._calculate_trade_metrics(trades_df)
        
        # 风险调整收益指标
        risk_adj_metrics = PerformanceMetrics._calculate_risk_adjusted_metrics(
            equity_curve
        )
        
        # 基准比较（如果有）
        benchmark_metrics = {}
        if benchmark_returns is not None:
            benchmark_metrics = PerformanceMetrics._calculate_benchmark_metrics(
                equity_curve, benchmark_returns
            )
        
        # 持仓分析
        position_metrics = PerformanceMetrics._calculate_position_metrics(trades_df)
        
        return {
            **basic_metrics,
            **return_metrics,
            **risk_metrics,
            **trade_metrics,
            **risk_adj_metrics,
            **benchmark_metrics,
            **position_metrics
        }
    
    @staticmethod
    def _calculate_basic_metrics(
        equity_curve: pd.DataFrame,
        initial_capital: float
    ) -> Dict:
        """计算基础指标"""
        final_equity = equity_curve['equity'].iloc[-1]
        total_return = (final_equity - initial_capital) / initial_capital
        
        # 交易天数
        trading_days = len(equity_curve)
        
        # 年化收益率
        annual_return = (1 + total_return) ** (252 / trading_days) - 1 if trading_days > 0 else 0
        
        return {
            'final_equity': final_equity,
            'total_return': total_return * 100,
            'annual_return': annual_return * 100,
            'trading_days': trading_days
        }
    
    @staticmethod
    def _calculate_return_metrics(
        equity_curve: pd.DataFrame,
        initial_capital: float
    ) -> Dict:
        """计算收益率相关指标"""
        daily_returns = equity_curve['equity'].pct_change().dropna()
        
        # 月度收益率
        equity_curve['month'] = pd.to_datetime(equity_curve['date']).dt.month
        equity_curve['year'] = pd.to_datetime(equity_curve['date']).dt.year
        
        monthly_returns = equity_curve.groupby(['year', 'month'])['equity'].apply(
            lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0]
        )
        
        # 年度收益率
        yearly_returns = equity_curve.groupby('year')['equity'].apply(
            lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0]
        )
        
        # 滚动收益率
        rolling_20d = daily_returns.rolling(20).sum()
        rolling_60d = daily_returns.rolling(60).sum()
        rolling_252d = daily_returns.rolling(252).sum()
        
        return {
            'avg_daily_return': daily_returns.mean() * 100,
            'std_daily_return': daily_returns.std() * 100,
            'avg_monthly_return': monthly_returns.mean() * 100,
            'avg_yearly_return': yearly_returns.mean() * 100,
            'best_monthly_return': monthly_returns.max() * 100,
            'worst_monthly_return': monthly_returns.min() * 100,
            'best_yearly_return': yearly_returns.max() * 100,
            'worst_yearly_return': yearly_returns.min() * 100,
            'avg_rolling_20d': rolling_20d.mean() * 100,
            'avg_rolling_60d': rolling_60d.mean() * 100,
            'avg_rolling_252d': rolling_252d.mean() * 100
        }
    
    @staticmethod
    def _calculate_risk_metrics(equity_curve: pd.DataFrame) -> Dict:
        """计算风险指标"""
        daily_returns = equity_curve['equity'].pct_change().dropna()
        
        # 最大回撤
        cumulative_returns = (1 + daily_returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # 回撤持续时间
        drawdown_periods = []
        in_drawdown = False
        drawdown_start = 0
        
        for i, dd in enumerate(drawdown):
            if dd < 0 and not in_drawdown:
                in_drawdown = True
                drawdown_start = i
            elif dd >= 0 and in_drawdown:
                drawdown_periods.append(i - drawdown_start)
                in_drawdown = False
        
        if in_drawdown:
            drawdown_periods.append(len(drawdown) - drawdown_start)
        
        avg_drawdown_duration = np.mean(drawdown_periods) if drawdown_periods else 0
        max_drawdown_duration = max(drawdown_periods) if drawdown_periods else 0
        
        # 下行风险
        downside_returns = daily_returns[daily_returns < 0]
        downside_std = downside_returns.std() if len(downside_returns) > 0 else 0
        downside_deviation = downside_std * np.sqrt(252) * 100
        
        # VaR和CVaR
        var_95 = np.percentile(daily_returns, 5) * 100
        var_99 = np.percentile(daily_returns, 1) * 100
        cvar_95 = daily_returns[daily_returns <= var_95/100].mean() * 100
        cvar_99 = daily_returns[daily_returns <= var_99/100].mean() * 100
        
        # 偏度和峰度
        skewness = stats.skew(daily_returns)
        kurtosis = stats.kurtosis(daily_returns)
        
        # 波动率
        volatility = daily_returns.std() * np.sqrt(252) * 100
        
        return {
            'max_drawdown': max_drawdown * 100,
            'avg_drawdown_duration': avg_drawdown_duration,
            'max_drawdown_duration': max_drawdown_duration,
            'downside_deviation': downside_deviation,
            'var_95': var_95,
            'var_99': var_99,
            'cvar_95': cvar_95,
            'cvar_99': cvar_99,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'volatility': volatility
        }
    
    @staticmethod
    def _calculate_trade_metrics(trades_df: pd.DataFrame) -> Dict:
        """计算交易统计指标"""
        if trades_df.empty:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'profit_factor': 0
            }
        
        # 盈亏交易
        win_trades = trades_df[trades_df['pnl'] > 0]
        lose_trades = trades_df[trades_df['pnl'] <= 0]
        
        # 胜率
        win_rate = len(win_trades) / len(trades_df) * 100
        
        # 平均收益
        avg_win = win_trades['pnl_pct'].mean() * 100 if len(win_trades) > 0 else 0
        avg_loss = lose_trades['pnl_pct'].mean() * 100 if len(lose_trades) > 0 else 0
        
        # 盈亏比
        total_win = win_trades['pnl'].sum() if len(win_trades) > 0 else 0
        total_lose = lose_trades['pnl'].sum() if len(lose_trades) > 0 else 0
        profit_factor = abs(total_win / total_lose) if total_lose != 0 else float('inf')
        
        # 最大单笔盈亏
        max_profit = trades_df['pnl_pct'].max() * 100
        max_loss = trades_df['pnl_pct'].min() * 100
        
        # 平均持仓天数
        trades_df['entry_date'] = pd.to_datetime(trades_df['entry_date'])
        trades_df['exit_date'] = pd.to_datetime(trades_df['exit_date'])
        trades_df['holding_days'] = (trades_df['exit_date'] - trades_df['entry_date']).dt.days
        avg_holding_days = trades_df['holding_days'].mean()
        
        # 盈利因子
        profit_factor_2 = abs(
            win_trades['pnl_pct'].sum() / lose_trades['pnl_pct'].sum()
        ) if lose_trades['pnl_pct'].sum() != 0 else float('inf')
        
        return {
            'total_trades': len(trades_df),
            'win_trades': len(win_trades),
            'lose_trades': len(lose_trades),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'profit_factor_pct': profit_factor_2,
            'max_profit': max_profit,
            'max_loss': max_loss,
            'avg_holding_days': avg_holding_days
        }
    
    @staticmethod
    def _calculate_risk_adjusted_metrics(equity_curve: pd.DataFrame) -> Dict:
        """计算风险调整收益指标"""
        daily_returns = equity_curve['equity'].pct_change().dropna()
        
        # 年化收益率
        trading_days = len(equity_curve)
        final_equity = equity_curve['equity'].iloc[-1]
        initial_equity = equity_curve['equity'].iloc[0]
        total_return = (final_equity - initial_equity) / initial_equity
        annual_return = (1 + total_return) ** (252 / trading_days) - 1 if trading_days > 0 else 0
        
        # 夏普比率（无风险利率3%）
        risk_free_rate = 0.03
        excess_returns = daily_returns - risk_free_rate / 252
        sharpe_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252) if excess_returns.std() > 0 else 0
        
        # Sortino比率（只考虑下行风险）
        downside_returns = daily_returns[daily_returns < 0]
        downside_std = downside_returns.std() if len(downside_returns) > 0 else 0
        sortino_ratio = (annual_return - risk_free_rate) / (downside_std * np.sqrt(252)) if downside_std > 0 else 0
        
        # Calmar比率（年化收益/最大回撤绝对值）
        cumulative_returns = (1 + daily_returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # 信息比率（假设基准收益为市场平均）
        # 如果没有基准，使用0
        excess_return = annual_return
        tracking_error = daily_returns.std() * np.sqrt(252)
        information_ratio = excess_return / tracking_error if tracking_error > 0 else 0
        
        # Omega比率
        threshold = risk_free_rate / 252
        excess_returns_above_threshold = daily_returns[daily_returns > threshold] - threshold
        excess_returns_below_threshold = threshold - daily_returns[daily_returns < threshold]
        omega_ratio = excess_returns_above_threshold.sum() / excess_returns_below_threshold.sum() if excess_returns_below_threshold.sum() != 0 else float('inf')
        
        # Sterling比率
        return_drawdowns = []
        for i in range(10, len(cumulative_returns)):
            window_return = cumulative_returns.iloc[i] / cumulative_returns.iloc[i-10] - 1
            window_dd = drawdown.iloc[i-10:i].min()
            if window_dd != 0:
                return_drawdowns.append(abs(window_return / window_dd))
        sterling_ratio = np.mean(return_drawdowns) if return_drawdowns else 0
        
        # Burke比率
        squared_drawdowns = drawdown ** 2
        burke_ratio = annual_return / np.sqrt(squared_drawdowns.sum()) if squared_drawdowns.sum() > 0 else 0
        
        return {
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'information_ratio': information_ratio,
            'omega_ratio': omega_ratio,
            'sterling_ratio': sterling_ratio,
            'burke_ratio': burke_ratio
        }
    
    @staticmethod
    def _calculate_benchmark_metrics(
        equity_curve: pd.DataFrame,
        benchmark_returns: pd.Series
    ) -> Dict:
        """计算基准比较指标"""
        daily_returns = equity_curve['equity'].pct_change().dropna()
        
        # 对齐索引
        common_index = daily_returns.index.intersection(benchmark_returns.index)
        strategy_returns = daily_returns.loc[common_index]
        benchmark_ret = benchmark_returns.loc[common_index]
        
        # Beta
        covariance = np.cov(strategy_returns, benchmark_ret)[0, 1]
        benchmark_variance = benchmark_ret.var()
        beta = covariance / benchmark_variance if benchmark_variance > 0 else 0
        
        # Alpha
        alpha = strategy_returns.mean() - beta * benchmark_ret.mean()
        alpha_annual = alpha * 252 * 100
        
        # 跟踪误差
        tracking_error = (strategy_returns - benchmark_ret).std() * np.sqrt(252) * 100
        
        # 相关性
        correlation = strategy_returns.corr(benchmark_ret)
        
        # 超额收益
        excess_returns = strategy_returns - benchmark_ret
        excess_return_cumulative = (1 + excess_returns).cumprod() - 1
        excess_return_annual = excess_returns.mean() * 252 * 100
        
        # 相对收益
        relative_return = strategy_returns.sum() - benchmark_ret.sum()
        
        # R-squared
        r_squared = correlation ** 2
        
        # Treynor比率
        treynor_ratio = (strategy_returns.mean() * 252 - 0.03) / beta if beta != 0 else 0
        
        # Modigliani比率（M2）
        market_volatility = benchmark_ret.std() * np.sqrt(252)
        strategy_volatility = strategy_returns.std() * np.sqrt(252)
        m_squared = (strategy_returns.mean() * 252 - 0.03) * (market_volatility / strategy_volatility) + 0.03 if strategy_volatility > 0 else 0
        
        return {
            'beta': beta,
            'alpha': alpha,
            'alpha_annual': alpha_annual,
            'tracking_error': tracking_error,
            'correlation': correlation,
            'excess_return_annual': excess_return_annual,
            'relative_return': relative_return * 100,
            'r_squared': r_squared,
            'treynor_ratio': treynor_ratio,
            'm_squared': m_squared
        }
    
    @staticmethod
    def _calculate_position_metrics(trades_df: pd.DataFrame) -> Dict:
        """计算持仓分析指标"""
        if trades_df.empty:
            return {}
        
        trades_df['entry_date'] = pd.to_datetime(trades_df['entry_date'])
        trades_df['exit_date'] = pd.to_datetime(trades_df['exit_date'])
        trades_df['holding_days'] = (trades_df['exit_date'] - trades_df['entry_date']).dt.days
        
        # 持仓天数分布
        avg_holding_days = trades_df['holding_days'].mean()
        std_holding_days = trades_df['holding_days'].std()
        min_holding_days = trades_df['holding_days'].min()
        max_holding_days = trades_df['holding_days'].max()
        
        # 按持仓天数分组统计
        short_term = len(trades_df[trades_df['holding_days'] <= 5])
        medium_term = len(trades_df[(trades_df['holding_days'] > 5) & (trades_df['holding_days'] <= 20)])
        long_term = len(trades_df[trades_df['holding_days'] > 20])
        
        # 持仓天数与收益的关系
        short_term_return = trades_df[trades_df['holding_days'] <= 5]['pnl_pct'].mean() * 100 if short_term > 0 else 0
        medium_term_return = trades_df[(trades_df['holding_days'] > 5) & (trades_df['holding_days'] <= 20)]['pnl_pct'].mean() * 100 if medium_term > 0 else 0
        long_term_return = trades_df[trades_df['holding_days'] > 20]['pnl_pct'].mean() * 100 if long_term > 0 else 0
        
        return {
            'avg_holding_days': avg_holding_days,
            'std_holding_days': std_holding_days,
            'min_holding_days': min_holding_days,
            'max_holding_days': max_holding_days,
            'short_term_trades': short_term,
            'medium_term_trades': medium_term,
            'long_term_trades': long_term,
            'short_term_return': short_term_return,
            'medium_term_return': medium_term_return,
            'long_term_return': long_term_return
        }
    
    @staticmethod
    def generate_metrics_report(metrics: Dict) -> str:
        """
        生成指标报告
        
        参数:
            metrics: 指标字典
        
        返回:
            格式化的报告字符串
        """
        report = []
        report.append("=" * 70)
        report.append("高级绩效评估报告")
        report.append("=" * 70)
        report.append("")
        
        # 基础指标
        report.append("--- 基础指标 ---")
        report.append(f"总收益率:     {metrics.get('total_return', 0):>10.2f}%")
        report.append(f"年化收益率:   {metrics.get('annual_return', 0):>10.2f}%")
        report.append(f"交易天数:     {metrics.get('trading_days', 0):>10}")
        report.append("")
        
        # 风险指标
        report.append("--- 风险指标 ---")
        report.append(f"最大回撤:     {metrics.get('max_drawdown', 0):>10.2f}%")
        report.append(f"平均回撤持续: {metrics.get('avg_drawdown_duration', 0):>10.1f}天")
        report.append(f"最大回撤持续: {metrics.get('max_drawdown_duration', 0):>10.1f}天")
        report.append(f"年化波动率:   {metrics.get('volatility', 0):>10.2f}%")
        report.append(f"下行偏差:     {metrics.get('downside_deviation', 0):>10.2f}%")
        report.append(f"VaR (95%):    {metrics.get('var_95', 0):>10.2f}%")
        report.append(f"CVaR (95%):   {metrics.get('cvar_95', 0):>10.2f}%")
        report.append("")
        
        # 风险调整收益
        report.append("--- 风险调整收益 ---")
        report.append(f"夏普比率:     {metrics.get('sharpe_ratio', 0):>10.2f}")
        report.append(f"Sortino比率:  {metrics.get('sortino_ratio', 0):>10.2f}")
        report.append(f"Calmar比率:   {metrics.get('calmar_ratio', 0):>10.2f}")
        report.append(f"信息比率:     {metrics.get('information_ratio', 0):>10.2f}")
        report.append(f"Omega比率:    {metrics.get('omega_ratio', 0):>10.2f}")
        report.append("")
        
        # 交易统计
        report.append("--- 交易统计 ---")
        report.append(f"总交易次数:   {metrics.get('total_trades', 0):>10}")
        report.append(f"盈利交易:     {metrics.get('win_trades', 0):>10}")
        report.append(f"亏损交易:     {metrics.get('lose_trades', 0):>10}")
        report.append(f"胜率:         {metrics.get('win_rate', 0):>10.2f}%")
        report.append(f"平均盈利:     {metrics.get('avg_win', 0):>10.2f}%")
        report.append(f"平均亏损:     {metrics.get('avg_loss', 0):>10.2f}%")
        report.append(f"盈亏比:       {metrics.get('profit_factor', 0):>10.2f}")
        report.append(f"最大单笔盈利: {metrics.get('max_profit', 0):>10.2f}%")
        report.append(f"最大单笔亏损: {metrics.get('max_loss', 0):>10.2f}%")
        report.append(f"平均持仓天数: {metrics.get('avg_holding_days', 0):>10.1f}天")
        report.append("")
        
        # 基准比较（如果有）
        if 'beta' in metrics:
            report.append("--- 基准比较 ---")
            report.append(f"Beta:         {metrics.get('beta', 0):>10.2f}")
            report.append(f"Alpha:        {metrics.get('alpha_annual', 0):>10.2f}%")
            report.append(f"跟踪误差:     {metrics.get('tracking_error', 0):>10.2f}%")
            report.append(f"相关性:       {metrics.get('correlation', 0):>10.2f}")
            report.append(f"超额收益:     {metrics.get('excess_return_annual', 0):>10.2f}%")
            report.append(f"R-squared:    {metrics.get('r_squared', 0):>10.2f}")
            report.append("")
        
        # 收益率分布
        report.append("--- 收益率分布 ---")
        report.append(f"最佳月收益:   {metrics.get('best_monthly_return', 0):>10.2f}%")
        report.append(f"最差月收益:   {metrics.get('worst_monthly_return', 0):>10.2f}%")
        report.append(f"最佳年收益:   {metrics.get('best_yearly_return', 0):>10.2f}%")
        report.append(f"最差年收益:   {metrics.get('worst_yearly_return', 0):>10.2f}%")
        report.append("")
        
        report.append("=" * 70)
        
        return "\n".join(report)
