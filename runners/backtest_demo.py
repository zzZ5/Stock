"""
增强版回测系统 - 一键运行演示脚本
快速体验回测系统的新功能
"""
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.transaction_cost import (
    TransactionCostCalculator,
    SlippageModel,
    MarketImpactModel,
    CommissionModel
)
from core.monte_carlo import MonteCarloSimulator, StressTester
from core.performance_metrics import PerformanceMetrics
from visualization.plotter import Plotter


def print_section(title):
    """打印章节标题"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)


def demo_transaction_cost():
    """演示交易成本计算"""
    print_section("1. 交易成本模型演示")
    
    # 创建交易成本计算器
    calculator = TransactionCostCalculator(
        slippage_model=SlippageModel(model_type='percentage', base_slippage=0.0002),
        market_impact_model=MarketImpactModel(model_type='almgren_christoss'),
        commission_model=CommissionModel(commission_type='percentage', base_rate=0.0003, min_commission=5.0)
    )
    
    # 模拟不同规模的交易
    trade_sizes = [
        {'price': 10.0, 'volume': 1000, 'shares': 1000, 'name': '小额交易 (1万元)'},
        {'price': 50.0, 'volume': 10000, 'shares': 10000, 'name': '中额交易 (50万元)'},
        {'price': 100.0, 'volume': 50000, 'shares': 50000, 'name': '大额交易 (500万元)'},
    ]
    
    print("\n交易成本对比:")
    print(f"{'交易类型':<20} {'买入总成本':<15} {'卖出净收益':<15} {'总成本占比':<15}")
    print("-"*70)
    
    for trade in trade_sizes:
        buy_result = calculator.calculate_buy_cost(
            price=trade['price'],
            shares=trade['shares'],
            volume=trade['volume'],
            avg_daily_volume=1000000,
            avg_price=trade['price'],
            volatility=0.02
        )
        
        sell_result = calculator.calculate_sell_cost(
            price=trade['price'],
            shares=trade['shares'],
            volume=trade['volume'],
            avg_daily_volume=1000000,
            avg_price=trade['price'],
            volatility=0.02
        )
        
        trade_value = trade['price'] * trade['shares']
        total_cost = buy_result['total_cost'] + (trade_value - sell_result['net_proceeds'])
        total_pct = total_cost / trade_value * 2
        
        print(f"{trade['name']:<20} {buy_result['total_cost']:>10.2f}元  {sell_result['net_proceeds']:>10.2f}元  {total_pct:>12.4f}%")
    
    print(f"\n成本明细 (以大额交易为例):")
    buy_result = calculator.calculate_buy_cost(
        price=100.0,
        shares=50000,
        volume=50000,
        avg_daily_volume=1000000,
        avg_price=100.0,
        volatility=0.02
    )
    print(f"  原始价格: {buy_result['original_price']:.2f}元")
    print(f"  滑点后价格: {buy_result['slippage_price']:.2f}元")
    print(f"  最终价格: {buy_result['final_price']:.2f}元")
    print(f"  手续费: {buy_result['commission']:.2f}元")
    print(f"  总成本: {buy_result['total_cost']:.2f}元 ({buy_result['total_cost_pct']:.4f}%)")
    print(f"  滑点占比: {buy_result['slippage_pct']:.4f}%")
    print(f"  市场冲击占比: {buy_result['impact_pct']:.4f}%")
    
    return calculator


def demo_monte_carlo():
    """演示蒙特卡洛模拟"""
    print_section("2. 蒙特卡洛模拟演示")
    
    # 生成模拟的资产曲线和交易数据
    np.random.seed(42)
    n_days = 252
    
    dates = pd.date_range(start='2023-01-01', periods=n_days, freq='D')
    returns = np.random.randn(n_days) * 0.015 + 0.0005
    equity_curve = pd.DataFrame({
        'date': dates,
        'equity': 1000000 * (1 + pd.Series(returns).cumsum())
    })
    
    trades_df = pd.DataFrame({
        'date': pd.date_range(start='2023-01-01', periods=20, freq='12D'),
        'symbol': ['000001'] * 20,
        'action': ['buy'] * 10 + ['sell'] * 10,
        'price': np.random.uniform(10, 20, 20),
        'volume': np.random.randint(100, 1000, 20)
    })
    
    # 创建模拟器
    simulator = MonteCarloSimulator(
        equity_curve=equity_curve,
        trades_df=trades_df,
        initial_capital=1000000,
        n_simulations=200
    )
    
    print(f"\n历史数据统计:")
    print(f"  交易日数: {len(equity_curve)}")
    print(f"  年化收益率: {simulator.mean_return * 252:.2%}")
    print(f"  年化波动率: {simulator.std_return * np.sqrt(252):.2%}")
    print(f"  偏度: {simulator.skewness:.3f}")
    print(f"  峰度: {simulator.kurtosis:.3f}")
    
    # 运行批量模拟
    print(f"\n运行蒙特卡洛模拟 (200次)...")
    batch_results = simulator.run_batch_simulation(
        n_simulations=200,
        method='gbm',
        n_workers=1,  # 使用单线程避免序列化问题
        show_progress=False
    )
    
    # 计算风险指标
    risk_metrics = simulator.calculate_risk_metrics(batch_results)
    
    print(f"\n蒙特卡洛模拟结果 (200次模拟):")
    print(f"  平均收益率: {risk_metrics['return_mean']:.2%}")
    print(f"  收益率中位数: {risk_metrics['return_median']:.2%}")
    print(f"  收益率标准差: {risk_metrics['return_std']:.2%}")
    print(f"  5%分位数: {risk_metrics['return_5th_percentile']:.2%}")
    print(f"  95%分位数: {risk_metrics['return_95th_percentile']:.2%}")
    print(f"  95% VaR: {risk_metrics['var_95']:.2%}")
    print(f"  95% CVaR: {risk_metrics['cvar_95']:.2%}")
    print(f"  平均最大回撤: {risk_metrics['max_drawdown_mean']:.2%}")
    print(f"  最差最大回撤: {risk_metrics['max_drawdown_worst']:.2%}")
    print(f"  盈利概率: {risk_metrics['prob_positive']:.1%}")
    print(f"  亏损概率: {risk_metrics['prob_negative']:.1%}")
    print(f"  亏损超过20%的概率: {risk_metrics['prob_loss_20pct']:.1%}")
    
    # 展示收益分布
    print(f"\n收益分布统计:")
    print(f"  最好情况 (95%分位): {risk_metrics['return_95th_percentile']:.2%}")
    print(f"  最差情况 (5%分位): {risk_metrics['return_5th_percentile']:.2%}")
    print(f"  95%置信区间: [{risk_metrics['return_5th_percentile']:.2%}, {risk_metrics['return_95th_percentile']:.2%}]")
    
    return batch_results


def demo_stress_test():
    """演示压力测试"""
    print_section("3. 压力测试演示")
    
    print(f"\n压力测试需要真实的回测结果数据")
    print(f"此演示跳过压力测试环节")
    print(f"\n提示: 运行实际回测后可以使用 StressTester 进行压力测试")
    print(f"支持的场景:")
    print(f"  - 市场暴跌场景 (如2008金融危机、2020疫情)")
    print(f"  - 波动率飙升场景 (如2007-2008)")
    print(f"  - 相关性危机场景 (资产相关性急剧上升)")
    print(f"\n建议:")
    print(f"  1. 先完成策略回测获得资产曲线")
    print(f"  2. 使用 StressTester 分析不同市场环境下的表现")
    print(f"  3. 关注极端情况下的最大损失")


def demo_performance_metrics():
    """演示绩效评估"""
    print_section("4. 绩效评估演示")
    
    # 生成模拟的资产曲线
    np.random.seed(42)
    n_days = 252
    dates = pd.date_range(start='2023-01-01', periods=n_days, freq='D')
    
    # 策略收益率（略好于基准）
    strategy_returns = np.random.randn(n_days) * 0.015 + 0.001
    equity_curve = pd.DataFrame({
        'date': dates,
        'equity': 1000000 * (1 + pd.Series(strategy_returns).cumsum())
    })
    
    # 基准收益率
    benchmark_returns = pd.Series(
        np.random.randn(n_days) * 0.012 + 0.0005,
        index=dates
    )
    
    # 生成模拟交易记录
    n_trades = 100
    trades_df = pd.DataFrame({
        'date': pd.date_range(start='2023-01-01', periods=n_trades, freq='3D'),
        'entry_date': pd.date_range(start='2023-01-01', periods=n_trades, freq='3D'),
        'exit_date': pd.date_range(start='2023-01-01', periods=n_trades, freq='3D') + pd.Timedelta(days=5),
        'symbol': np.random.choice(['000001', '000002', '000003', '600000', '600519'], n_trades),
        'action': np.random.choice(['buy', 'sell'], n_trades),
        'price': np.random.uniform(10, 100, n_trades),
        'volume': np.random.randint(100, 1000, n_trades),
        'pnl': np.random.randn(n_trades) * 1000,  # 添加pnl列
        'pnl_pct': np.random.randn(n_trades) * 0.1,  # 添加pnl_pct列
        'holding_days': np.random.randint(1, 30, n_trades),  # 添加holding_days列
        'position_size': np.random.uniform(0.1, 0.2, n_trades)  # 添加position_size列
    })
    
    # 计算绩效指标
    metrics = PerformanceMetrics.calculate_all_metrics(
        equity_curve=equity_curve,
        trades_df=trades_df,
        initial_capital=1000000,
        benchmark_returns=benchmark_returns
    )
    
    print(f"\n=== 基础指标 ===")
    print(f"总收益率:           {metrics['total_return']:.2%}")
    print(f"年化收益率:         {metrics['annual_return']:.2%}")
    print(f"日均收益率:         {metrics['avg_daily_return']:.2%}")
    print(f"最大回撤:           {metrics['max_drawdown']:.2%}")
    
    print(f"\n=== 风险调整收益 ===")
    print(f"夏普比率:           {metrics['sharpe_ratio']:.3f}")
    print(f"Sortino比率:        {metrics['sortino_ratio']:.3f}")
    print(f"Calmar比率:         {metrics['calmar_ratio']:.3f}")
    print(f"Omega比率:          {metrics['omega_ratio']:.3f}")
    
    print(f"\n=== 基准比较 ===")
    print(f"Beta:               {metrics['beta']:.3f}")
    print(f"Alpha (年化):       {metrics['alpha_annual']:.2%}")
    print(f"跟踪误差:           {metrics['tracking_error']:.2%}")
    print(f"信息比率:           {metrics['information_ratio']:.3f}")
    print(f"相关系数:           {metrics['correlation']:.3f}")
    print(f"R-squared:          {metrics['r_squared']:.2%}")
    print(f"超基准收益 (年化): {metrics['excess_return_annual']:.2%}")
    
    print(f"\n=== 交易统计 ===")
    print(f"总交易次数:         {metrics.get('total_trades', 'N/A')}")
    print(f"盈利交易次数:       {metrics.get('total_win_trades', 'N/A')}")
    print(f"亏损交易次数:       {metrics.get('total_lose_trades', 'N/A')}")
    print(f"胜率:               {metrics.get('win_rate', 0):.1%}")
    print(f"平均盈利:           {metrics.get('avg_win', 0):.2f}")
    print(f"平均亏损:           {metrics.get('avg_loss', 0):.2f}")
    print(f"平均持仓天数:       {metrics.get('avg_holding_days', 0):.1f}天")
    
    # 绩效评级
    print(f"\n=== 综合评级 ===")
    if metrics['sharpe_ratio'] > 2 and metrics['max_drawdown'] > -20:
        grade = "优秀"
    elif metrics['sharpe_ratio'] > 1.5 and metrics['max_drawdown'] > -30:
        grade = "良好"
    elif metrics['sharpe_ratio'] > 1.0 and metrics['max_drawdown'] > -40:
        grade = "一般"
    else:
        grade = "需要改进"
    print(f"策略评级: {grade}")
    
    return metrics


def demo_integration():
    """演示集成使用"""
    print_section("5. 综合应用演示")
    
    print(f"\n模拟一个完整的策略回测流程:")
    print(f"1. 定义策略 → 2. 运行回测 → 3. 计算成本 → 4. 风险评估 → 5. 压力测试")
    
    # 模拟策略表现
    np.random.seed(42)
    n_days = 252
    
    # 生成多个场景的资产曲线
    scenarios = {
        '乐观': 0.002,  # 年化约50%
        '基准': 0.001,  # 年化约25%
        '悲观': 0.0003  # 年化约8%
    }
    
    print(f"\n不同市场环境下的策略表现:")
    print(f"{'市场环境':<12} {'年化收益':>12} {'最大回撤':>12} {'夏普比率':>12} {'抗风险能力':>12}")
    print("-"*70)
    
    for scenario, drift in scenarios.items():
        returns = np.random.randn(n_days) * 0.02 + drift
        equity = 1000000 * (1 + pd.Series(returns).cumsum())
        
        equity_curve = pd.DataFrame({
            'date': pd.date_range(start='2023-01-01', periods=n_days),
            'equity': equity
        })
        
        metrics = PerformanceMetrics.calculate_all_metrics(
            equity_curve=equity_curve,
            trades_df=pd.DataFrame(),
            initial_capital=1000000
        )
        
        # 抗风险能力评分 (基于最大回撤和夏普比率)
        risk_score = (metrics['sharpe_ratio'] * 50 + abs(metrics['max_drawdown']) * 100)
        
        print(f"{scenario:<12} {metrics['annual_return']:>10.2%}  {metrics['max_drawdown']:>10.2%}  {metrics['sharpe_ratio']:>10.2f}  {risk_score:>10.1f}")
    
    print(f"\n建议:")
    print(f"  1. 根据市场环境调整仓位大小")
    print(f"  2. 在高波动环境下加强风险控制")
    print(f"  3. 定期进行压力测试评估抗风险能力")
    print(f"  4. 关注交易成本对收益的影响")


def main():
    """主函数"""
    print("\n")
    print("╔" + "═"*68 + "╗")
    print("║" + " "*68 + "║")
    print("║" + "    增强版回测系统 - 一键演示".center(68) + "║")
    print("║" + "    Enhanced Backtest System Demo".center(68) + "║")
    print("║" + " "*68 + "║")
    print("╚" + "═"*68 + "╝")
    
    try:
        # 1. 交易成本模型
        demo_transaction_cost()
        
        # 2. 蒙特卡洛模拟
        demo_monte_carlo()
        
        # 3. 压力测试
        demo_stress_test()
        
        # 4. 绩效评估
        demo_performance_metrics()
        
        # 5. 综合应用
        demo_integration()
        
        print_section("演示完成")
        print(f"\n[OK] 交易成本模型: 支持4种滑点模型、3种市场冲击模型、3种手续费模型")
        print(f"[OK] 蒙特卡洛模拟: 支持GBM、Bootstrap等方法，提供VaR、CVaR等风险指标")
        print(f"[OK] 压力测试: 支持3种压力场景（市场暴跌、波动率飙升、相关性危机）")
        print(f"[OK] 绩效评估: 支持20+个风险调整收益指标和基准比较指标")
        print(f"\n下一步操作:")
        print(f"  1. 查看 BACKTEST_GUIDE.md 了解详细使用方法")
        print(f"  2. 根据实际需求修改配置参数")
        print(f"  3. 集成到你的策略回测流程中")
        print(f"\n提示: 运行实际回测可以使用 python runners/backtest_runner.py")
        print(f"="*70)
        
    except Exception as e:
        print(f"\n[X] 演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
