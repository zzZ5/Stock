"""
趋势雷达选股系统 - 参数优化运行脚本
执行网格搜索和Walk-Forward分析
"""
import os
import sys
import pandas as pd

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import settings
from core.logger import Logger, get_logger
from core.data_fetcher import DataFetcher
from core.utils import RateLimiter
from analysis.backtest import BacktestConfig
from analysis.optimizer import ParameterOptimizer

# 初始化日志系统
Logger.setup_logging(
    log_level=settings.LOG_LEVEL,
    log_dir=settings.LOG_DIR,
    console_output=settings.LOG_CONSOLE_OUTPUT,
    file_output=settings.LOG_FILE_OUTPUT,
    max_file_size=settings.LOG_MAX_FILE_SIZE,
    backup_count=settings.LOG_BACKUP_COUNT
)
logger = get_logger(__name__)


def grid_search_example():
    """网格搜索示例"""
    print("="*70)
    print("参数优化 - 网格搜索")
    print("="*70)

    # 初始化
    token = os.getenv("TUSHARE_TOKEN", "706b1dbca05800fea1d77c3a727f6ad5e0b3a1d0687f8a4e3266fe9c")
    rate_limiter = RateLimiter(max_calls_per_minute=200)
    fetcher = DataFetcher(token, rate_limiter)

    # 回测配置
    backtest_config = BacktestConfig(
        start_date="20240101",
        end_date="20241231",
        initial_capital=1000000.0,
        max_positions=5,
        position_size=0.15,
        slippage=0.001,
        commission=0.0003,
        stop_loss=-0.10,
        take_profit=0.25,
        max_holding_days=20,
        rebalance_days=5
    )

    # 参数网格（可根据需要调整）
    # 注意：组合数量 = len(values1) * len(values2) * ...
    # 组合越多，优化时间越长
    # 快速测试模式：2*2*2*1*2*2*2*2 = 128组合
    # 完整测试模式：3*3*3*1*3*3*3*3 = 2187组合
    quick_test = True  # 设为True使用快速测试，False使用完整测试
    if quick_test:
        param_grid = {
            'BREAKOUT_N': [50, 60],               # 日线突破周期
            'WEEKLY_BREAKOUT_N': [8, 12],          # 周线突破周期
            'MONTHLY_BREAKOUT_N': [4, 6],           # 月线突破周期
            'MULTI_TIMEFRAME_MODE': [True],           # 多周期模式
            'MA_FAST': [15, 20],                  # 快速均线
            'MA_SLOW': [50, 60],                  # 慢速均线
            'VOL_CONFIRM_MULT': [1.2, 1.5],         # 量能确认倍数
            'RSI_MAX': [70, 75],                   # RSI最大值
        }
        print("\n【快速测试模式】")
    else:
        param_grid = {
            'BREAKOUT_N': [50, 60, 70],           # 日线突破周期
            'WEEKLY_BREAKOUT_N': [8, 12, 16],     # 周线突破周期
            'MONTHLY_BREAKOUT_N': [4, 6, 8],      # 月线突破周期
            'MULTI_TIMEFRAME_MODE': [True],       # 多周期模式（可添加False对比）
            'MA_FAST': [15, 20, 25],              # 快速均线
            'MA_SLOW': [50, 60, 70],              # 慢速均线
            'VOL_CONFIRM_MULT': [1.2, 1.5, 2.0],   # 量能确认倍数
            'RSI_MAX': [70, 75, 80],              # RSI最大值
        }
        print("\n【完整测试模式】")

    print(f"\n参数网格:")
    for name, values in param_grid.items():
        print(f"  {name}: {values}")

    total_combinations = 1
    for values in param_grid.values():
        total_combinations *= len(values)

    print(f"\n总组合数: {total_combinations}")
    print(f"预计耗时: 约 {total_combinations * 0.5:.1f} 分钟\n")

    # 运行网格搜索
    optimizer = ParameterOptimizer(fetcher, backtest_config)
    results_df = optimizer.grid_search(param_grid, show_progress=True)

    # 保存结果
    if not results_df.empty:
        output_dir = "./optimization_results"
        os.makedirs(output_dir, exist_ok=True)

        output_file = f"{output_dir}/grid_search_{backtest_config.start_date}_{backtest_config.end_date}.csv"
        optimizer.save_results(results_df, output_file)

        # 保存最优参数
        best_params = results_df.iloc[0]
        print(f"\n{'='*70}")
        print(f"最优参数组合:")
        print(f"{'='*70}")
        for col in ['MULTI_TIMEFRAME_MODE', 'BREAKOUT_N', 'WEEKLY_BREAKOUT_N',
                    'MONTHLY_BREAKOUT_N', 'MA_FAST', 'MA_SLOW',
                    'VOL_CONFIRM_MULT', 'RSI_MAX']:
            if col in best_params:
                print(f"  {col}: {best_params[col]}")
        print(f"\n最优得分: {best_params['score']:.2f}")
        print(f"总收益率: {best_params['total_return']:.2f}%")
        print(f"年化收益: {best_params['annual_return']:.2f}%")
        print(f"夏普比率: {best_params['sharpe_ratio']:.2f}")
        print(f"最大回撤: {best_params['max_drawdown']:.2f}%")
        print(f"胜率: {best_params['win_rate']:.2f}%")
        print(f"{'='*70}\n")

        return best_params


def walk_forward_example():
    """Walk-Forward分析示例"""
    print("="*70)
    print("参数优化 - Walk-Forward分析")
    print("="*70)

    # 初始化
    token = os.getenv("TUSHARE_TOKEN", "706b1dbca05800fea1d77c3a727f6ad5e0b3a1d0687f8a4e3266fe9c")
    rate_limiter = RateLimiter(max_calls_per_minute=200)
    fetcher = DataFetcher(token, rate_limiter)

    # 回测配置（需要至少1年数据）
    backtest_config = BacktestConfig(
        start_date="20230101",
        end_date="20241231",
        initial_capital=1000000.0,
        max_positions=5,
        position_size=0.15,
        slippage=0.001,
        commission=0.0003,
        stop_loss=-0.10,
        take_profit=0.25,
        max_holding_days=20,
        rebalance_days=5
    )

    # 参数网格（为了加快速度，使用较小网格）
    quick_test = True  # 设为True使用快速测试，False使用完整测试
    if quick_test:
        param_grid = {
            'BREAKOUT_N': [50, 60],
            'WEEKLY_BREAKOUT_N': [8, 12],
            'MONTHLY_BREAKOUT_N': [4, 6],
            'MULTI_TIMEFRAME_MODE': [True],
            'MA_FAST': [15, 20],
            'MA_SLOW': [50, 60],
            'VOL_CONFIRM_MULT': [1.2, 1.5],
            'RSI_MAX': [70, 75]
        }
        print("\n【快速测试模式】")
    else:
        param_grid = {
            'BREAKOUT_N': [50, 60, 70],
            'WEEKLY_BREAKOUT_N': [8, 12, 16],
            'MONTHLY_BREAKOUT_N': [4, 6, 8],
            'MULTI_TIMEFRAME_MODE': [True],
            'MA_FAST': [15, 20],
            'MA_SLOW': [50, 60],
            'VOL_CONFIRM_MULT': [1.2, 1.5],
            'RSI_MAX': [70, 75]
        }
        print("\n【完整测试模式】")

    print(f"\n参数网格:")
    for name, values in param_grid.items():
        print(f"  {name}: {values}")

    # Walk-Forward配置
    train_days = 252   # 训练期：1年
    test_days = 63     # 测试期：3个月
    step_days = 63      # 步长：3个月

    print(f"\nWalk-Forward配置:")
    print(f"  训练期: {train_days}交易日 (约1年)")
    print(f"  测试期: {test_days}交易日 (约3个月)")
    print(f"  滚动步长: {step_days}交易日")

    # 运行Walk-Forward分析
    optimizer = ParameterOptimizer(fetcher, backtest_config)
    wf_df = optimizer.walk_forward_analysis(
        train_days=train_days,
        test_days=test_days,
        step_days=step_days,
        param_grid=param_grid
    )

    # 保存结果
    if not wf_df.empty:
        output_dir = "./optimization_results"
        os.makedirs(output_dir, exist_ok=True)

        output_file = f"{output_dir}/walk_forward_{backtest_config.start_date}_{backtest_config.end_date}.csv"
        optimizer.save_results(wf_df, output_file)

        # 打印最终建议参数
        print(f"\n{'='*70}")
        print(f"最终建议参数 (基于稳定性分析):")
        print(f"{'='*70}")
        for col in ['MULTI_TIMEFRAME_MODE', 'BREAKOUT_N', 'WEEKLY_BREAKOUT_N',
                    'MONTHLY_BREAKOUT_N', 'MA_FAST', 'MA_SLOW',
                    'VOL_CONFIRM_MULT', 'RSI_MAX']:
            if col in wf_df.columns:
                most_common = wf_df[col].mode().iloc[0] if len(wf_df[col].mode()) > 0 else wf_df[col].iloc[0]
                print(f"  {col}: {most_common}")
        print(f"{'='*70}\n")

        return wf_df


def main():
    """主函数"""
    print("\n请选择优化方式:")
    print("1. 网格搜索 (Grid Search)")
    print("2. Walk-Forward分析")
    print("3. 贝叶斯优化 (随机搜索)")
    print("0. 退出")

    choice = input("\n请输入选项 (0/1/2/3): ").strip()

    if choice == "1":
        grid_search_example()
    elif choice == "2":
        walk_forward_example()
    elif choice == "3":
        print("贝叶斯优化功能开发中...")
        # TODO: 实现 bayesian_optimization_example()
    elif choice == "0":
        print("退出")
        return
    else:
        print("无效选项")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n用户中断")
        sys.exit(0)
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
