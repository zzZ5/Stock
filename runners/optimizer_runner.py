"""
趋势雷达选股系统 - 参数优化运行脚本
执行网格搜索和Walk-Forward分析
"""
import os
import sys
import pandas as pd

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import DEFAULT_HOLDING_DAYS
from core.data_fetcher import DataFetcher
from core.utils import RateLimiter
from analysis.backtest import BacktestConfig
from analysis.optimizer import ParameterOptimizer
import config.settings as config


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
    param_grid = {
        'BREAKOUT_N': [50, 60, 70],           # 突破周期
        'MA_FAST': [15, 20, 25],              # 快速均线
        'MA_SLOW': [50, 60, 70],              # 慢速均线
        'VOL_CONFIRM_MULT': [1.2, 1.5, 2.0],   # 量能确认倍数
        'RSI_MAX': [70, 75, 80],              # RSI最大值
    }

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
        for col in param_grid.keys():
            print(f"  {col}: {best_params[col]}")
        print(f"\n最优得分: {best_params['score']:.2f}")
        print(f"总收益率: {best_params['total_return']:.2f}%")
        print(f"年化收益: {best_params['annual_return']:.2f}%")
        print(f"夏普比率: {best_params['sharpe_ratio']:.2f}")
        print(f"最大回撤: {best_params['max_drawdown']:.2f}%")
        print(f"胜率: {best_params['win_rate']:.2f}%")
        print(f"{'='*70}\n")

        return best_params


def main():
    """主函数"""
    print("\n请选择优化方式:")
    print("1. 网格搜索 (Grid Search)")
    print("2. Walk-Forward分析")
    print("0. 退出")

    choice = input("\n请输入选项 (0/1/2): ").strip()

    if choice == "1":
        grid_search_example()
    elif choice == "2":
        print("Walk-Forward分析功能开发中...")
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
