"""
趋势雷达选股系统 - 回测运行脚本
执行历史回测验证策略效果
"""
import os
import sys

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import DEFAULT_HOLDING_DAYS
from core.data_fetcher import DataFetcher
from core.utils import RateLimiter
from strategy.strategy import StockStrategy
from analysis.backtest import BacktestEngine, BacktestConfig


def main():
    """主函数"""
    print("="*70)
    print("趋势雷达选股系统 - 回测模块")
    print("="*70)

    # 初始化
    token = os.getenv("TUSHARE_TOKEN", "706b1dbca05800fea1d77c3a727f6ad5e0b3a1d0687f8a4e3266fe9c")
    rate_limiter = RateLimiter(max_calls_per_minute=200)
    fetcher = DataFetcher(token, rate_limiter)

    # 回测配置
    # 注意：第一次回测时需要下载大量历史数据，可能需要较长时间
    backtest_config = BacktestConfig(
        start_date="20240101",      # 回测开始日期
        end_date="20241231",        # 回测结束日期
        initial_capital=1000000.0,  # 100万初始资金
        max_positions=5,             # 最多持仓5只
        position_size=0.15,         # 单只15%仓位
        slippage=0.001,             # 0.1%滑点
        commission=0.0003,         # 0.03%手续费
        stop_loss=-0.10,           # -10%止损
        take_profit=0.25,          # 25%止盈
        max_holding_days=20,        # 最多持有20天
        rebalance_days=5           # 每5个交易日重新选股
    )

    print(f"\n回测配置:")
    print(f"  日期范围: {backtest_config.start_date} ~ {backtest_config.end_date}")
    print(f"  初始资金: {backtest_config.initial_capital:,.0f} 元")
    print(f"  最大持仓: {backtest_config.max_positions} 只")
    print(f"  单股仓位: {backtest_config.position_size*100:.1f}%")
    print(f"  止损比例: {backtest_config.stop_loss*100:.1f}%")
    print(f"  止盈比例: {backtest_config.take_profit*100:.1f}%")
    print(f"  最大持仓天数: {backtest_config.max_holding_days} 天")
    print(f"  选股间隔: {backtest_config.rebalance_days} 天")
    print(f"  滑点: {backtest_config.slippage*100:.2f}%")
    print(f"  手续费: {backtest_config.commission*100:.3f}%")

    # 获取股票基础信息
    print(f"\n正在获取股票基础信息...")
    basic_all = fetcher.get_stock_basic()
    strategy = StockStrategy(basic_all)

    # 运行回测
    engine = BacktestEngine(backtest_config, strategy, fetcher)
    results = engine.run()

    # 打印结果
    if results:
        engine.print_summary()

        # 保存结果
        output_dir = "./backtest_results"
        os.makedirs(output_dir, exist_ok=True)

        trades_file = f"{output_dir}/trades_{backtest_config.start_date}_{backtest_config.end_date}.csv"
        engine.save_results(trades_file)

        print(f"\n回测完成！")
        print(f"结果保存在: {output_dir}/")

    else:
        print("\n回测失败，未生成结果")


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
