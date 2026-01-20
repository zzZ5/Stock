"""
趋势雷达选股系统 - 参数优化器模块
包含网格搜索、Walk-Forward分析、贝叶斯优化等参数优化功能
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import itertools
from tqdm import tqdm
from datetime import datetime, timedelta

from .backtest import BacktestEngine, BacktestConfig
from config.settings import DEFAULT_HOLDING_DAYS
import config.settings as config


class ParameterOptimizer:
    """参数优化器"""

    def __init__(self, fetcher, backtest_config: BacktestConfig):
        """
        初始化参数优化器

        参数:
            fetcher: 数据获取器实例
            backtest_config: 回测配置
        """
        self.fetcher = fetcher
        self.backtest_config = backtest_config
        self.results_history = []

    def grid_search(self, param_grid: Dict[str, List],
                    show_progress: bool = True) -> pd.DataFrame:
        """
        网格搜索最优参数

        参数:
            param_grid: 参数网格字典
                {
                    'BREAKOUT_N': [40, 60, 80],
                    'MA_FAST': [10, 20, 30],
                    'MA_SLOW': [40, 60, 80],
                    'VOL_CONFIRM_MULT': [1.2, 1.5, 2.0],
                    'RSI_MAX': [70, 75, 80]
                }
            show_progress: 是否显示进度条

        返回:
            包含所有参数组合结果的DataFrame，按综合得分排序
        """
        results = []
        param_combinations = list(itertools.product(*param_grid.values()))
        param_names = list(param_grid.keys())

        print(f"\n{'='*70}")
        print(f"参数网格搜索")
        print(f"{'='*70}")
        print(f"参数范围:")
        for name, values in param_grid.items():
            print(f"  {name}: {values}")
        print(f"总组合数: {len(param_combinations)}")
        print(f"{'='*70}\n")

        # 遍历所有参数组合
        iterator = tqdm(param_combinations, desc="参数优化") if show_progress else param_combinations

        for params in iterator:
            # 保存原始参数
            original_params = {}
            for name in param_names:
                original_params[name] = getattr(config, name)

            # 更新参数
            for i, name in enumerate(param_names):
                setattr(config, name, params[i])

            # 运行回测
            from strategy import StockStrategy
            basic_all = self.fetcher.get_stock_basic()
            strategy = StockStrategy(basic_all)

            engine = BacktestEngine(self.backtest_config, strategy, self.fetcher)
            result = engine.run()

            # 计算综合得分
            score = self._calculate_composite_score(result)

            # 记录结果
            result_row = {
                **{name: params[i] for i, name in enumerate(param_names)},
                'total_return': result.get('total_return', 0),
                'annual_return': result.get('annual_return', 0),
                'sharpe_ratio': result.get('sharpe_ratio', 0),
                'max_drawdown': result.get('max_drawdown', 0),
                'win_rate': result.get('win_rate', 0),
                'profit_factor': result.get('profit_factor', 0),
                'total_trades': result.get('total_trades', 0),
                'score': score
            }
            results.append(result_row)

            # 恢复原始参数
            for name in param_names:
                setattr(config, name, original_params[name])

        # 转换为DataFrame并排序
        df = pd.DataFrame(results)
        df = df.sort_values('score', ascending=False).reset_index(drop=True)

        # 打印结果摘要
        self._print_optimization_summary(df)

        return df

    def _calculate_composite_score(self, result: Dict) -> float:
        """
        计算综合得分（可自定义权重）

        得分计算公式:
        Score = 0.4 * 归一化年化收益
              + 0.3 * 归一化夏普比率
              - 0.2 * 归一化最大回撤
              + 0.1 * 归一化胜率

        参数:
            result: 回测结果字典

        返回:
            综合得分
        """
        if not result:
            return 0.0

        # 提取指标
        annual_return = result.get('annual_return', 0) / 100  # 转换为小数
        sharpe_ratio = result.get('sharpe_ratio', 0)
        max_drawdown = abs(result.get('max_drawdown', 0)) / 100  # 取绝对值并转换为小数
        win_rate = result.get('win_rate', 0) / 100  # 转换为小数

        # 归一化（使用经验阈值）
        # 年化收益：假设优秀范围为0-50%
        normalized_return = min(max(annual_return / 0.5, 0), 1)

        # 夏普比率：假设优秀范围为0-3
        normalized_sharpe = min(max(sharpe_ratio / 3.0, 0), 1)

        # 最大回撤：假设可接受范围为0-50%，越小越好
        normalized_drawdown = min(max(1 - max_drawdown / 0.5, 0), 1)

        # 胜率：假设优秀范围为50%-70%
        normalized_winrate = min(max((win_rate - 0.5) / 0.2, 0), 1)

        # 加权求和
        score = (
            0.35 * normalized_return +
            0.30 * normalized_sharpe +
            0.25 * normalized_drawdown +
            0.10 * normalized_winrate
        ) * 100  # 转换为0-100分

        return score

    def _print_optimization_summary(self, df: pd.DataFrame):
        """打印优化摘要"""
        print(f"\n{'='*70}")
        print(f"参数优化结果（Top 10）")
        print(f"{'='*70}\n")

        # 显示前10名
        top10 = df.head(10)

        for idx, row in top10.iterrows():
            print(f"排名 {idx+1}:")
            for col in df.columns:
                if col not in ['score', 'total_return', 'annual_return', 'sharpe_ratio',
                               'max_drawdown', 'win_rate', 'profit_factor', 'total_trades']:
                    print(f"  {col}: {row[col]}")

            print(f"  总收益率: {row['total_return']:.2f}%")
            print(f"  年化收益: {row['annual_return']:.2f}%")
            print(f"  夏普比率: {row['sharpe_ratio']:.2f}")
            print(f"  最大回撤: {row['max_drawdown']:.2f}%")
            print(f"  胜率: {row['win_rate']:.2f}%")
            print(f"  综合得分: {row['score']:.2f}")
            print()

        # 统计信息
        print(f"{'='*70}")
        print(f"统计信息:")
        print(f"  平均年化收益: {df['annual_return'].mean():.2f}%")
        print(f"  最大年化收益: {df['annual_return'].max():.2f}%")
        print(f"  最小年化收益: {df['annual_return'].min():.2f}%")
        print(f"  平均夏普比率: {df['sharpe_ratio'].mean():.2f}")
        print(f"  平均最大回撤: {df['max_drawdown'].mean():.2f}%")
        print(f"  平均胜率: {df['win_rate'].mean():.2f}%")
        print(f"{'='*70}\n")

    def save_results(self, df: pd.DataFrame, filepath: str):
        """保存优化结果"""
        df.to_csv(filepath, index=False, encoding='utf-8-sig')
        print(f"优化结果已保存到: {filepath}")

    def walk_forward_analysis(self,
                             train_days: int = 252,      # 训练期（1年=252交易日）
                             test_days: int = 63,        # 测试期（3个月=63交易日）
                             step_days: int = 63,        # 步长
                             param_grid: Dict[str, List] = None) -> pd.DataFrame:
        """
        Walk-Forward滚动验证分析
        避免过拟合，测试参数的稳定性

        参数:
            train_days: 训练期交易日数
            test_days: 测试期交易日数
            step_days: 滚动步长
            param_grid: 参数网格（如果为None则使用默认网格）

        返回:
            包含所有窗口结果的DataFrame
        """
        print(f"\n{'='*70}")
        print(f"Walk-Forward分析")
        print(f"{'='*70}")
        print(f"训练期: {train_days}交易日")
        print(f"测试期: {test_days}交易日")
        print(f"滚动步长: {step_days}交易日")
        print(f"{'='*70}\n")

        # 获取所有交易日
        all_dates = self.fetcher.get_trade_cal(
            end_date=self.backtest_config.end_date,
            lookback_calendar_days=2000
        )

        # 筛选回测日期范围
        date_range = [d for d in all_dates
                     if self.backtest_config.start_date <= d <= self.backtest_config.end_date]

        if len(date_range) < (train_days + test_days):
            print(f"错误: 数据不足，需要至少 {train_days + test_days} 个交易日")
            return pd.DataFrame()

        # 创建时间窗口
        windows = self._create_walk_forward_windows(
            date_range, train_days, test_days, step_days
        )

        print(f"共创建 {len(windows)} 个时间窗口\n")

        # 默认参数网格
        if param_grid is None:
            param_grid = {
                'BREAKOUT_N': [40, 60, 80],
                'MA_FAST': [10, 20],
                'MA_SLOW': [40, 60],
                'VOL_CONFIRM_MULT': [1.2, 1.5],
                'RSI_MAX': [70, 75]
            }

        results = []

        # 逐窗口分析
        for i, window in enumerate(tqdm(windows, desc="Walk-Forward")):
            print(f"\n--- 窗口 {i+1}/{len(windows)} ---")
            print(f"训练期: {window['train_start']} ~ {window['train_end']}")
            print(f"测试期: {window['test_start']} ~ {window['test_end']}")

            # 1. 训练期参数优化
            train_config = BacktestConfig(
                start_date=window['train_start'],
                end_date=window['train_end'],
                initial_capital=self.backtest_config.initial_capital,
                max_positions=self.backtest_config.max_positions,
                position_size=self.backtest_config.position_size,
                slippage=self.backtest_config.slippage,
                commission=self.backtest_config.commission,
                stop_loss=self.backtest_config.stop_loss,
                take_profit=self.backtest_config.take_profit,
                max_holding_days=self.backtest_config.max_holding_days,
                rebalance_days=self.backtest_config.rebalance_days
            )

            # 在训练期优化参数
            print(f"  -> 训练期优化参数...")
            train_optimizer = ParameterOptimizer(self.fetcher, train_config)
            train_results = train_optimizer.grid_search(param_grid, show_progress=False)

            if train_results.empty:
                print(f"  -> 训练期优化失败，跳过此窗口")
                continue

            best_params = train_results.iloc[0]
            best_params_dict = {
                'BREAKOUT_N': int(best_params['BREAKOUT_N']),
                'MA_FAST': int(best_params['MA_FAST']),
                'MA_SLOW': int(best_params['MA_SLOW']),
                'VOL_CONFIRM_MULT': float(best_params['VOL_CONFIRM_MULT']),
                'RSI_MAX': int(best_params['RSI_MAX'])
            }

            print(f"  -> 最优参数: {best_params_dict}")

            # 2. 测试期验证
            # 更新参数
            for param_name, param_value in best_params_dict.items():
                setattr(config, param_name, param_value)

            test_config = BacktestConfig(
                start_date=window['test_start'],
                end_date=window['test_end'],
                initial_capital=self.backtest_config.initial_capital,
                max_positions=self.backtest_config.max_positions,
                position_size=self.backtest_config.position_size,
                slippage=self.backtest_config.slippage,
                commission=self.backtest_config.commission,
                stop_loss=self.backtest_config.stop_loss,
                take_profit=self.backtest_config.take_profit,
                max_holding_days=self.backtest_config.max_holding_days,
                rebalance_days=self.backtest_config.rebalance_days
            )

            from strategy import StockStrategy
            basic_all = self.fetcher.get_stock_basic()
            strategy = StockStrategy(basic_all)

            engine = BacktestEngine(test_config, strategy, self.fetcher)
            test_result = engine.run()

            # 记录结果
            result_row = {
                'window': i + 1,
                'train_start': window['train_start'],
                'train_end': window['train_end'],
                'test_start': window['test_start'],
                'test_end': window['test_end'],
                **best_params_dict,
                'train_return': best_params['annual_return'],
                'train_sharpe': best_params['sharpe_ratio'],
                'train_drawdown': best_params['max_drawdown'],
                'test_return': test_result.get('annual_return', 0),
                'test_sharpe': test_result.get('sharpe_ratio', 0),
                'test_drawdown': test_result.get('max_drawdown', 0),
                'test_winrate': test_result.get('win_rate', 0),
                'test_trades': test_result.get('total_trades', 0)
            }

            results.append(result_row)

        # 转换为DataFrame
        df = pd.DataFrame(results)

        if not df.empty:
            # 分析稳定性
            self._print_walk_forward_summary(df)
        else:
            print("\nWalk-Forward分析完成，但无有效结果")

        return df

    def _create_walk_forward_windows(self,
                                    date_range: List[str],
                                    train_days: int,
                                    test_days: int,
                                    step_days: int) -> List[Dict]:
        """
        创建Walk-Forward时间窗口

        返回:
            窗口列表，每个窗口包含训练期和测试期的起止日期
        """
        windows = []
        idx = 0

        while idx + train_days + test_days <= len(date_range):
            train_start = date_range[idx]
            train_end = date_range[idx + train_days - 1]
            test_start = date_range[idx + train_days]
            test_end = date_range[idx + train_days + test_days - 1]

            windows.append({
                'train_start': train_start,
                'train_end': train_end,
                'test_start': test_start,
                'test_end': test_end
            })

            idx += step_days

        return windows

    def _print_walk_forward_summary(self, df: pd.DataFrame):
        """打印Walk-Forward分析摘要"""
        print(f"\n{'='*70}")
        print(f"Walk-Forward分析摘要")
        print(f"{'='*70}\n")

        # 训练期 vs 测试期对比
        print(f"训练期平均表现:")
        print(f"  年化收益: {df['train_return'].mean():.2f}%")
        print(f"  夏普比率: {df['train_sharpe'].mean():.2f}")
        print(f"  最大回撤: {df['train_drawdown'].mean():.2f}%\n")

        print(f"测试期平均表现:")
        print(f"  年化收益: {df['test_return'].mean():.2f}%")
        print(f"  夏普比率: {df['test_sharpe'].mean():.2f}")
        print(f"  最大回撤: {df['test_drawdown'].mean():.2f}%")
        print(f"  胜率: {df['test_winrate'].mean():.2f}%\n")

        # 参数稳定性分析
        print(f"参数稳定性:")
        param_cols = ['BREAKOUT_N', 'MA_FAST', 'MA_SLOW', 'VOL_CONFIRM_MULT', 'RSI_MAX']
        for col in param_cols:
            if col in df.columns:
                unique_values = df[col].nunique()
                most_common = df[col].mode().iloc[0] if len(df[col].mode()) > 0 else df[col].iloc[0]
                print(f"  {col}: {unique_values}个不同值, 最常见: {most_common}")

        print(f"\n表现相关性:")
        corr_return = df[['train_return', 'test_return']].corr().iloc[0, 1]
        corr_sharpe = df[['train_sharpe', 'test_sharpe']].corr().iloc[0, 1]
        print(f"  训练/测试收益率相关性: {corr_return:.3f}")
        print(f"  训练/测试夏普比率相关性: {corr_sharpe:.3f}")

        # 成功率统计
        success_count = len(df[df['test_return'] > 0])
        success_rate = success_count / len(df) * 100
        print(f"\n测试期盈利窗口: {success_count}/{len(df)} ({success_rate:.1f}%)")

        print(f"\n{'='*70}\n")

    def bayesian_optimization(self,
                           param_bounds: Dict[str, Tuple],
                           n_iterations: int = 50) -> Dict:
        """
        贝叶斯优化（简化版，使用随机搜索替代）
        实际项目建议使用 hyperopt 或 optuna

        参数:
            param_bounds: 参数边界
                {
                    'BREAKOUT_N': (20, 100),  # 整数参数
                    'MA_FAST': (5, 30),
                    'MA_SLOW': (20, 100),
                    'VOL_CONFIRM_MULT': (1.0, 3.0),  # 浮点参数
                    'RSI_MAX': (60, 90)
                }
            n_iterations: 迭代次数

        返回:
            最优参数和结果
        """
        print(f"\n{'='*70}")
        print(f"贝叶斯优化（简化版: 随机搜索）")
        print(f"{'='*70}")
        print(f"迭代次数: {n_iterations}")
        print(f"参数边界:")
        for name, bounds in param_bounds.items():
            print(f"  {name}: {bounds}")
        print(f"{'='*70}\n")

        import random

        best_score = -float('inf')
        best_params = None
        best_result = None

        for iteration in tqdm(range(n_iterations), desc="贝叶斯优化"):
            # 随机采样参数
            current_params = {}
            for param_name, bounds in param_bounds.items():
                if isinstance(bounds[0], int):
                    # 整数参数
                    current_params[param_name] = random.randint(bounds[0], bounds[1])
                else:
                    # 浮点参数
                    current_params[param_name] = random.uniform(bounds[0], bounds[1])

            # 更新参数
            for param_name, param_value in current_params.items():
                setattr(config, param_name, param_value)

            # 运行回测
            from strategy import StockStrategy
            basic_all = self.fetcher.get_stock_basic()
            strategy = StockStrategy(basic_all)

            engine = BacktestEngine(self.backtest_config, strategy, self.fetcher)
            result = engine.run()

            # 计算得分
            score = self._calculate_composite_score(result)

            if score > best_score:
                best_score = score
                best_params = current_params.copy()
                best_result = result
                print(f"\n  新最优参数 (迭代 {iteration+1}): 得分={score:.2f}")

        print(f"\n最优参数:")
        for param_name, param_value in best_params.items():
            print(f"  {param_name}: {param_value}")
        print(f"最优得分: {best_score:.2f}")
        print(f"年化收益: {best_result.get('annual_return', 0):.2f}%")
        print(f"夏普比率: {best_result.get('sharpe_ratio', 0):.2f}")

        return {
            'best_params': best_params,
            'best_score': best_score,
            'best_result': best_result
        }
