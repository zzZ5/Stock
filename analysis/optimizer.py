"""
趋势雷达选股系统 - 参数优化器模块
包含网格搜索、Walk-Forward分析等参数优化功能
"""
import pandas as pd
import numpy as np
from typing import Dict, List
import itertools
from tqdm import tqdm

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
