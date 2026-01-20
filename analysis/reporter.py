"""
趋势雷达选股系统 - 报告生成模块
负责生成Markdown格式的选股报告、回测报告、优化报告
"""
import pandas as pd
import numpy as np
from datetime import datetime
from config.settings import SAVE_REPORT, REPORT_DIR, TOP_N
from core.utils import ensure_dir


class Reporter:
    """报告生成器"""

    @staticmethod
    def render_markdown(trade_date: str, market_status: dict,
                       top_df: pd.DataFrame, excluded_stats: dict) -> str:
        """
        渲染Markdown格式的报告

        参数:
            trade_date: 交易日期
            market_status: 市场状态字典
            top_df: Top候选股票DataFrame
            excluded_stats: 排除统计信息

        返回:
            Markdown格式字符串
        """
        lines = []

        # 标题
        lines.append(f"# 趋势雷达选股报告")
        lines.append(f"")
        lines.append(f"**日期**: {trade_date}")
        lines.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"")
        lines.append(f"---")
        lines.append(f"")

        # 市场环境
        lines.append(f"## 市场环境")
        lines.append(f"")
        lines.append(f"- **指数MA20**: {market_status.get('ma20', 0):.2f}")
        lines.append(f"- **指数MA60**: {market_status.get('ma60', 0):.2f}")
        lines.append(f"- **20日波动率**: {market_status.get('vol20', 0)*100:.2f}%")
        lines.append(f"- **市场提示**: {market_status.get('hint', '')}")
        lines.append(f"")

        # 统计信息
        lines.append(f"## 选股统计")
        lines.append(f"")
        for key, value in excluded_stats.items():
            lines.append(f"- **{key}**: {value}")
        lines.append(f"")

        # 候选股票
        lines.append(f"## 候选股票 (Top {TOP_N})")
        lines.append(f"")

        if top_df.empty:
            lines.append(f"无符合条件的股票")
        else:
            candidates = top_df[top_df['candidate'] == True]
            watches = top_df[top_df['watch'] == True]

            # 候选股票表格
            if not candidates.empty:
                lines.append(f"### 推荐候选")
                lines.append(f"")
                lines.append(f"| 代码 | 名称 | 行业 | 收盘价 | 得分 | 止损价 | 距突破 |")
                lines.append(f"|------|------|------|--------|------|--------|--------|")
                for _, row in candidates.iterrows():
                    code = row['ts_code']
                    name = row.get('name', '')
                    industry = row.get('industry', '')
                    close = row['close']
                    score = row['score']
                    stop = row['stop_price']
                    dist = row.get('dist_to_break_pct', 0)

                    lines.append(f"| {code} | {name} | {industry} | {close:.2f} | {score:.1f} | {stop:.2f} | {dist:.2f}% |")
                lines.append(f"")

                # 详细信息
                for _, row in candidates.iterrows():
                    lines.append(f"#### {row['ts_code']} - {row.get('name', '')}")
                    lines.append(f"")
                    lines.append(f"- **行业**: {row.get('industry', '')}")
                    lines.append(f"- **收盘价**: {row['close']:.2f}")
                    lines.append(f"- **最高价**: {row['high_today']:.2f}")
                    lines.append(f"- **突破价**: {row['breakout_price']:.2f}")
                    lines.append(f"- **止损价**: {row['stop_price']:.2f}")
                    lines.append(f"- **得分**: {row['score']:.1f}")
                    lines.append(f"- **ADX**: {row.get('adx', 0):.1f}")
                    lines.append(f"- **KDJ-J**: {row.get('kdj_j', 0):.1f}")
                    lines.append(f"- **价格位置**: {row.get('price_position', 0)*100:.1f}%")
                    lines.append(f"")
                    lines.append(f"**理由**:")
                    for reason in row['reasons']:
                        lines.append(f"- {reason}")
                    lines.append(f"")

            # 观察股票
            if not watches.empty:
                lines.append(f"### 观察股票")
                lines.append(f"")
                lines.append(f"| 代码 | 名称 | 收盘价 | 得分 | 止损价 | 距突破 |")
                lines.append(f"|------|------|--------|------|--------|--------|")
                for _, row in watches.iterrows():
                    code = row['ts_code']
                    name = row.get('name', '')
                    close = row['close']
                    score = row['score']
                    stop = row['stop_price']
                    dist = row.get('dist_to_break_pct', 0)

                    lines.append(f"| {code} | {name} | {close:.2f} | {score:.1f} | {stop:.2f} | {dist:.2f}% |")
                lines.append(f"")

        # 免责声明
        lines.append(f"---")
        lines.append(f"")
        lines.append(f"**免责声明**: 本报告仅供参考，不构成投资建议。投资有风险，入市需谨慎。")
        lines.append(f"")

        return "\n".join(lines)

    @staticmethod
    def print_console(report: str, top_df: pd.DataFrame):
        """
        在控制台打印报告

        参数:
            report: Markdown报告字符串
            top_df: Top候选股票DataFrame
        """
        print("\n" + "="*70)
        print(report)
        print("="*70 + "\n")

        if not top_df.empty:
            print("\n【候选股票表格】")
            print("-" * 100)
            print(f"{'代码':<10} {'名称':<12} {'行业':<12} {'收盘价':>8} {'得分':>6} {'止损价':>8} {'距突破':>8}")
            print("-" * 100)

            candidates = top_df[top_df['candidate'] == True]
            for _, row in candidates.iterrows():
                code = row['ts_code']
                name = row.get('name', '')[:10]
                industry = row.get('industry', '')[:10]
                close = row['close']
                score = row['score']
                stop = row['stop_price']
                dist = row.get('dist_to_break_pct', 0)

                print(f"{code:<10} {name:<12} {industry:<12} {close:>8.2f} {score:>6.1f} {stop:>8.2f} {dist:>7.2f}%")

            print("-" * 100 + "\n")

    @staticmethod
    def render_backtest_summary(backtest_result: dict, holding_days: int) -> str:
        """
        渲染回测摘要

        参数:
            backtest_result: 回测结果字典
            holding_days: 持仓天数

        返回:
            Markdown格式字符串
        """
        lines = []
        lines.append(f"\n{'='*70}")
        lines.append(f"{'简单回测结果':^66}")
        lines.append(f"{'='*70}\n")
        lines.append(f"持仓天数: {holding_days} 天")
        lines.append(f"")

        if backtest_result['count'] == 0:
            lines.append(f"无回测数据（未来无交易日）")
        else:
            lines.append(f"交易数量: {backtest_result['count']}")
            lines.append(f"平均收益: {backtest_result['avg_return']:.2f}%")
            lines.append(f"胜率: {backtest_result['win_rate']:.2f}%")
            lines.append(f"最大盈利: {backtest_result['max_return']:.2f}%")
            lines.append(f"最大亏损: {backtest_result['min_return']:.2f}%")
            lines.append(f"")

            lines.append(f"详细交易记录:")
            lines.append(f"")
            lines.append(f"| 代码 | 入场价 | 出场价 | 收益率 |")
            lines.append(f"|------|--------|--------|--------|")
            for detail in backtest_result['details']:
                lines.append(f"| {detail['code']} | {detail['entry']:.2f} | {detail['exit']:.2f} | {detail['return']*100:.2f}% |")

        return "\n".join(lines)

    @staticmethod
    def save_report(trade_date: str, content: str, report_dir: str = REPORT_DIR):
        """
        保存报告到文件

        参数:
            trade_date: 交易日期
            content: 报告内容
            report_dir: 报告目录
        """
        if not SAVE_REPORT:
            return

        ensure_dir(report_dir)
        filename = f"trend_radar_{trade_date}.md"
        filepath = f"{report_dir}/{filename}"

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)

        print(f"报告已保存: {filepath}")

    @staticmethod
    def render_backtest_report(backtest_result: dict, backtest_config: object) -> str:
        """
        生成详细回测报告

        参数:
            backtest_result: 回测结果字典
            backtest_config: 回测配置对象

        返回:
            Markdown格式字符串
        """
        lines = []

        # 标题
        lines.append(f"# 历史回测报告")
        lines.append(f"")
        lines.append(f"**回测周期**: {backtest_config.start_date} ~ {backtest_config.end_date}")
        lines.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"")
        lines.append(f"---")
        lines.append(f"")

        # 回测配置
        lines.append(f"## 回测配置")
        lines.append(f"")
        lines.append(f"- **初始资金**: {backtest_config.initial_capital:,.0f} 元")
        lines.append(f"- **最大持仓数**: {backtest_config.max_positions}")
        lines.append(f"- **单只仓位**: {backtest_config.position_size*100:.1f}%")
        lines.append(f"- **滑点**: {backtest_config.slippage*100:.2f}%")
        lines.append(f"- **手续费率**: {backtest_config.commission*100:.3f}%")
        lines.append(f"- **止损**: {backtest_config.stop_loss*100:.1f}%")
        lines.append(f"- **止盈**: {backtest_config.take_profit*100:.1f}%")
        lines.append(f"- **最大持仓天数**: {backtest_config.max_holding_days}")
        lines.append(f"- **调仓周期**: {backtest_config.rebalance_days}天")
        lines.append(f"")

        # 收益指标
        lines.append(f"## 收益指标")
        lines.append(f"")
        lines.append(f"### 核心指标")
        lines.append(f"")
        lines.append(f"| 指标 | 数值 |")
        lines.append(f"|------|------|")
        lines.append(f"| 初始资金 | {backtest_config.initial_capital:,.0f} 元 |")
        lines.append(f"| 期末资金 | {backtest_result['final_equity']:,.0f} 元 |")
        lines.append(f"| 总收益率 | **{backtest_result['total_return']:.2f}%** |")
        lines.append(f"| 年化收益率 | **{backtest_result['annual_return']:.2f}%** |")
        lines.append(f"| 平均单笔收益 | {backtest_result['avg_return']:.2f}% |")
        lines.append(f"")
        lines.append(f"### 最佳/最差表现")
        lines.append(f"")
        lines.append(f"| 指标 | 数值 |")
        lines.append(f"|------|------|")
        lines.append(f"| 最大单笔盈利 | {backtest_result['max_profit']:.2f}% |")
        lines.append(f"| 最大单笔亏损 | {backtest_result['max_loss']:.2f}% |")
        lines.append(f"")

        # 风险指标
        lines.append(f"## 风险指标")
        lines.append(f"")
        lines.append(f"| 指标 | 数值 | 评级 |")
        lines.append(f"|------|------|------|")

        # 最大回撤评级
        dd = abs(backtest_result['max_drawdown'])
        if dd < 10:
            dd_rating = "优秀"
        elif dd < 20:
            dd_rating = "良好"
        elif dd < 30:
            dd_rating = "一般"
        else:
            dd_rating = "较差"

        lines.append(f"| 最大回撤 | {dd:.2f}% | {dd_rating} |")

        # 夏普比率评级
        sharpe = backtest_result['sharpe_ratio']
        if sharpe > 2:
            sharpe_rating = "优秀"
        elif sharpe > 1:
            sharpe_rating = "良好"
        elif sharpe > 0.5:
            sharpe_rating = "一般"
        else:
            sharpe_rating = "较差"

        lines.append(f"| 夏普比率 | {sharpe:.2f} | {sharpe_rating} |")
        lines.append(f"| Sortino比率 | {backtest_result['sortino_ratio']:.2f} | - |")
        lines.append(f"| Calmar比率 | {backtest_result['calmar_ratio']:.2f} | - |")
        lines.append(f"| 年化波动率 | {backtest_result['volatility']:.2f}% | - |")
        lines.append(f"")

        # 交易统计
        lines.append(f"## 交易统计")
        lines.append(f"")
        lines.append(f"| 指标 | 数值 |")
        lines.append(f"|------|------|")
        lines.append(f"| 总交易次数 | {backtest_result['total_trades']} |")
        lines.append(f"| 交易天数 | {backtest_result['trading_days']} |")

        # 胜率评级
        win_rate = backtest_result['win_rate']
        if win_rate > 60:
            win_rating = "优秀"
        elif win_rate > 50:
            win_rating = "良好"
        elif win_rate > 40:
            win_rating = "一般"
        else:
            win_rating = "较差"

        lines.append(f"| 胜率 | {win_rate:.2f}% ({win_rating}) |")
        lines.append(f"| 盈亏比 | {backtest_result['profit_factor']:.2f} |")
        lines.append(f"| 平均盈利 | {backtest_result['avg_win']:.2f}% |")
        lines.append(f"| 平均亏损 | {backtest_result['avg_loss']:.2f}% |")
        lines.append(f"")

        # 月度收益
        if backtest_result.get('equity_curve'):
            lines.append(f"## 资金曲线分析")
            lines.append(f"")
            equity_df = pd.DataFrame(backtest_result['equity_curve'])
            equity_df['date'] = pd.to_datetime(equity_df['date'], format='%Y%m%d')
            equity_df['month'] = equity_df['date'].dt.to_period('M')
            equity_df['monthly_return'] = equity_df['equity'].pct_change()

            monthly_returns = equity_df.groupby('month').last()
            monthly_returns['monthly_return'] = monthly_returns['equity'].pct_change() * 100

            lines.append(f"| 月份 | 资金 | 月收益率 |")
            lines.append(f"|------|------|----------|")
            for idx, row in monthly_returns.iterrows():
                month_str = str(idx)
                equity = row['equity']
                ret = row['monthly_return']
                lines.append(f"| {month_str} | {equity:,.0f} | {ret if pd.isna(ret) else ret:.2f}% |")
            lines.append(f"")

        # 交易记录
        if backtest_result.get('trades'):
            lines.append(f"## 交易记录")
            lines.append(f"")
            lines.append(f"| 代码 | 名称 | 入场日期 | 出场日期 | 入场价 | 出场价 | 收益率 | 原因 |")
            lines.append(f"|------|------|----------|----------|--------|--------|--------|------|")

            trades = backtest_result['trades'][:50]  # 只显示前50笔
            for trade in trades:
                lines.append(f"| {trade['ts_code']} | {trade['name']} | {trade['entry_date']} | {trade['exit_date']} | "
                           f"{trade['entry_price']:.2f} | {trade['exit_price']:.2f} | {trade['pnl_pct']*100:.2f}% | {trade['reason']} |")

            if len(backtest_result['trades']) > 50:
                lines.append(f"")
                lines.append(f"*仅显示前50笔交易，共{len(backtest_result['trades'])}笔*")
            lines.append(f"")

        # 退出原因统计
        if backtest_result.get('trades'):
            lines.append(f"## 退出原因分析")
            lines.append(f"")

            reason_counts = {}
            for trade in backtest_result['trades']:
                reason = trade['reason']
                if reason not in reason_counts:
                    reason_counts[reason] = 0
                reason_counts[reason] += 1

            reason_map = {
                'stop_loss': '止损',
                'take_profit': '止盈',
                'atr_stop': 'ATR止损',
                'breakeven': '保本止盈',
                'max_hold': '最大持仓天数',
                'end_backtest': '回测结束'
            }

            lines.append(f"| 退出原因 | 次数 | 占比 |")
            lines.append(f"|----------|------|------|")
            for reason, count in reason_counts.items():
                reason_text = reason_map.get(reason, reason)
                pct = count / len(backtest_result['trades']) * 100
                lines.append(f"| {reason_text} | {count} | {pct:.1f}% |")
            lines.append(f"")

        # 免责声明
        lines.append(f"---")
        lines.append(f"")
        lines.append(f"**免责声明**: 本回测结果仅供参考，不构成投资建议。实际交易受滑点、手续费、流动性等因素影响，可能与回测结果存在差异。投资有风险，入市需谨慎。")
        lines.append(f"")

        return "\n".join(lines)

    @staticmethod
    def render_optimization_report(optimization_df: pd.DataFrame,
                                   best_params: dict,
                                   optimization_method: str = "网格搜索") -> str:
        """
        生成参数优化报告

        参数:
            optimization_df: 优化结果DataFrame
            best_params: 最优参数
            optimization_method: 优化方法

        返回:
            Markdown格式字符串
        """
        lines = []

        # 标题
        lines.append(f"# 参数优化报告")
        lines.append(f"")
        lines.append(f"**优化方法**: {optimization_method}")
        lines.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"**参数组合数**: {len(optimization_df)}")
        lines.append(f"")
        lines.append(f"---")
        lines.append(f"")

        # 最优参数
        lines.append(f"## 最优参数")
        lines.append(f"")
        lines.append(f"| 参数 | 最优值 |")
        lines.append(f"|------|--------|")
        for param, value in best_params.items():
            lines.append(f"| {param} | {value} |")
        lines.append(f"")

        # 最优参数表现
        if not optimization_df.empty:
            best_row = optimization_df.iloc[0]
            lines.append(f"### 最优参数表现")
            lines.append(f"")
            lines.append(f"| 指标 | 数值 |")
            lines.append(f"|------|------|")
            lines.append(f"| 年化收益率 | {best_row['annual_return']:.2f}% |")
            lines.append(f"| 夏普比率 | {best_row['sharpe_ratio']:.2f} |")
            lines.append(f"| 最大回撤 | {best_row['max_drawdown']:.2f}% |")
            lines.append(f"| 胜率 | {best_row['win_rate']:.2f}% |")
            lines.append(f"| 盈亏比 | {best_row['profit_factor']:.2f} |")
            lines.append(f"| 交易次数 | {int(best_row['total_trades'])} |")
            lines.append(f"| 综合得分 | {best_row['score']:.2f} |")
            lines.append(f"")

        # Top 10 参数组合
        lines.append(f"## Top 10 参数组合")
        lines.append(f"")

        param_cols = [col for col in optimization_df.columns
                     if col not in ['score', 'total_return', 'annual_return', 'sharpe_ratio',
                                   'max_drawdown', 'win_rate', 'profit_factor', 'total_trades']]

        top10 = optimization_df.head(10)

        for idx, row in top10.iterrows():
            lines.append(f"### 排名 {idx+1}")
            lines.append(f"")
            lines.append(f"**参数配置**:")
            for col in param_cols:
                lines.append(f"- {col}: {row[col]}")
            lines.append(f"")
            lines.append(f"**表现指标**:")
            lines.append(f"- 年化收益: {row['annual_return']:.2f}%")
            lines.append(f"- 夏普比率: {row['sharpe_ratio']:.2f}")
            lines.append(f"- 最大回撤: {row['max_drawdown']:.2f}%")
            lines.append(f"- 胜率: {row['win_rate']:.2f}%")
            lines.append(f"- 综合得分: {row['score']:.2f}")
            lines.append(f"")

        # 参数敏感性分析
        lines.append(f"## 参数敏感性分析")
        lines.append(f"")

        for col in param_cols:
            if col in optimization_df.columns:
                col_data = optimization_df.groupby(col)['score'].agg(['mean', 'std', 'min', 'max']).sort_index()
                lines.append(f"### {col}")
                lines.append(f"")
                lines.append(f"| 参数值 | 平均得分 | 标准差 | 最小值 | 最大值 |")
                lines.append(f"|--------|----------|--------|--------|--------|")
                for idx, row in col_data.iterrows():
                    lines.append(f"| {idx} | {row['mean']:.2f} | {row['std']:.2f} | {row['min']:.2f} | {row['max']:.2f} |")
                lines.append(f"")

        # 统计摘要
        lines.append(f"## 统计摘要")
        lines.append(f"")
        lines.append(f"| 指标 | 数值 |")
        lines.append(f"|------|------|")
        lines.append(f"| 平均年化收益 | {optimization_df['annual_return'].mean():.2f}% |")
        lines.append(f"| 最高年化收益 | {optimization_df['annual_return'].max():.2f}% |")
        lines.append(f"| 最低年化收益 | {optimization_df['annual_return'].min():.2f}% |")
        lines.append(f"| 平均夏普比率 | {optimization_df['sharpe_ratio'].mean():.2f} |")
        lines.append(f"| 平均最大回撤 | {optimization_df['max_drawdown'].mean():.2f}% |")
        lines.append(f"| 平均胜率 | {optimization_df['win_rate'].mean():.2f}% |")
        lines.append(f"")

        # 相关性分析
        numeric_cols = ['annual_return', 'sharpe_ratio', 'max_drawdown', 'win_rate', 'profit_factor']
        corr_matrix = optimization_df[numeric_cols].corr()

        lines.append(f"## 指标相关性")
        lines.append(f"")
        lines.append(f"| | 收益 | 夏普 | 回撤 | 胜率 | 盈亏比 |")
        lines.append(f"|---|------|------|------|------|--------|")
        for idx in corr_matrix.index:
            row_str = f"| {idx[:4]} |"
            for col in corr_matrix.columns:
                val = corr_matrix.loc[idx, col]
                row_str += f" {val:.2f} |"
            lines.append(row_str)
        lines.append(f"")

        # 免责声明
        lines.append(f"---")
        lines.append(f"")
        lines.append(f"**免责声明**: 本优化结果基于历史数据计算，不保证未来表现。参数优化可能存在过拟合风险，建议使用Walk-Forward分析验证参数稳定性。投资有风险，入市需谨慎。")
        lines.append(f"")

        return "\n".join(lines)

    @staticmethod
    def render_walk_forward_report(wf_df: pd.DataFrame) -> str:
        """
        生成Walk-Forward分析报告

        参数:
            wf_df: Walk-Forward结果DataFrame

        返回:
            Markdown格式字符串
        """
        lines = []

        # 标题
        lines.append(f"# Walk-Forward分析报告")
        lines.append(f"")
        lines.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"**窗口数量**: {len(wf_df)}")
        lines.append(f"")
        lines.append(f"---")
        lines.append(f"")

        # 整体表现
        lines.append(f"## 整体表现")
        lines.append(f"")

        train_avg_return = wf_df['train_return'].mean()
        test_avg_return = wf_df['test_return'].mean()

        lines.append(f"### 训练期 vs 测试期")
        lines.append(f"")
        lines.append(f"| 指标 | 训练期 | 测试期 | 差异 |")
        lines.append(f"|------|--------|--------|------|")
        lines.append(f"| 平均年化收益 | {train_avg_return:.2f}% | {test_avg_return:.2f}% | {test_avg_return-train_avg_return:.2f}% |")
        lines.append(f"| 平均夏普比率 | {wf_df['train_sharpe'].mean():.2f} | {wf_df['test_sharpe'].mean():.2f} | {wf_df['test_sharpe'].mean()-wf_df['train_sharpe'].mean():.2f} |")
        lines.append(f"| 平均最大回撤 | {wf_df['train_drawdown'].mean():.2f}% | {wf_df['test_drawdown'].mean():.2f}% | {wf_df['test_drawdown'].mean()-wf_df['train_drawdown'].mean():.2f}% |")
        lines.append(f"")

        # 参数稳定性
        lines.append(f"## 参数稳定性")
        lines.append(f"")

        param_cols = ['BREAKOUT_N', 'MA_FAST', 'MA_SLOW', 'VOL_CONFIRM_MULT', 'RSI_MAX']
        for col in param_cols:
            if col in wf_df.columns:
                unique_values = wf_df[col].nunique()
                most_common = wf_df[col].mode().iloc[0] if len(wf_df[col].mode()) > 0 else wf_df[col].iloc[0]
                lines.append(f"- **{col}**: {unique_values}个不同值, 最常见: {most_common}")
        lines.append(f"")

        # 表现相关性
        lines.append(f"## 表现相关性")
        lines.append(f"")

        corr_return = wf_df[['train_return', 'test_return']].corr().iloc[0, 1]
        corr_sharpe = wf_df[['train_sharpe', 'test_sharpe']].corr().iloc[0, 1]

        lines.append(f"| 相关性 | 系数 |")
        lines.append(f"|--------|------|")
        lines.append(f"| 训练/测试收益率 | {corr_return:.3f} |")
        lines.append(f"| 训练/测试夏普比率 | {corr_sharpe:.3f} |")
        lines.append(f"")

        # 稳定性评估
        lines.append(f"## 稳定性评估")
        lines.append(f"")

        if corr_return > 0.5:
            stability = "优秀"
        elif corr_return > 0.3:
            stability = "良好"
        elif corr_return > 0:
            stability = "一般"
        else:
            stability = "较差"

        success_count = len(wf_df[wf_df['test_return'] > 0])
        success_rate = success_count / len(wf_df) * 100

        lines.append(f"| 指标 | 评估 |")
        lines.append(f"|------|------|")
        lines.append(f"| 表现相关性 | {stability} ({corr_return:.3f}) |")
        lines.append(f"| 盈利窗口比例 | {success_count}/{len(wf_df)} ({success_rate:.1f}%) |")
        lines.append(f"")

        # 窗口详细结果
        lines.append(f"## 窗口详细结果")
        lines.append(f"")
        lines.append(f"| 窗口 | 训练期 | 测试期 | 最优参数 | 测试收益 | 测试夏普 |")
        lines.append(f"|------|--------|--------|----------|----------|---------|")

        for _, row in wf_df.iterrows():
            train_period = f"{row['train_start']}~{row['train_end']}"
            test_period = f"{row['test_start']}~{row['test_end']}"
            params_str = f"BN:{int(row['BREAKOUT_N'])}, MF:{int(row['MA_FAST'])}, MS:{int(row['MA_SLOW'])}"

            lines.append(f"| {int(row['window'])} | {train_period} | {test_period} | {params_str} | "
                       f"{row['test_return']:.2f}% | {row['test_sharpe']:.2f} |")

        lines.append(f"")

        # 免责声明
        lines.append(f"---")
        lines.append(f"")
        lines.append(f"**免责声明**: Walk-Forward分析用于评估参数稳定性，但历史表现不保证未来收益。投资有风险，入市需谨慎。")
        lines.append(f"")

        return "\n".join(lines)
