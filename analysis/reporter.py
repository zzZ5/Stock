"""
趋势雷达选股系统 - 报告生成模块
负责生成Markdown格式的选股报告
"""
import pandas as pd
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
