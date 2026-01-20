"""
趋势雷达选股系统 - 图表绘制模块
提供各种类型的股票和回测图表绘制功能
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Union
import warnings
import platform

warnings.filterwarnings('ignore')

# 设置中文字体 - 支持多平台
def setup_chinese_font():
    """设置中文字体"""
    system = platform.system()
    if system == 'Windows':
        font_list = ['SimHei', 'Microsoft YaHei', 'SimSun', 'KaiTi']
    elif system == 'Darwin':  # macOS
        font_list = ['Heiti TC', 'STHeiti', 'Arial Unicode MS']
    else:  # Linux
        font_list = ['WenQuanYi Micro Hei', 'Droid Sans Fallback', 'DejaVu Sans']

    for font in font_list:
        try:
            plt.rcParams['font.sans-serif'] = [font] + plt.rcParams['font.sans-serif']
            break
        except:
            continue

setup_chinese_font()
plt.rcParams['axes.unicode_minus'] = False


class Plotter:
    """图表绘制器"""

    @staticmethod
    def setup_style(style: str = 'seaborn-v0_8-darkgrid'):
        """设置绘图风格"""
        try:
            plt.style.use(style)
        except:
            plt.style.use('seaborn-v0_8-darkgrid')

    @staticmethod
    def save_figure(fig: plt.Figure, filepath: str, dpi: int = 150):
        """保存图表"""
        fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        return filepath


def plot_stock_candlestick(
    df: pd.DataFrame,
    title: str = "股票K线图",
    indicators: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (14, 10),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    绘制股票K线图

    参数:
        df: 股票数据，包含 open, high, low, close, volume 列
        title: 图表标题
        indicators: 要绘制的指标列表 ['ma', 'bollinger', 'volume']
        figsize: 图表大小
        save_path: 保存路径(可选)

    返回:
        matplotlib Figure对象
    """
    if indicators is None:
        indicators = ['ma', 'volume']

    # 确保数据包含所需列
    required_cols = ['open', 'high', 'low', 'close']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"DataFrame缺少必需列: {col}")

    df = df.copy()
    df.index = pd.to_datetime(df.index)

    Plotter.setup_style()

    # 根据指标数量调整子图布局
    if 'volume' in indicators:
        fig, axes = plt.subplots(3, 1, figsize=figsize, gridspec_kw={'height_ratios': [3, 1, 1]})
    else:
        fig, axes = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'height_ratios': [3, 1]})

    fig.suptitle(title, fontsize=16, fontweight='bold')

    # 绘制K线图
    ax1 = axes[0]

    # 计算上涨和下跌
    up = df[df['close'] >= df['open']]
    down = df[df['close'] < df['open']]

    # 绘制K线
    width = 0.6
    width2 = 0.1

    for idx, row in df.iterrows():
        date_num = mdates.date2num(idx)
        color = 'red' if row['close'] >= row['open'] else 'green'

        # 绘制影线
        ax1.plot([date_num, date_num], [row['low'], row['high']],
                 color=color, linewidth=1, alpha=0.8)

        # 绘制实体
        if color == 'red':
            ax1.bar(date_num, row['close'] - row['open'], width, bottom=row['open'],
                    color='red', alpha=0.8, edgecolor='red')
        else:
            ax1.bar(date_num, row['open'] - row['close'], width, bottom=row['close'],
                    color='green', alpha=0.8, edgecolor='green')

    # 绘制移动平均线
    if 'ma' in indicators:
        if 'ma5' not in df.columns:
            df['ma5'] = df['close'].rolling(window=5).mean()
        if 'ma10' not in df.columns:
            df['ma10'] = df['close'].rolling(window=10).mean()
        if 'ma20' not in df.columns:
            df['ma20'] = df['close'].rolling(window=20).mean()
        if 'ma60' not in df.columns:
            df['ma60'] = df['close'].rolling(window=60).mean()

        ax1.plot(df.index, df['ma5'], label='MA5', linewidth=1, alpha=0.8, color='orange')
        ax1.plot(df.index, df['ma10'], label='MA10', linewidth=1, alpha=0.8, color='purple')
        ax1.plot(df.index, df['ma20'], label='MA20', linewidth=1.2, alpha=0.8, color='blue')
        ax1.plot(df.index, df['ma60'], label='MA60', linewidth=1.5, alpha=0.8, color='brown')

    # 绘制布林带
    if 'bollinger' in indicators:
        if 'bb_upper' not in df.columns:
            df['bb_middle'] = df['close'].rolling(window=20).mean()
            std = df['close'].rolling(window=20).std()
            df['bb_upper'] = df['bb_middle'] + 2 * std
            df['bb_lower'] = df['bb_middle'] - 2 * std

        ax1.fill_between(df.index, df['bb_upper'], df['bb_lower'],
                        alpha=0.2, color='gray', label='布林带')

    ax1.set_ylabel('价格', fontsize=10)
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # 绘制成交量
    if 'volume' in indicators:
        ax2 = axes[1] if len(axes) > 2 else axes[1]
        colors = ['red' if df['close'].iloc[i] >= df['open'].iloc[i] else 'green'
                 for i in range(len(df))]
        ax2.bar(df.index, df['volume'].values, width=0.8, alpha=0.6, color=colors)
        ax2.set_ylabel('成交量', fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # 如果有第三个子图,绘制MACD或其他指标
        if len(axes) > 2 and 'macd' in indicators:
            ax3 = axes[2]
            if 'macd' not in df.columns:
                df['ema12'] = df['close'].ewm(span=12).mean()
                df['ema26'] = df['close'].ewm(span=26).mean()
                df['dif'] = df['ema12'] - df['ema26']
                df['dea'] = df['dif'].ewm(span=9).mean()
                df['macd'] = (df['dif'] - df['dea']) * 2

            ax3.bar(df.index, df['macd'].values, alpha=0.5, color='orange', label='MACD')
            ax3.plot(df.index, df['dif'], label='DIF', linewidth=1, color='red')
            ax3.plot(df.index, df['dea'], label='DEA', linewidth=1, color='blue')
            ax3.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
            ax3.set_ylabel('MACD', fontsize=10)
            ax3.legend(loc='upper left', fontsize=9)
            ax3.grid(True, alpha=0.3)
            ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
            plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()

    if save_path:
        Plotter.save_figure(fig, save_path)

    return fig


def plot_stock_indicators(
    df: pd.DataFrame,
    indicators: List[str] = ['rsi', 'kdj', 'cci', 'atr'],
    figsize: Tuple[int, int] = (14, 10),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    绘制技术指标图表

    参数:
        df: 股票数据，包含 close, high, low 列
        indicators: 要绘制的指标列表
        figsize: 图表大小
        save_path: 保存路径(可选)

    返回:
        matplotlib Figure对象
    """
    Plotter.setup_style()

    # 计算指标
    df = df.copy()

    # RSI
    if 'rsi' in indicators:
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

    # KDJ
    if 'kdj' in indicators:
        low_min = df['low'].rolling(window=9).min()
        high_max = df['high'].rolling(window=9).max()
        rsv = (df['close'] - low_min) / (high_max - low_min) * 100
        df['k'] = rsv.ewm(com=2).mean()
        df['d'] = df['k'].ewm(com=2).mean()
        df['j'] = 3 * df['k'] - 2 * df['d']

    # CCI
    if 'cci' in indicators:
        tp = (df['high'] + df['low'] + df['close']) / 3
        ma_tp = tp.rolling(window=14).mean()
        md = tp.rolling(window=14).apply(lambda x: np.fabs(x - x.mean()).mean())
        df['cci'] = (tp - ma_tp) / (0.015 * md)

    # ATR
    if 'atr' in indicators:
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = tr.rolling(window=14).mean()

    # 创建子图
    n_plots = len(indicators)
    fig, axes = plt.subplots(n_plots, 1, figsize=figsize, sharex=True)
    if n_plots == 1:
        axes = [axes]

    fig.suptitle('技术指标分析', fontsize=16, fontweight='bold')

    for i, indicator in enumerate(indicators):
        ax = axes[i]

        if indicator == 'rsi':
            ax.plot(df.index, df['rsi'], linewidth=1.5, color='purple')
            ax.axhline(y=70, color='red', linestyle='--', linewidth=1, alpha=0.7, label='超买线(70)')
            ax.axhline(y=30, color='green', linestyle='--', linewidth=1, alpha=0.7, label='超卖线(30)')
            ax.fill_between(df.index, 70, 100, alpha=0.1, color='red')
            ax.fill_between(df.index, 0, 30, alpha=0.1, color='green')
            ax.set_ylabel('RSI', fontsize=10)
            ax.set_ylim(0, 100)
            ax.legend(loc='upper right', fontsize=9)

        elif indicator == 'kdj':
            ax.plot(df.index, df['k'], linewidth=1.5, color='orange', label='K')
            ax.plot(df.index, df['d'], linewidth=1.5, color='blue', label='D')
            ax.plot(df.index, df['j'], linewidth=1.5, color='purple', label='J')
            ax.axhline(y=80, color='red', linestyle='--', linewidth=1, alpha=0.7)
            ax.axhline(y=20, color='green', linestyle='--', linewidth=1, alpha=0.7)
            ax.fill_between(df.index, 80, 100, alpha=0.1, color='red')
            ax.fill_between(df.index, 0, 20, alpha=0.1, color='green')
            ax.set_ylabel('KDJ', fontsize=10)
            ax.set_ylim(0, 100)
            ax.legend(loc='upper right', fontsize=9)

        elif indicator == 'cci':
            ax.plot(df.index, df['cci'], linewidth=1.5, color='darkgreen')
            ax.axhline(y=100, color='red', linestyle='--', linewidth=1, alpha=0.7, label='超买线(100)')
            ax.axhline(y=-100, color='green', linestyle='--', linewidth=1, alpha=0.7, label='超卖线(-100)')
            ax.fill_between(df.index, 100, 300, alpha=0.1, color='red')
            ax.fill_between(df.index, -300, -100, alpha=0.1, color='green')
            ax.set_ylabel('CCI', fontsize=10)
            ax.legend(loc='upper right', fontsize=9)

        elif indicator == 'atr':
            ax.plot(df.index, df['atr'], linewidth=1.5, color='darkblue', label='ATR(14)')
            ax.fill_between(df.index, 0, df['atr'], alpha=0.3, color='blue')
            ax.set_ylabel('ATR', fontsize=10)
            ax.legend(loc='upper right', fontsize=9)

        ax.grid(True, alpha=0.3)

    # 设置x轴
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()

    if save_path:
        Plotter.save_figure(fig, save_path)

    return fig


def plot_backtest_results(
    results: Dict,
    figsize: Tuple[int, int] = (16, 12),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    绘制回测结果图表

    参数:
        results: 回测结果字典
        figsize: 图表大小
        save_path: 保存路径(可选)

    返回:
        matplotlib Figure对象
    """
    Plotter.setup_style()

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    fig.suptitle('回测结果分析', fontsize=18, fontweight='bold')

    # 1. 净值曲线
    ax1 = fig.add_subplot(gs[0, :])

    if 'equity_curve' in results:
        equity_df = results['equity_curve']
        ax1.plot(equity_df.index, equity_df['equity'].values,
                linewidth=2, color='blue', label='策略净值')

        if 'benchmark_equity' in equity_df.columns:
            ax1.plot(equity_df.index, equity_df['benchmark_equity'].values,
                    linewidth=2, color='orange', linestyle='--', label='基准净值')

    ax1.set_title('净值曲线', fontsize=12, fontweight='bold')
    ax1.set_ylabel('净值', fontsize=10)
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # 2. 回撤曲线
    ax2 = fig.add_subplot(gs[1, :])

    if 'equity_curve' in results:
        equity_df = results['equity_curve']
        if 'drawdown' in equity_df.columns:
            ax2.fill_between(equity_df.index, equity_df['drawdown'].values, 0,
                            alpha=0.3, color='red')
            ax2.plot(equity_df.index, equity_df['drawdown'].values,
                    linewidth=1.5, color='darkred', label='回撤')

    ax2.set_title('回撤曲线', fontsize=12, fontweight='bold')
    ax2.set_ylabel('回撤 (%)', fontsize=10)
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # 3. 每月收益热力图
    ax3 = fig.add_subplot(gs[2, 0])

    if 'monthly_returns' in results:
        monthly_df = results['monthly_returns']
        # 转换为年-月格式
        monthly_df['year'] = monthly_df.index.year
        monthly_df['month'] = monthly_df.index.month

        # 创建透视表
        pivot_df = monthly_df.pivot(index='year', columns='month', values='returns')

        if not pivot_df.empty:
            import seaborn as sns
            sns.heatmap(pivot_df, annot=True, fmt='.2f', cmap='RdYlGn',
                       center=0, ax=ax3, cbar_kws={'label': '收益率(%)'})

    ax3.set_title('每月收益率热力图', fontsize=12, fontweight='bold')
    ax3.set_xlabel('月份', fontsize=10)
    ax3.set_ylabel('年份', fontsize=10)

    # 4. 交易分布
    ax4 = fig.add_subplot(gs[2, 1])

    if 'trades' in results:
        trades_df = results['trades']
        if len(trades_df) > 0:
            if 'pnl_pct' in trades_df.columns:
                ax4.hist(trades_df['pnl_pct'].values, bins=30, alpha=0.7,
                        color='steelblue', edgecolor='black')
                ax4.axvline(x=0, color='red', linestyle='--', linewidth=2)
                ax4.set_title('交易收益率分布', fontsize=12, fontweight='bold')
                ax4.set_xlabel('收益率 (%)', fontsize=10)
                ax4.set_ylabel('频次', fontsize=10)
                ax4.grid(True, alpha=0.3)

    if save_path:
        Plotter.save_figure(fig, save_path)

    return fig


def plot_drawdown_chart(
    equity_curve: pd.DataFrame,
    figsize: Tuple[int, int] = (14, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    绘制详细的回撤图

    参数:
        equity_curve: 包含 equity 和 drawdown 列的 DataFrame
        figsize: 图表大小
        save_path: 保存路径(可选)

    返回:
        matplotlib Figure对象
    """
    Plotter.setup_style()

    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
    fig.suptitle('净值与回撤分析', fontsize=16, fontweight='bold')

    # 净值曲线
    ax1 = axes[0]
    ax1.plot(equity_curve.index, equity_curve['equity'].values,
            linewidth=2, color='blue', label='净值')
    ax1.set_ylabel('净值', fontsize=10)
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # 回撤曲线
    ax2 = axes[1]
    ax2.fill_between(equity_curve.index, equity_curve['drawdown'].values, 0,
                    alpha=0.4, color='red')
    ax2.plot(equity_curve.index, equity_curve['drawdown'].values,
            linewidth=1.5, color='darkred', label='回撤')
    ax2.set_ylabel('回撤 (%)', fontsize=10)
    ax2.set_xlabel('日期', fontsize=10)
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3)

    # 标注最大回撤
    if 'drawdown' in equity_curve.columns:
        max_dd_idx = equity_curve['drawdown'].idxmax()
        max_dd_value = equity_curve['drawdown'].max()
        ax2.annotate(f'最大回撤: {max_dd_value:.2f}%',
                    xy=(max_dd_idx, max_dd_value),
                    xytext=(max_dd_idx, max_dd_value + 5),
                    arrowprops=dict(arrowstyle='->', color='red', lw=2),
                    fontsize=10, fontweight='bold', color='darkred',
                    ha='center')

    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()

    if save_path:
        Plotter.save_figure(fig, save_path)

    return fig


def plot_monthly_returns(
    monthly_returns: pd.DataFrame,
    figsize: Tuple[int, int] = (14, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    绘制每月收益率图表

    参数:
        monthly_returns: 包含 returns 列的 DataFrame,索引为日期
        figsize: 图表大小
        save_path: 保存路径(可选)

    返回:
        matplotlib Figure对象
    """
    Plotter.setup_style()

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    fig.suptitle('月度收益分析', fontsize=16, fontweight='bold')

    # 1. 月度收益率柱状图
    ax1 = fig.add_subplot(gs[0, :])
    colors = ['green' if x >= 0 else 'red' for x in monthly_returns['returns'].values]
    ax1.bar(range(len(monthly_returns)), monthly_returns['returns'].values,
           color=colors, alpha=0.7, edgecolor='black')
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax1.set_title('月度收益率', fontsize=12, fontweight='bold')
    ax1.set_ylabel('收益率 (%)', fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')

    # 设置x轴标签
    months = [f"{d.year}-{d.month:02d}" for d in monthly_returns.index]
    ax1.set_xticks(range(len(monthly_returns)))
    ax1.set_xticklabels(months, rotation=45, ha='right')

    # 2. 年度收益率
    ax2 = fig.add_subplot(gs[1, 0])
    monthly_returns['year'] = monthly_returns.index.year
    yearly_returns = monthly_returns.groupby('year')['returns'].sum()

    colors = ['green' if x >= 0 else 'red' for x in yearly_returns.values]
    ax2.bar(range(len(yearly_returns)), yearly_returns.values,
           color=colors, alpha=0.7, edgecolor='black')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax2.set_title('年度收益率', fontsize=12, fontweight='bold')
    ax2.set_ylabel('收益率 (%)', fontsize=10)
    ax2.set_xticks(range(len(yearly_returns)))
    ax2.set_xticklabels(yearly_returns.index, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3, axis='y')

    # 3. 收益率统计
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axis('off')

    stats_text = f"""
    统计指标

    平均月收益: {monthly_returns['returns'].mean():.2f}%
    月收益标准差: {monthly_returns['returns'].std():.2f}%
    最大月收益: {monthly_returns['returns'].max():.2f}%
    最小月收益: {monthly_returns['returns'].min():.2f}%

    盈利月份: {(monthly_returns['returns'] > 0).sum()} / {len(monthly_returns)}
    胜率: {(monthly_returns['returns'] > 0).mean() * 100:.1f}%

    夏普比率: {monthly_returns['returns'].mean() / monthly_returns['returns'].std() * np.sqrt(12):.2f}
    """
    ax3.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
            verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()

    if save_path:
        Plotter.save_figure(fig, save_path)

    return fig


def plot_parameter_heatmap(
    results: pd.DataFrame,
    param1: str,
    param2: str,
    metric: str = 'total_return',
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    绘制参数优化热力图

    参数:
        results: 参数优化结果 DataFrame
        param1: x轴参数
        param2: y轴参数
        metric: 评估指标
        figsize: 图表大小
        save_path: 保存路径(可选)

    返回:
        matplotlib Figure对象
    """
    Plotter.setup_style()

    fig, ax = plt.subplots(figsize=figsize)

    # 创建透视表
    pivot_df = results.pivot(index=param2, columns=param1, values=metric)

    # 绘制热力图
    import seaborn as sns
    sns.heatmap(pivot_df, annot=True, fmt='.2f', cmap='RdYlGn',
               center=0, ax=ax, cbar_kws={'label': metric})

    ax.set_title(f'参数优化热力图 - {metric}', fontsize=14, fontweight='bold')
    ax.set_xlabel(param1, fontsize=12)
    ax.set_ylabel(param2, fontsize=12)

    plt.tight_layout()

    if save_path:
        Plotter.save_figure(fig, save_path)

    return fig


def plot_parameter_sensitivity(
    results: pd.DataFrame,
    params: List[str],
    metric: str = 'total_return',
    figsize: Tuple[int, int] = (14, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    绘制参数敏感性分析图

    参数:
        results: 参数优化结果 DataFrame
        params: 要分析的参数列表
        metric: 评估指标
        figsize: 图表大小
        save_path: 保存路径(可选)

    返回:
        matplotlib Figure对象
    """
    Plotter.setup_style()

    n_params = len(params)
    fig, axes = plt.subplots(1, n_params, figsize=figsize, sharey=True)
    if n_params == 1:
        axes = [axes]

    fig.suptitle(f'参数敏感性分析 - {metric}', fontsize=14, fontweight='bold')

    for i, param in enumerate(params):
        ax = axes[i]

        # 计算每个参数值的平均指标
        param_stats = results.groupby(param)[metric].agg(['mean', 'std'])
        param_stats = param_stats.sort_index()

        # 绘制均值线
        x_pos = np.arange(len(param_stats))
        ax.errorbar(x_pos, param_stats['mean'], yerr=param_stats['std'],
                   fmt='o-', linewidth=2, markersize=8, capsize=5,
                   color='steelblue', label='均值±标准差')

        ax.set_title(param, fontsize=11, fontweight='bold')
        ax.set_xlabel(param, fontsize=10)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(param_stats.index, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')

        if i == 0:
            ax.set_ylabel(metric, fontsize=10)
            ax.legend(fontsize=9)

    plt.tight_layout()

    if save_path:
        Plotter.save_figure(fig, save_path)

    return fig
