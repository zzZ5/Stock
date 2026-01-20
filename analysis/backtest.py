"""
趋势雷达选股系统 - 回测系统模块
包含回测引擎、交易记录、风险管理等核心功能
"""
import pandas as pd
import numpy as np
from typing import Dict, List
from dataclasses import dataclass
from datetime import datetime
import os

from config.settings import (
    INDEX_CODE, MIN_LIST_DAYS, MIN_PRICE, MIN_AVG_AMOUNT_20D,
    EXCLUDE_ONE_WORD_LIMITUP, BREAKOUT_N, MA_FAST, MA_SLOW,
    VOL_LOOKBACK, VOL_CONFIRM_MULT, RSI_MAX, MAX_LOSS_PCT,
    ATR_N, ATR_MULT, TOP_N
)
from indicators.indicators import sma
from strategy.strategy import StockStrategy


@dataclass
class BacktestConfig:
    """回测配置"""
    start_date: str           # 回测开始日期 (YYYYMMDD)
    end_date: str             # 回测结束日期
    initial_capital: float    # 初始资金
    max_positions: int        # 最大持仓数量
    position_size: float      # 单只股票仓位比例
    slippage: float           # 滑点 (0.001 = 0.1%)
    commission: float         # 手续费率 (0.0003 = 0.03%)
    stop_loss: float          # 止损比例
    take_profit: float        # 止盈比例
    max_holding_days: int     # 最大持仓天数
    rebalance_days: int      # 重新选股间隔天数


@dataclass
class Trade:
    """单笔交易记录"""
    entry_date: str
    exit_date: str
    ts_code: str
    name: str
    entry_price: float
    exit_price: float
    shares: int
    pnl: float
    pnl_pct: float
    reason: str  # 'stop_loss', 'take_profit', 'signal', 'max_hold', 'end_backtest'


class BacktestEngine:
    """回测引擎"""

    def __init__(self, config: BacktestConfig, strategy: StockStrategy, fetcher):
        """
        初始化回测引擎

        参数:
            config: 回测配置
            strategy: 选股策略实例
            fetcher: 数据获取器实例
        """
        self.config = config
        self.strategy = strategy
        self.fetcher = fetcher
        self.trades: List[Trade] = []
        self.equity_curve: List[Dict] = []
        self.positions: Dict[str, Dict] = {}  # {ts_code: position_info}

    def run(self) -> Dict:
        """
        执行全量回测

        返回:
            回测结果字典，包含收益率、夏普比率、最大回撤等指标
        """
        print(f"\n{'='*60}")
        print(f"开始回测: {self.config.start_date} 至 {self.config.end_date}")
        print(f"初始资金: {self.config.initial_capital:,.0f} 元")
        print(f"{'='*60}\n")

        # 获取交易日历
        all_trade_dates = self.fetcher.get_trade_cal(
            end_date=self.config.end_date,
            lookback_calendar_days=1000
        )

        # 筛选回测日期范围
        backtest_dates = [
            d for d in all_trade_dates
            if self.config.start_date <= d <= self.config.end_date
        ]

        if not backtest_dates:
            print("错误: 没有可用的回测日期")
            return {}

        print(f"回测交易日数: {len(backtest_dates)}\n")

        # 初始化
        capital = self.config.initial_capital
        self.positions = {}
        self.trades = []
        self.equity_curve = []

        # 逐日回测
        for i, trade_date in enumerate(backtest_dates):
            # 进度显示
            if i % 20 == 0 or i == len(backtest_dates) - 1:
                print(f"[进度] {trade_date} ({i+1}/{len(backtest_dates)}) | "
                      f"现金: {capital:,.0f} | 持仓: {len(self.positions)}")

            # 1. 获取当日行情
            daily_df = self.fetcher.get_daily_by_date(trade_date)
            if daily_df.empty:
                continue

            # 2. 检查持仓止损/止盈/最大持仓天数
            self._check_positions(daily_df, trade_date, capital)

            # 3. 按周期重新选股（第一天必须选股）
            if i == 0 or i % self.config.rebalance_days == 0:
                self._run_stock_selection(trade_date, all_trade_dates, capital)

            # 4. 计算当日总资产
            equity = self._calculate_equity(daily_df, trade_date, capital)
            self.equity_curve.append(equity)

            # 更新现金（在平仓后）
            capital = equity['cash']

        # 回测结束，平掉所有持仓
        self._close_all_positions(backtest_dates[-1], capital)

        # 计算回测指标
        results = self._calculate_metrics()

        return results

    def _check_positions(self, daily_df: pd.DataFrame, trade_date: str, capital: float):
        """
        检查持仓，执行止损/止盈/最大持仓天数逻辑

        参数:
            daily_df: 当日行情数据
            trade_date: 交易日期
            capital: 当前现金
        """
        for ts_code in list(self.positions.keys()):
            position = self.positions[ts_code]
            current_data = daily_df[daily_df['ts_code'] == ts_code]

            if current_data.empty:
                continue

            current_price = float(current_data.iloc[0]['close'])

            # 计算收益率
            pnl_pct = (current_price - position['entry_price']) / position['entry_price']

            # 计算持仓天数（交易日）
            holding_days = self._calc_holding_days(position['entry_date'], trade_date)

            # 判断是否触发平仓条件
            exit_reason = None

            # 硬止损
            if pnl_pct <= self.config.stop_loss:
                exit_reason = 'stop_loss'
            # 止盈
            elif pnl_pct >= self.config.take_profit:
                exit_reason = 'take_profit'
            # 最大持仓天数
            elif holding_days >= self.config.max_holding_days:
                exit_reason = 'max_hold'

            if exit_reason:
                self._close_position(ts_code, current_price, trade_date, exit_reason, capital)

    def _run_stock_selection(self, trade_date: str, all_trade_dates: List[str], capital: float):
        """
        运行选股策略，开仓新股票

        参数:
            trade_date: 交易日期
            all_trade_dates: 所有交易日历
            capital: 当前现金
        """
        if len(self.positions) >= self.config.max_positions:
            return

        print(f"  -> 选股日期: {trade_date}")

        # 获取指数数据判断市场环境
        need_days = 120
        idx_hist = self.fetcher.get_index_window(INDEX_CODE, all_trade_dates, need_days)
        idx_hist = idx_hist.sort_values("trade_date")
        idx_close = idx_hist["close"].astype(float)

        market_ok = bool(sma(idx_close, 20).iloc[-1] > sma(idx_close, 60).iloc[-1])

        # 获取历史数据窗口
        daily_hist = self.fetcher.get_daily_window(all_trade_dates, 160)

        # 过滤基础股票信息
        basic_all = self.fetcher.get_stock_basic()
        basic = self.strategy.filter_basic(basic_all, trade_date=trade_date, trade_dates=all_trade_dates)
        daily_hist = daily_hist[daily_hist['ts_code'].isin(basic['ts_code'])].copy()

        # 运行策略
        top_stocks = self.strategy.analyze_stocks(daily_hist, market_ok)

        if top_stocks.empty:
            print("  -> 未找到符合条件的股票")
            return

        # 获取Top候选股票
        candidates = top_stocks[top_stocks['candidate'] == True].head(TOP_N)

        if candidates.empty:
            candidates = top_stocks.head(TOP_N)

        # 开仓
        opened_count = 0
        for _, stock in candidates.iterrows():
            if len(self.positions) >= self.config.max_positions:
                break

            if stock['ts_code'] not in self.positions:
                entry_price = stock['close']
                shares = int(capital * self.config.position_size / entry_price)

                if shares <= 0:
                    continue

                # 计算手续费和滑点
                price_with_slippage = entry_price * (1 + self.config.slippage)
                commission = price_with_slippage * shares * self.config.commission
                total_cost = price_with_slippage * shares + commission

                if total_cost > capital:
                    # 资金不足，减少股数
                    shares = int((capital * 0.99) / (price_with_slippage * (1 + self.config.commission)))
                    if shares <= 0:
                        continue
                    commission = price_with_slippage * shares * self.config.commission
                    total_cost = price_with_slippage * shares + commission

                capital -= total_cost

                self.positions[stock['ts_code']] = {
                    'entry_price': entry_price,
                    'entry_date': trade_date,
                    'shares': shares,
                    'name': stock.get('name', '')
                }

                print(f"  -> 开仓: {stock['ts_code']} {stock.get('name', '')} "
                      f"@ {entry_price:.2f} x {shares}股")

                opened_count += 1

        print(f"  -> 开仓数量: {opened_count}, 当前持仓: {len(self.positions)}")

    def _close_position(self, ts_code: str, exit_price: float,
                       exit_date: str, reason: str, capital: float):
        """
        平仓单只股票

        参数:
            ts_code: 股票代码
            exit_price: 平仓价格
            exit_date: 平仓日期
            reason: 平仓原因
            capital: 当前现金（用于更新）
        """
        position = self.positions.get(ts_code)
        if not position:
            return

        # 计算滑点后的价格
        price_with_slippage = exit_price * (1 - self.config.slippage)

        # 计算手续费
        commission = price_with_slippage * position['shares'] * self.config.commission

        # PnL计算
        gross_pnl = (exit_price - position['entry_price']) * position['shares']
        net_pnl = (price_with_slippage * position['shares'] - commission) - \
                  (position['entry_price'] * position['shares'])

        pnl_pct = net_pnl / (position['entry_price'] * position['shares'])

        # 更新现金
        capital += price_with_slippage * position['shares'] - commission

        # 记录交易
        trade = Trade(
            entry_date=position['entry_date'],
            exit_date=exit_date,
            ts_code=ts_code,
            name=position['name'],
            entry_price=position['entry_price'],
            exit_price=exit_price,
            shares=position['shares'],
            pnl=net_pnl,
            pnl_pct=pnl_pct,
            reason=reason
        )
        self.trades.append(trade)

        print(f"  -> 平仓: {ts_code} {position['name']} "
              f"@ {exit_price:.2f}, 收益: {pnl_pct*100:.2f}%, 原因: {reason}")

        # 移除持仓
        del self.positions[ts_code]

    def _close_all_positions(self, date: str, capital: float):
        """平掉所有持仓"""
        if not self.positions:
            return

        print(f"\n[平仓] 回测结束，平掉所有持仓...")

        daily_df = self.fetcher.get_daily_by_date(date)
        for ts_code in list(self.positions.keys()):
            position = self.positions[ts_code]
            current_data = daily_df[daily_df['ts_code'] == ts_code]

            if current_data.empty:
                continue

            exit_price = float(current_data.iloc[0]['close'])
            self._close_position(ts_code, exit_price, date, 'end_backtest', capital)

    def _calculate_equity(self, daily_df: pd.DataFrame, trade_date: str, cash: float) -> Dict:
        """
        计算当日总资产

        返回:
            {'date': str, 'cash': float, 'equity': float, 'positions_value': float}
        """
        positions_value = 0.0

        for ts_code, position in self.positions.items():
            current_data = daily_df[daily_df['ts_code'] == ts_code]
            if not current_data.empty:
                current_price = float(current_data.iloc[0]['close'])
                positions_value += current_price * position['shares']

        total_equity = cash + positions_value

        return {
            'date': trade_date,
            'cash': cash,
            'positions_value': positions_value,
            'equity': total_equity
        }

    def _calc_holding_days(self, entry_date: str, exit_date: str) -> int:
        """计算持仓天数（交易日）"""
        all_dates = self.fetcher.get_trade_cal(
            end_date=exit_date,
            lookback_calendar_days=1000
        )

        if entry_date not in all_dates or exit_date not in all_dates:
            return 0

        entry_idx = all_dates.index(entry_date)
        exit_idx = all_dates.index(exit_date)

        return exit_idx - entry_idx

    def _calculate_metrics(self) -> Dict:
        """
        计算回测指标

        返回:
            包含各种回测指标的字典
        """
        if not self.equity_curve or not self.trades:
            return {}

        # 转换为DataFrame方便计算
        equity_df = pd.DataFrame(self.equity_curve)
        trades_df = pd.DataFrame([t.__dict__ for t in self.trades])

        # 基础指标
        final_equity = equity_df['equity'].iloc[-1]
        total_return = (final_equity - self.config.initial_capital) / self.config.initial_capital

        # 年化收益率（按交易日252天计算）
        trading_days = len(equity_df)
        annual_return = (1 + total_return) ** (252 / trading_days) - 1 if trading_days > 0 else 0

        # 每日收益率
        daily_returns = equity_df['equity'].pct_change().dropna()

        # 夏普比率（假设无风险利率3%）
        if len(daily_returns) > 0:
            excess_returns = daily_returns - 0.03 / 252
            sharpe_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252) if excess_returns.std() > 0 else 0
        else:
            sharpe_ratio = 0

        # 最大回撤
        cumulative_returns = (1 + daily_returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = drawdown.min() if len(drawdown) > 0 else 0

        # 交易统计
        if len(trades_df) > 0:
            win_trades = trades_df[trades_df['pnl'] > 0]
            lose_trades = trades_df[trades_df['pnl'] <= 0]

            win_rate = len(win_trades) / len(trades_df) * 100
            avg_return = trades_df['pnl_pct'].mean() * 100
            avg_win = win_trades['pnl_pct'].mean() * 100 if len(win_trades) > 0 else 0
            avg_loss = lose_trades['pnl_pct'].mean() * 100 if len(lose_trades) > 0 else 0

            # 盈亏比
            profit_factor = abs(win_trades['pnl'].sum() / lose_trades['pnl'].sum()) if lose_trades['pnl'].sum() != 0 else float('inf')

            # 最大单笔盈利/亏损
            max_profit = trades_df['pnl_pct'].max() * 100
            max_loss = trades_df['pnl_pct'].min() * 100
        else:
            win_rate = 0
            avg_return = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
            max_profit = 0
            max_loss = 0

        # Calmar比率（年化收益/最大回撤绝对值）
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0

        # 波动率（年化）
        volatility = daily_returns.std() * np.sqrt(252) if len(daily_returns) > 0 else 0

        # Sortino比率（只考虑下行波动）
        downside_returns = daily_returns[daily_returns < 0]
        downside_std = downside_returns.std() if len(downside_returns) > 0 else 0
        sortino_ratio = (annual_return - 0.03) / (downside_std * np.sqrt(252)) if downside_std > 0 else 0

        results = {
            'total_return': total_return * 100,
            'annual_return': annual_return * 100,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown': max_drawdown * 100,
            'volatility': volatility * 100,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_return': avg_return,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'max_profit': max_profit,
            'max_loss': max_loss,
            'total_trades': len(trades_df),
            'trading_days': trading_days,
            'final_equity': final_equity,
            'trades': trades_df.to_dict('records') if len(trades_df) > 0 else [],
            'equity_curve': equity_df.to_dict('records')
        }

        return results

    def print_summary(self):
        """打印回测摘要"""
        if not self.equity_curve:
            print("\n无回测数据")
            return

        metrics = self._calculate_metrics()

        print(f"\n{'='*70}")
        print(f"{'回测结果摘要':^68}")
        print(f"{'='*70}\n")

        print(f"回测周期: {self.config.start_date} ~ {self.config.end_date} "
              f"({metrics['trading_days']}交易日)")
        print(f"初始资金: {self.config.initial_capital:,.0f} 元")
        print(f"期末资金: {metrics['final_equity']:,.0f} 元\n")

        print(f"--- 收益指标 ---")
        print(f"总收益率:     {metrics['total_return']:>10.2f}%")
        print(f"年化收益率:   {metrics['annual_return']:>10.2f}%")
        print(f"平均单笔:     {metrics['avg_return']:>10.2f}%")
        print(f"最大单笔盈利: {metrics['max_profit']:>10.2f}%")
        print(f"最大单笔亏损: {metrics['max_loss']:>10.2f}%\n")

        print(f"--- 风险指标 ---")
        print(f"最大回撤:     {metrics['max_drawdown']:>10.2f}%")
        print(f"年化波动率:   {metrics['volatility']:>10.2f}%")
        print(f"夏普比率:     {metrics['sharpe_ratio']:>10.2f}")
        print(f"Sortino比率:  {metrics['sortino_ratio']:>10.2f}")
        print(f"Calmar比率:   {metrics['calmar_ratio']:>10.2f}\n")

        print(f"--- 交易统计 ---")
        print(f"总交易次数:   {metrics['total_trades']:>10}")
        print(f"胜率:         {metrics['win_rate']:>10.2f}%")
        print(f"盈亏比:       {metrics['profit_factor']:>10.2f}")
        print(f"平均盈利:     {metrics['avg_win']:>10.2f}%")
        print(f"平均亏损:     {metrics['avg_loss']:>10.2f}%\n")

        print(f"{'='*70}\n")

    def save_results(self, filepath: str):
        """保存回测结果到CSV"""
        if self.trades:
            trades_df = pd.DataFrame([t.__dict__ for t in self.trades])
            trades_df.to_csv(filepath, index=False, encoding='utf-8-sig')
            print(f"交易记录已保存到: {filepath}")

        if self.equity_curve:
            equity_df = pd.DataFrame(self.equity_curve)
            equity_path = filepath.replace('_trades.csv', '_equity.csv')
            equity_df.to_csv(equity_path, index=False, encoding='utf-8-sig')
            print(f"资产曲线已保存到: {equity_path}")
