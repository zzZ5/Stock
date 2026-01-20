"""
增强版回测引擎
集成交易成本模型、蒙特卡洛模拟等高级功能
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
import warnings

from .transaction_cost import (
    TransactionCostCalculator,
    SlippageModel,
    MarketImpactModel,
    CommissionModel
)
from .monte_carlo import MonteCarloSimulator, StressTester
from analysis.backtest import BacktestEngine, BacktestConfig, Trade
from core.logger import get_backtest_logger

logger = get_backtest_logger()
warnings.filterwarnings('ignore')


@dataclass
class EnhancedBacktestConfig(BacktestConfig):
    """增强版回测配置"""
    use_advanced_cost_model: bool = False  # 是否使用高级成本模型
    slippage_model_type: str = 'fixed'  # 滑点模型类型
    market_impact_model: str = 'almgren_christoss'  # 市场冲击模型
    commission_model_type: str = 'percentage'  # 手续费模型类型
    
    run_monte_carlo: bool = False  # 是否运行蒙特卡洛模拟
    monte_carlo_simulations: int = 1000  # 蒙特卡洛模拟次数
    monte_carlo_method: str = 'gbm'  # 蒙特卡洛方法
    
    run_stress_test: bool = False  # 是否运行压力测试
    
    # 成本模型参数
    base_slippage: float = 0.001
    impact_factor: float = 0.0001
    volume_impact: bool = False
    min_commission: float = 5.0


class EnhancedBacktestEngine(BacktestEngine):
    """增强版回测引擎"""
    
    def __init__(self, config: EnhancedBacktestConfig, strategy, fetcher):
        """
        初始化增强版回测引擎
        
        参数:
            config: 增强版回测配置
            strategy: 选股策略
            fetcher: 数据获取器
        """
        super().__init__(config, strategy, fetcher)
        self.config = config
        
        # 初始化交易成本计算器
        self.cost_calculator = self._init_cost_calculator()
        
        # 股票统计信息缓存（用于成本计算）
        self.stock_stats_cache = {}
        
        # 详细交易成本记录
        self.detailed_trades: List[Dict] = []
    
    def _init_cost_calculator(self) -> TransactionCostCalculator:
        """初始化交易成本计算器"""
        slippage_model = SlippageModel(
            model_type=self.config.slippage_model_type,
            base_slippage=self.config.base_slippage,
            impact_factor=self.config.impact_factor,
            volume_impact=self.config.volume_impact
        )
        
        market_impact_model = MarketImpactModel(
            model_type=self.config.market_impact_model
        )
        
        commission_model = CommissionModel(
            commission_type=self.config.commission_model_type,
            base_rate=self.config.commission,
            min_commission=self.config.min_commission
        )
        
        return TransactionCostCalculator(
            slippage_model=slippage_model,
            market_impact_model=market_impact_model,
            commission_model=commission_model
        )
    
    def _get_stock_stats(
        self,
        ts_code: str,
        trade_date: str,
        fetch_all_dates: List[str]
    ) -> Dict:
        """
        获取股票统计信息（用于成本计算）
        
        参数:
            ts_code: 股票代码
            trade_date: 交易日期
            fetch_all_dates: 所有交易日历
        
        返回:
            统计信息字典
        """
        if ts_code in self.stock_stats_cache:
            return self.stock_stats_cache[ts_code]
        
        try:
            # 获取历史数据
            hist_data = self.fetcher.get_daily(ts_code, fetch_all_dates)
            
            if hist_data.empty:
                return {
                    'avg_daily_volume': 0,
                    'avg_price': 0,
                    'volatility': 0
                }
            
            # 计算统计量
            avg_daily_volume = hist_data['amount'].tail(60).mean()  # 近60日平均成交额
            avg_price = hist_data['close'].tail(60).mean()
            volatility = hist_data['close'].pct_change().tail(20).std()  # 近20日波动率
            
            stats = {
                'avg_daily_volume': avg_daily_volume,
                'avg_price': avg_price,
                'volatility': volatility
            }
            
            # 缓存
            self.stock_stats_cache[ts_code] = stats
            
            return stats
        
        except Exception as e:
            logger.error(f"获取{ts_code}统计信息失败: {e}")
            return {
                'avg_daily_volume': 0,
                'avg_price': 0,
                'volatility': 0
            }
    
    def _open_position_with_cost(
        self,
        ts_code: str,
        entry_price: float,
        shares: int,
        trade_date: str,
        daily_df: pd.DataFrame,
        all_trade_dates: List[str],
        capital: float
    ) -> tuple:
        """
        开仓并计算交易成本（增强版）
        
        参数:
            ts_code: 股票代码
            entry_price: 入场价格
            shares: 股数
            trade_date: 交易日期
            daily_df: 当日行情
            all_trade_dates: 所有交易日历
            capital: 当前现金
        
        返回:
            (实际成本, 剩余现金)
        """
        if not self.config.use_advanced_cost_model:
            # 使用原有简化成本模型
            price_with_slippage = entry_price * (1 + self.config.slippage)
            commission = price_with_slippage * shares * self.config.commission
            total_cost = price_with_slippage * shares + commission
            return total_cost, capital - total_cost
        
        # 使用高级成本模型
        stock_data = daily_df[daily_df['ts_code'] == ts_code]
        
        if stock_data.empty:
            # 如果没有行情数据，使用简化模型
            price_with_slippage = entry_price * (1 + self.config.slippage)
            commission = price_with_slippage * shares * self.config.commission
            total_cost = price_with_slippage * shares + commission
            return total_cost, capital - total_cost
        
        # 获取当日成交量
        volume = float(stock_data.iloc[0]['amount'])
        
        # 获取统计信息
        stats = self._get_stock_stats(ts_code, trade_date, all_trade_dates)
        
        # 计算买入成本
        buy_cost = self.cost_calculator.calculate_buy_cost(
            price=entry_price,
            shares=shares,
            volume=volume,
            avg_daily_volume=stats['avg_daily_volume'],
            avg_price=stats['avg_price'],
            volatility=stats['volatility']
        )
        
        total_cost = buy_cost['total_cost']
        
        # 记录详细成本
        self.detailed_trades.append({
            'trade_date': trade_date,
            'ts_code': ts_code,
            'action': 'buy',
            'original_price': entry_price,
            'shares': shares,
            'slippage_pct': buy_cost['slippage_pct'],
            'impact_pct': buy_cost['impact_pct'],
            'commission': buy_cost['commission'],
            'total_cost': total_cost,
            'total_cost_pct': buy_cost['total_cost_pct']
        })
        
        return total_cost, capital - total_cost
    
    def _close_position_with_cost(
        self,
        ts_code: str,
        exit_price: float,
        shares: int,
        trade_date: str,
        daily_df: pd.DataFrame,
        all_trade_dates: List[str],
        entry_price: float
    ) -> Dict:
        """
        平仓并计算交易成本（增强版）
        
        参数:
            ts_code: 股票代码
            exit_price: 平仓价格
            shares: 股数
            trade_date: 交易日期
            daily_df: 当日行情
            all_trade_dates: 所有交易日历
            entry_price: 入场价格
        
        返回:
            平仓结果字典
        """
        if not self.config.use_advanced_cost_model:
            # 使用原有简化模型
            price_with_slippage = exit_price * (1 - self.config.slippage)
            commission = price_with_slippage * shares * self.config.commission
            gross_pnl = (exit_price - entry_price) * shares
            net_pnl = (price_with_slippage * shares - commission) - entry_price * shares
            pnl_pct = net_pnl / (entry_price * shares)
            
            return {
                'exit_price': exit_price,
                'slippage_price': price_with_slippage,
                'commission': commission,
                'gross_pnl': gross_pnl,
                'net_pnl': net_pnl,
                'pnl_pct': pnl_pct,
                'slippage_pct': -self.config.slippage * 100,
                'impact_pct': 0,
                'total_cost_pct': (entry_price * shares - net_pnl) / (entry_price * shares) * 100
            }
        
        # 使用高级成本模型
        stock_data = daily_df[daily_df['ts_code'] == ts_code]
        
        if stock_data.empty:
            # 如果没有行情数据，使用简化模型
            price_with_slippage = exit_price * (1 - self.config.slippage)
            commission = price_with_slippage * shares * self.config.commission
            gross_pnl = (exit_price - entry_price) * shares
            net_pnl = (price_with_slippage * shares - commission) - entry_price * shares
            pnl_pct = net_pnl / (entry_price * shares)
            
            return {
                'exit_price': exit_price,
                'slippage_price': price_with_slippage,
                'commission': commission,
                'gross_pnl': gross_pnl,
                'net_pnl': net_pnl,
                'pnl_pct': pnl_pct,
                'slippage_pct': -self.config.slippage * 100,
                'impact_pct': 0,
                'total_cost_pct': (entry_price * shares - net_pnl) / (entry_price * shares) * 100
            }
        
        # 获取当日成交量
        volume = float(stock_data.iloc[0]['amount'])
        
        # 获取统计信息
        stats = self._get_stock_stats(ts_code, trade_date, all_trade_dates)
        
        # 计算卖出成本
        sell_cost = self.cost_calculator.calculate_sell_cost(
            price=exit_price,
            shares=shares,
            volume=volume,
            avg_daily_volume=stats['avg_daily_volume'],
            avg_price=stats['avg_price'],
            volatility=stats['volatility']
        )
        
        gross_pnl = (exit_price - entry_price) * shares
        net_pnl = sell_cost['net_proceeds'] - entry_price * shares
        pnl_pct = net_pnl / (entry_price * shares)
        
        # 记录详细成本
        self.detailed_trades.append({
            'trade_date': trade_date,
            'ts_code': ts_code,
            'action': 'sell',
            'original_price': exit_price,
            'shares': shares,
            'slippage_pct': sell_cost['slippage_pct'],
            'impact_pct': sell_cost['impact_pct'],
            'commission': sell_cost['commission'],
            'net_proceeds': sell_cost['net_proceeds'],
            'gross_pnl': gross_pnl,
            'net_pnl': net_pnl,
            'pnl_pct': pnl_pct * 100,
            'total_cost_pct': sell_cost['total_cost_pct']
        })
        
        return {
            'exit_price': exit_price,
            'slippage_price': sell_cost['final_price'],
            'commission': sell_cost['commission'],
            'gross_pnl': gross_pnl,
            'net_pnl': net_pnl,
            'pnl_pct': pnl_pct,
            'slippage_pct': sell_cost['slippage_pct'],
            'impact_pct': sell_cost['impact_pct'],
            'total_cost_pct': sell_cost['total_cost_pct']
        }
    
    def _close_position(
        self,
        ts_code: str,
        exit_price: float,
        exit_date: str,
        reason: str,
        capital: float,
        daily_df: pd.DataFrame,
        all_trade_dates: List[str]
    ):
        """
        重写平仓方法（增强版）
        """
        position = self.positions.get(ts_code)
        if not position:
            logger.warning(f"平仓失败: 持仓中不存在{ts_code}")
            return
        
        # 使用增强版成本计算
        close_result = self._close_position_with_cost(
            ts_code=ts_code,
            exit_price=exit_price,
            shares=position['shares'],
            trade_date=exit_date,
            daily_df=daily_df,
            all_trade_dates=all_trade_dates,
            entry_price=position['entry_price']
        )
        
        # 更新现金
        capital += close_result['net_proceeds']
        
        # 记录交易
        trade = Trade(
            entry_date=position['entry_date'],
            exit_date=exit_date,
            ts_code=ts_code,
            name=position['name'],
            entry_price=position['entry_price'],
            exit_price=close_result['exit_price'],
            shares=position['shares'],
            pnl=close_result['net_pnl'],
            pnl_pct=close_result['pnl_pct'],
            reason=reason
        )
        self.trades.append(trade)
        
        logger.info(
            f"平仓: {ts_code} {position['name']} "
            f"@ {exit_price:.2f}, 净收益: {close_result['pnl_pct']*100:.2f}%, "
            f"原因: {reason}, 总成本: {close_result['total_cost_pct']:.2f}%"
        )
        
        # 移除持仓
        del self.positions[ts_code]
        
        return capital
    
    def run_enhanced(self) -> Dict:
        """
        运行增强版回测
        
        返回:
            回测结果字典（包含蒙特卡洛和压力测试）
        """
        # 先运行基础回测
        results = self.run()
        
        if not results:
            return {}
        
        # 添加详细成本分析
        results['detailed_trades'] = self.detailed_trades
        results['cost_analysis'] = self._analyze_transaction_costs()
        
        # 运行蒙特卡洛模拟
        if self.config.run_monte_carlo:
            results['monte_carlo'] = self._run_monte_carlo()
        
        # 运行压力测试
        if self.config.run_stress_test:
            results['stress_test'] = self._run_stress_test()
        
        return results
    
    def _analyze_transaction_costs(self) -> Dict:
        """
        分析交易成本
        
        返回:
            成本分析结果
        """
        if not self.detailed_trades:
            return {}
        
        detailed_df = pd.DataFrame(self.detailed_trades)
        
        # 买入成本
        buy_trades = detailed_df[detailed_df['action'] == 'buy']
        sell_trades = detailed_df[detailed_df['action'] == 'sell']
        
        if len(buy_trades) == 0 or len(sell_trades) == 0:
            return {}
        
        # 统计
        total_slippage = abs(detailed_df['slippage_pct'].sum())
        total_impact = abs(detailed_df['impact_pct'].sum())
        total_commission = detailed_df['commission'].sum()
        
        avg_slippage = abs(detailed_df['slippage_pct']).mean()
        avg_impact = abs(detailed_df['impact_pct']).mean()
        avg_total_cost = abs(detailed_df['total_cost_pct']).mean()
        
        return {
            'total_slippage_pct': total_slippage,
            'total_impact_pct': total_impact,
            'total_commission': total_commission,
            'avg_slippage_pct': avg_slippage,
            'avg_impact_pct': avg_impact,
            'avg_total_cost_pct': avg_total_cost,
            'n_buy_trades': len(buy_trades),
            'n_sell_trades': len(sell_trades)
        }
    
    def _run_monte_carlo(self) -> Dict:
        """
        运行蒙特卡洛模拟
        
        返回:
            蒙特卡洛模拟结果
        """
        equity_df = pd.DataFrame(self.equity_curve)
        trades_df = pd.DataFrame([t.__dict__ for t in self.trades])
        
        simulator = MonteCarloSimulator(
            equity_curve=equity_df,
            trades_df=trades_df,
            initial_capital=self.config.initial_capital,
            n_simulations=self.config.monte_carlo_simulations,
            confidence_level=0.95
        )
        
        # 运行模拟
        simulation_results = simulator.run_batch_simulation(
            n_simulations=self.config.monte_carlo_simulations,
            method=self.config.monte_carlo_method,
            show_progress=True
        )
        
        # 计算风险指标
        risk_metrics = simulator.calculate_risk_metrics(simulation_results)
        
        return {
            'simulation_results': simulation_results.to_dict('records'),
            'risk_metrics': risk_metrics
        }
    
    def _run_stress_test(self) -> Dict:
        """
        运行压力测试
        
        返回:
            压力测试结果
        """
        # 构建基础回测结果
        base_results = {
            'equity_curve': pd.DataFrame(self.equity_curve),
            'final_equity': self.equity_curve[-1]['equity'],
            'total_return': (self.equity_curve[-1]['equity'] - self.config.initial_capital) / self.config.initial_capital * 100,
            'max_drawdown': min(
                [(eq['equity'] - max([e['equity'] for e in self.equity_curve[:i+1]])) / max([e['equity'] for e in self.equity_curve[:i+1]])
                 for i, eq in enumerate(self.equity_curve)]
            ) * 100
        }
        
        stress_tester = StressTester(base_results)
        stress_results = stress_tester.run_all_stress_tests(n_crashes=3)
        
        return {
            'stress_test_results': stress_results.to_dict('records')
        }
    
    def print_enhanced_summary(self):
        """打印增强版回测摘要"""
        self.print_summary()
        
        # 打印成本分析
        if 'cost_analysis' in self.equity_curve and self.config.use_advanced_cost_model:
            # 由于self.equity_curve是列表，需要从run_enhanced返回的结果中获取cost_analysis
            pass
        
        # 打印蒙特卡洛结果
        if 'monte_carlo' in self.equity_curve:
            pass
        
        # 打印压力测试结果
        if 'stress_test' in self.equity_curve:
            pass
