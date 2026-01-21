"""
向量化回测引擎 - 优化版本
使用NumPy和pandas向量化操作提升性能
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp

from config.config_loader import load_config
from analysis.backtest import BacktestConfig, Trade, BacktestEngine
from indicators.indicators import sma, atr
from core.logger import get_backtest_logger

logger = get_backtest_logger()

# 全局配置
config = load_config()
INDEX_CODE = config.get('INDEX_CODE', '000300.SH')
TOP_N = config.get('TOP_N', 20)
ATR_N = config.get('ATR_N', 14)
ATR_MULT = config.get('ATR_MULT', 2.5)


class VectorizedMetrics:
    """向量化指标计算器"""
    
    @staticmethod
    def calculate_all_metrics_vectorized(
        equity_curve: pd.DataFrame,
        trades_df: pd.DataFrame,
        initial_capital: float
    ) -> Dict:
        """
        向量化计算所有回测指标
        
        Args:
            equity_curve: 资产曲线DataFrame
            trades_df: 交易记录DataFrame
            initial_capital: 初始资金
        
        Returns:
            包含所有指标的字典
        """
        if equity_curve.empty or trades_df.empty:
            return VectorizedMetrics._empty_metrics()
        
        # 基础指标（向量化）
        final_equity = equity_curve['equity'].iloc[-1]
        total_return = (final_equity - initial_capital) / initial_capital
        
        # 每日收益率（向量化）
        daily_returns = equity_curve['equity'].pct_change().dropna()
        
        # 年化收益率（向量化计算）
        trading_days = len(equity_curve)
        annual_return = (1 + total_return) ** (252 / trading_days) - 1 if trading_days > 0 else 0
        
        # 夏普比率（向量化）
        if len(daily_returns) > 0:
            excess_returns = daily_returns - 0.03 / 252
            sharpe_ratio = np.sqrt(252) * excess_returns.mean() / excess_returns.std() if excess_returns.std() > 0 else 0
        else:
            sharpe_ratio = 0
        
        # 最大回撤（向量化）
        cumulative_returns = (1 + daily_returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = drawdown.min() if len(drawdown) > 0 else 0
        
        # 波动率（向量化）
        volatility = daily_returns.std() * np.sqrt(252) if len(daily_returns) > 0 else 0
        
        # Sortino比率（向量化）
        downside_returns = daily_returns[daily_returns < 0]
        if len(downside_returns) > 0:
            downside_std = downside_returns.std()
            sortino_ratio = (annual_return - 0.03) / (downside_std * np.sqrt(252)) if downside_std > 0 else 0
        else:
            sortino_ratio = 0
        
        # 交易统计（向量化）
        win_mask = trades_df['pnl'] > 0
        lose_mask = trades_df['pnl'] <= 0
        
        win_trades = trades_df[win_mask]
        lose_trades = trades_df[lose_mask]
        
        if len(trades_df) > 0:
            win_rate = np.mean(win_mask) * 100
            avg_return = trades_df['pnl_pct'].mean() * 100
            
            # 向量化计算平均值
            avg_win = win_trades['pnl_pct'].mean() * 100 if len(win_trades) > 0 else 0
            avg_loss = lose_trades['pnl_pct'].mean() * 100 if len(lose_trades) > 0 else 0
            
            # 向量化计算盈亏比
            total_win = win_trades['pnl'].sum() if len(win_trades) > 0 else 0
            total_lose = lose_trades['pnl'].sum() if len(lose_trades) > 0 else 0
            profit_factor = abs(total_win / total_lose) if total_lose != 0 else float('inf')
            
            # 向量化计算极值
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
        
        # Calmar比率
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        return {
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
            'trades': trades_df.to_dict('records'),
            'equity_curve': equity_curve.to_dict('records')
        }
    
    @staticmethod
    def _empty_metrics() -> Dict:
        """返回空指标字典"""
        return {
            'total_return': 0,
            'annual_return': 0,
            'sharpe_ratio': 0,
            'sortino_ratio': 0,
            'calmar_ratio': 0,
            'max_drawdown': 0,
            'volatility': 0,
            'win_rate': 0,
            'profit_factor': 0,
            'avg_return': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'max_profit': 0,
            'max_loss': 0,
            'total_trades': 0,
            'trading_days': 0,
            'final_equity': 0,
            'trades': [],
            'equity_curve': []
        }


class VectorizedPositionChecker:
    """向量化持仓检查器"""
    
    def __init__(self, config: BacktestConfig):
        self.config = config
    
    def check_positions_vectorized(
        self,
        positions: Dict[str, Dict],
        daily_df: pd.DataFrame,
        trade_date: str,
        capital: float
    ) -> tuple:
        """
        向量化检查所有持仓
        
        Args:
            positions: 持仓字典
            daily_df: 当日行情数据
            trade_date: 交易日期
            capital: 当前现金
        
        Returns:
            (positions_to_close, updated_capital)
        """
        if not positions:
            return [], capital
        
        positions_to_close = []
        
        # 将daily_df转换为字典以加速查找
        price_map = dict(zip(
            daily_df['ts_code'],
            zip(daily_df['close'], daily_df['high'], daily_df['low'])
        ))
        
        for ts_code, position in positions.items():
            if ts_code not in price_map:
                continue
            
            close_price, high_price, low_price = price_map[ts_code]
            
            # 向量化计算收益率
            entry_price = position['entry_price']
            if entry_price <= 0:
                continue
            
            pnl_pct = (close_price - entry_price) / entry_price
            
            # 检查平仓条件
            exit_reason = self._check_exit_conditions(position, pnl_pct, low_price)
            
            if exit_reason:
                positions_to_close.append({
                    'ts_code': ts_code,
                    'price': close_price,
                    'reason': exit_reason
                })
        
        return positions_to_close, capital
    
    def _check_exit_conditions(
        self,
        position: Dict,
        pnl_pct: float,
        low_price: float
    ) -> Optional[str]:
        """检查是否触发平仓条件"""
        # 硬止损
        if pnl_pct <= self.config.stop_loss:
            return 'stop_loss'
        
        # ATR止损
        if 'atr_stop_price' in position and low_price <= position['atr_stop_price']:
            return 'atr_stop'
        
        # 移动止盈
        if 'breakeven_price' in position and low_price <= position['breakeven_price'] and pnl_pct > 0.1:
            return 'breakeven'
        
        # 固定止盈
        if pnl_pct >= self.config.take_profit:
            return 'take_profit'
        
        # 最大持仓天数需要单独计算（非向量化）
        return None
    
    def update_breakeven_vectorized(self, positions: Dict):
        """向量化更新保本止盈价格"""
        for position in positions.values():
            # 计算当前收益率（假设需要额外数据，这里简化处理）
            # 实际应用中可能需要传入当前价格
            if 'breakeven_price' not in position:
                # 当盈利超过10%时设置保本价
                pass


class VectorizedBacktestEngine(BacktestEngine):
    """
    向量化回测引擎
    继承原有BacktestEngine，优化关键计算路径
    """
    
    def __init__(self, config: BacktestConfig, strategy, fetcher):
        super().__init__(config, strategy, fetcher)
        self.position_checker = VectorizedPositionChecker(config)
    
    def _check_positions(self, daily_df: pd.DataFrame, trade_date: str, capital: float) -> float:
        """
        向量化版本的持仓检查

        覆盖父类方法，使用向量化计算提升性能

        返回:
            更新后的现金
        """
        # 先检查持仓天数（这部分无法向量化）
        positions_to_close_by_days = self._check_holding_days_vectorized(
            trade_date
        )

        # 向量化检查价格相关止损止盈
        positions_to_close_by_price, capital = self.position_checker.check_positions_vectorized(
            self.positions, daily_df, trade_date, capital
        )

        # 合并需要平仓的列表
        positions_to_close = []
        close_set = set()

        # 添加天数平仓
        for item in positions_to_close_by_days:
            if item['ts_code'] not in close_set:
                positions_to_close.append(item)
                close_set.add(item['ts_code'])

        # 添加价格平仓
        for item in positions_to_close_by_price:
            if item['ts_code'] not in close_set:
                positions_to_close.append(item)
                close_set.add(item['ts_code'])

        # 执行平仓
        for item in positions_to_close:
            capital = self._close_position(
                item['ts_code'],
                item['price'],
                trade_date,
                item['reason'],
                capital
            )

        return capital
        
        # 合并需要平仓的列表
        positions_to_close = []
        close_set = set()
        
        # 添加天数平仓
        for item in positions_to_close_by_days:
            if item['ts_code'] not in close_set:
                positions_to_close.append(item)
                close_set.add(item['ts_code'])
        
        # 添加价格平仓
        for item in positions_to_close_by_price:
            if item['ts_code'] not in close_set:
                positions_to_close.append(item)
                close_set.add(item['ts_code'])
        
        # 执行平仓
        for item in positions_to_close:
            self._close_position(
                item['ts_code'],
                item['price'],
                trade_date,
                item['reason'],
                capital
            )
    
    def _check_holding_days_vectorized(self, trade_date: str) -> List[Dict]:
        """
        向量化检查持仓天数
        
        Returns:
            需要平仓的持仓列表
        """
        positions_to_close = []
        
        for ts_code, position in self.positions.items():
            holding_days = self._calc_holding_days(
                position['entry_date'],
                trade_date
            )
            
            if holding_days >= self.config.max_holding_days:
                positions_to_close.append({
                    'ts_code': ts_code,
                    'price': 0,  # 稍后获取
                    'reason': 'max_hold'
                })
        
        return positions_to_close
    
    def _calculate_equity(self, daily_df: pd.DataFrame, trade_date: str, cash: float) -> Dict:
        """
        向量化版本的总资产计算
        
        覆盖父类方法，使用向量化查找价格
        """
        if not self.positions:
            return {
                'date': trade_date,
                'cash': cash,
                'positions_value': 0.0,
                'equity': cash
            }
        
        # 创建价格查找字典
        price_map = dict(zip(daily_df['ts_code'], daily_df['close']))
        
        # 向量化计算持仓价值
        ts_codes = []
        prices = []
        shares_list = []
        
        for ts_code, position in self.positions.items():
            if ts_code in price_map:
                ts_codes.append(ts_code)
                prices.append(price_map[ts_code])
                shares_list.append(position['shares'])
        
        if ts_codes:
            prices_arr = np.array(prices)
            shares_arr = np.array(shares_list)
            positions_value = np.sum(prices_arr * shares_arr)
        else:
            positions_value = 0.0
        
        total_equity = cash + positions_value
        
        return {
            'date': trade_date,
            'cash': cash,
            'positions_value': positions_value,
            'equity': total_equity
        }
    
    def _calculate_metrics(self) -> Dict:
        """
        向量化版本的回测指标计算
        
        覆盖父类方法，使用VectorizedMetrics
        """
        if not self.equity_curve or not self.trades:
            return VectorizedMetrics._empty_metrics()
        
        equity_df = pd.DataFrame(self.equity_curve)
        trades_df = pd.DataFrame([t.__dict__ for t in self.trades])
        
        return VectorizedMetrics.calculate_all_metrics_vectorized(
            equity_df,
            trades_df,
            self.config.initial_capital
        )


class ParallelBacktestRunner:
    """并行回测运行器"""
    
    def __init__(self, config: BacktestConfig, strategy, fetcher, max_workers: Optional[int] = None):
        self.config = config
        self.strategy = strategy
        self.fetcher = fetcher
        self.max_workers = max_workers or min(4, mp.cpu_count())
    
    def run_parallel_backtest(
        self,
        param_sets: List[Dict],
        use_vectorized: bool = True
    ) -> List[Dict]:
        """
        并行运行多组参数的回测
        
        Args:
            param_sets: 参数配置列表
            use_vectorized: 是否使用向量化引擎
        
        Returns:
            回测结果列表
        """
        results = []
        
        logger.info(f"开始并行回测，参数组数: {len(param_sets)}，工作线程: {self.max_workers}")
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            future_to_params = {}
            for i, params in enumerate(param_sets):
                # 创建回测引擎
                engine_class = VectorizedBacktestEngine if use_vectorized else BacktestEngine
                engine = engine_class(self.config, self.strategy, self.fetcher)
                
                # 更新配置
                for key, value in params.items():
                    setattr(self.config, key, value)
                
                # 提交任务
                future = executor.submit(engine.run)
                future_to_params[future] = (i, params)
            
            # 收集结果
            for future in as_completed(future_to_params):
                idx, params = future_to_params[future]
                try:
                    result = future.result()
                    result['params'] = params
                    results.append(result)
                    logger.info(f"参数组 {idx+1}/{len(param_sets)} 完成")
                except Exception as e:
                    logger.error(f"参数组 {idx+1} 回测失败: {e}")
                    results.append({
                        'params': params,
                        'error': str(e)
                    })
        
        return results


def vectorized_batch_backtest(
    backtest_configs: List[Dict],
    strategy,
    fetcher,
    use_parallel: bool = True
) -> pd.DataFrame:
    """
    批量回测（向量化版本）
    
    Args:
        backtest_configs: 回测配置列表
        strategy: 策略实例
        fetcher: 数据获取器
        use_parallel: 是否使用并行
    
    Returns:
        包含所有回测结果的DataFrame
    """
    results = []
    
    for i, config_dict in enumerate(backtest_configs):
        config = BacktestConfig(**config_dict)
        engine = VectorizedBacktestEngine(config, strategy, fetcher)
        
        try:
            result = engine.run()
            result['config'] = config_dict
            results.append(result)
        except Exception as e:
            logger.error(f"回测 {i+1}/{len(backtest_configs)} 失败: {e}")
    
    return pd.DataFrame(results)
