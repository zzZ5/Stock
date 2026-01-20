"""
组合优化模块
提供投资组合理论、风险平价、Kelly公式等组合优化方法
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy.optimize import minimize
from core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class OptimalPortfolio:
    """最优组合"""
    weights: Dict[str, float]  # 权重字典
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float
    method: str  # 优化方法
    constraints: List[str]  # 使用的约束条件


class MarkowitzOptimizer:
    """Markowitz均值-方差优化器"""

    @staticmethod
    def optimize(returns: pd.DataFrame,
                risk_free_rate: float = 0.03,
                target_return: float = None,
                allow_short: bool = False) -> OptimalPortfolio:
        """
        Markowitz组合优化

        参数:
            returns: 收益率DataFrame (columns为资产名)
            risk_free_rate: 无风险利率
            target_return: 目标收益率（None表示最大化Sharpe比）
            allow_short: 是否允许做空

        返回:
            最优组合
        """
        assets = returns.columns.tolist()
        n_assets = len(assets)

        # 计算预期收益率和协方差矩阵
        expected_returns = returns.mean() * 252  # 年化
        cov_matrix = returns.cov() * 252  # 年化协方差

        # 目标函数
        def objective_function(weights):
            """目标函数：最小化波动率"""
            portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            return portfolio_volatility

        # 目标收益率约束
        def return_constraint(weights, target):
            portfolio_return = np.dot(weights, expected_returns)
            return portfolio_return - target

        # 约束条件
        constraints = []

        # 权重和为1
        constraints.append({
            'type': 'eq',
            'fun': lambda w: np.sum(w) - 1.0
        })

        # 目标收益率约束（如果指定）
        if target_return is not None:
            constraints.append({
                'type': 'eq',
                'fun': lambda w: return_constraint(w, target_return)
            })

        # 权重边界
        if allow_short:
            bounds = [(-0.5, 1.5) for _ in range(n_assets)]
        else:
            bounds = [(0, 1) for _ in range(n_assets)]

        # 初始权重（等权重）
        initial_weights = np.array([1.0 / n_assets] * n_assets)

        # 优化
        result = minimize(
            objective_function,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        if not result.success:
            logger.warning(f"Markowitz优化失败: {result.message}")
            # 返回等权重组合
            optimal_weights = np.array([1.0 / n_assets] * n_assets)
        else:
            optimal_weights = result.x

        # 计算组合指标
        portfolio_return = np.dot(optimal_weights, expected_returns)
        portfolio_volatility = np.sqrt(np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights)))
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility if portfolio_volatility > 0 else 0

        weights_dict = {asset: weight for asset, weight in zip(assets, optimal_weights)}

        constraint_names = ['sum_to_one']
        if target_return is not None:
            constraint_names.append('target_return')
        if allow_short:
            constraint_names.append('allow_short')

        return OptimalPortfolio(
            weights=weights_dict,
            expected_return=portfolio_return,
            expected_volatility=portfolio_volatility,
            sharpe_ratio=sharpe_ratio,
            method='markowitz',
            constraints=constraint_names
        )

    @staticmethod
    def maximize_sharpe(returns: pd.DataFrame,
                        risk_free_rate: float = 0.03,
                        allow_short: bool = False) -> OptimalPortfolio:
        """
        最大化Sharpe比

        参数:
            returns: 收益率DataFrame
            risk_free_rate: 无风险利率
            allow_short: 是否允许做空

        返回:
            最优组合
        """
        assets = returns.columns.tolist()
        n_assets = len(assets)

        expected_returns = returns.mean() * 252
        cov_matrix = returns.cov() * 252

        def objective_function(weights):
            """目标函数：最大化Sharpe比（最小化负Sharpe比）"""
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

            if portfolio_volatility == 0:
                return -np.inf

            sharpe = (portfolio_return - risk_free_rate) / portfolio_volatility
            return -sharpe

        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
        ]

        bounds = [(-0.5, 1.5) for _ in range(n_assets)] if allow_short else [(0, 1) for _ in range(n_assets)]
        initial_weights = np.array([1.0 / n_assets] * n_assets)

        result = minimize(
            objective_function,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        if not result.success:
            logger.warning(f"Sharpe优化失败: {result.message}")
            optimal_weights = np.array([1.0 / n_assets] * n_assets)
        else:
            optimal_weights = result.x

        portfolio_return = np.dot(optimal_weights, expected_returns)
        portfolio_volatility = np.sqrt(np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights)))
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility if portfolio_volatility > 0 else 0

        weights_dict = {asset: weight for asset, weight in zip(assets, optimal_weights)}

        return OptimalPortfolio(
            weights=weights_dict,
            expected_return=portfolio_return,
            expected_volatility=portfolio_volatility,
            sharpe_ratio=sharpe_ratio,
            method='max_sharpe',
            constraints=['sum_to_one', 'allow_short' if allow_short else 'long_only']
        )


class RiskParityOptimizer:
    """风险平价优化器"""

    @staticmethod
    def optimize(returns: pd.DataFrame) -> OptimalPortfolio:
        """
        风险平价优化

        参数:
            returns: 收益率DataFrame

        返回:
            最优组合
        """
        assets = returns.columns.tolist()
        n_assets = len(assets)

        cov_matrix = returns.cov() * 252

        def risk_budget_objective(weights):
            """目标函数：风险贡献差异最小化"""
            portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

            # 计算每个资产的风险贡献
            marginal_contrib = np.dot(cov_matrix, weights)
            risk_contrib = weights * marginal_contrib / portfolio_volatility
            target_risk = portfolio_volatility / n_assets

            # 最小化风险贡献与目标风险的差异
            return np.sum((risk_contrib - target_risk) ** 2)

        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
        ]

        bounds = [(0, 1) for _ in range(n_assets)]
        initial_weights = np.array([1.0 / n_assets] * n_assets)

        result = minimize(
            risk_budget_objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        if not result.success:
            logger.warning(f"风险平价优化失败: {result.message}")
            optimal_weights = np.array([1.0 / n_assets] * n_assets)
        else:
            optimal_weights = result.x

        expected_returns = returns.mean() * 252
        portfolio_return = np.dot(optimal_weights, expected_returns)
        portfolio_volatility = np.sqrt(np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights)))
        sharpe_ratio = portfolio_return / portfolio_volatility if portfolio_volatility > 0 else 0

        weights_dict = {asset: weight for asset, weight in zip(assets, optimal_weights)}

        return OptimalPortfolio(
            weights=weights_dict,
            expected_return=portfolio_return,
            expected_volatility=portfolio_volatility,
            sharpe_ratio=sharpe_ratio,
            method='risk_parity',
            constraints=['sum_to_one', 'long_only', 'equal_risk_contribution']
        )


class KellyOptimizer:
    """Kelly公式优化器"""

    @staticmethod
    def optimize(win_rates: Dict[str, float],
                avg_wins: Dict[str, float],
                avg_losses: Dict[str, float],
                capital: float = 100000,
                max_single_position: float = 0.25) -> Dict[str, float]:
        """
        Kelly公式优化

        参数:
            win_rates: 各资产胜率字典
            avg_wins: 各资产平均盈利字典
            avg_losses: 各资产平均亏损字典
            capital: 总资金
            max_single_position: 单个资产最大仓位比例

        返回:
            仓位字典
        """
        positions = {}

        for asset in win_rates.keys():
            win_rate = win_rates.get(asset, 0.5)
            avg_win = avg_wins.get(asset, 0.1)
            avg_loss = avg_losses.get(asset, 0.1)

            if avg_loss == 0:
                kelly_fraction = 0
            else:
                # Kelly公式: f = (bp - q) / b
                b = avg_win / avg_loss  # 盈亏比
                p = win_rate
                q = 1 - p

                kelly_fraction = (b * p - q) / b
                kelly_fraction = max(0, min(kelly_fraction, max_single_position))

            positions[asset] = capital * kelly_fraction

        return positions


class BlackLittermanOptimizer:
    """Black-Litterman优化器"""

    @staticmethod
    def optimize(returns: pd.DataFrame,
                market_weights: Dict[str, float],
                views: List[Dict] = None,
                tau: float = 0.05,
                risk_free_rate: float = 0.03) -> OptimalPortfolio:
        """
        Black-Litterman组合优化

        参数:
            returns: 历史收益率DataFrame
            market_weights: 市场权重字典
            views: 投资者观点列表 [{'assets': [asset1, asset2], 'view': 0.05}]
            tau: 不确定性参数
            risk_free_rate: 无风险利率

        返回:
            最优组合
        """
        assets = returns.columns.tolist()
        n_assets = len(assets)

        # 历史统计量
        expected_returns = returns.mean() * 252
        cov_matrix = returns.cov() * 252

        # 市场隐含收益率
        market_weight_vector = np.array([market_weights.get(asset, 1/n_assets) for asset in assets])
        market_return = np.dot(market_weight_vector, expected_returns)
        market_variance = np.dot(market_weight_vector.T, np.dot(cov_matrix, market_weight_vector))
        risk_aversion = (market_return - risk_free_rate) / market_variance if market_variance > 0 else 1

        pi = risk_aversion * np.dot(cov_matrix, market_weight_vector)

        # Black-Litterman调整
        if views is None or len(views) == 0:
            # 无投资者观点，使用市场均衡收益率
            bl_returns = pi
        else:
            # 构建观点矩阵和不确定性矩阵
            P = np.zeros((len(views), n_assets))
            Q = np.zeros(len(views))
            Omega = np.zeros((len(views), len(views)))

            for i, view in enumerate(views):
                # 观点权重向量
                view_assets = view.get('assets', [])
                for j, asset in enumerate(assets):
                    if asset in view_assets:
                        P[i, j] = 1 / len(view_assets)

                Q[i] = view.get('view', 0)

                # 观点不确定性（简化处理）
                Omega[i, i] = np.dot(P[i, :], np.dot(cov_matrix, P[i, :])) * tau

            # Black-Litterman公式
            M1 = np.linalg.inv(tau * cov_matrix)
            M2 = np.dot(P.T, np.linalg.inv(Omega))
            M3 = np.dot(M2, P)

            bl_returns = np.dot(np.linalg.inv(M1 + M3), np.dot(M1, pi) + np.dot(M2, Q))

        # Markowitz优化
        def objective_function(weights):
            portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            return portfolio_volatility

        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
        bounds = [(0, 1) for _ in range(n_assets)]
        initial_weights = np.array([1.0 / n_assets] * n_assets)

        result = minimize(
            objective_function,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        if not result.success:
            optimal_weights = np.array([1.0 / n_assets] * n_assets)
        else:
            optimal_weights = result.x

        portfolio_return = np.dot(optimal_weights, bl_returns)
        portfolio_volatility = np.sqrt(np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights)))
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility if portfolio_volatility > 0 else 0

        weights_dict = {asset: weight for asset, weight in zip(assets, optimal_weights)}

        return OptimalPortfolio(
            weights=weights_dict,
            expected_return=portfolio_return,
            expected_volatility=portfolio_volatility,
            sharpe_ratio=sharpe_ratio,
            method='black_litterman',
            constraints=['sum_to_one', 'long_only']
        )


class EfficientFrontier:
    """有效前沿"""

    @staticmethod
    def calculate_efficient_frontier(returns: pd.DataFrame,
                                  num_portfolios: int = 100,
                                  risk_free_rate: float = 0.03) -> pd.DataFrame:
        """
        计算有效前沿

        参数:
            returns: 收益率DataFrame
            num_portfolios: 模拟组合数量
            risk_free_rate: 无风险利率

        返回:
            有效前沿DataFrame
        """
        assets = returns.columns.tolist()
        n_assets = len(assets)

        expected_returns = returns.mean() * 252
        cov_matrix = returns.cov() * 252

        # 模拟随机组合
        results = []

        for _ in range(num_portfolios):
            weights = np.random.random(n_assets)
            weights /= np.sum(weights)

            portfolio_return = np.dot(weights, expected_returns)
            portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility

            results.append({
                'volatility': portfolio_volatility,
                'return': portfolio_return,
                'sharpe': sharpe_ratio,
                **{f'weight_{asset}': w for asset, w in zip(assets, weights)}
            })

        frontier_df = pd.DataFrame(results)

        # 筛选有效前沿（在相同波动率下收益率最高）
        frontier_df = frontier_df.sort_values('volatility')
        frontier_df['max_return'] = frontier_df['return'].cummax()
        efficient_frontier = frontier_df[frontier_df['return'] == frontier_df['max_return']]

        return efficient_frontier

    @staticmethod
    def get_max_sharpe_portfolio(returns: pd.DataFrame,
                               risk_free_rate: float = 0.03) -> OptimalPortfolio:
        """
        获取最大Sharpe组合

        参数:
            returns: 收益率DataFrame
            risk_free_rate: 无风险利率

        返回:
            最大Sharpe组合
        """
        return MarkowitzOptimizer.maximize_sharpe(returns, risk_free_rate)

    @staticmethod
    def get_min_variance_portfolio(returns: pd.DataFrame) -> OptimalPortfolio:
        """
        获取最小方差组合

        参数:
            returns: 收益率DataFrame

        返回:
            最小方差组合
        """
        return MarkowitzOptimizer.optimize(returns, target_return=None)
