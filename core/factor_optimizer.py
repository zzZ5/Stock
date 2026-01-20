"""
因子权重动态分配模块
提供IC分析、历史回测、因子正交化等功能
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ICAnalysisResult:
    """IC分析结果"""
    factor_name: str
    ic_mean: float
    ic_std: float
    ic_ir: float  # IC信息比率 = IC均值 / IC标准差
    ic_positive_ratio: float  # IC为正的比例
    rank_ic_mean: float
    rank_ic_std: float
    rank_ic_ir: float


@dataclass
class FactorWeight:
    """因子权重"""
    factor_name: str
    weight: float
    weight_source: str  # 'ic', 'backtest', 'ortho', 'manual'
    confidence: float  # 权重置信度 (0-1)


class ICAnalyzer:
    """IC分析器"""

    @staticmethod
    def calculate_ic(factor_values: pd.Series,
                     returns: pd.Series,
                     method: str = 'pearson') -> float:
        """
        计算IC（Information Coefficient）

        参数:
            factor_values: 因子值序列
            returns: 收益率序列
            method: 计算方法 ('pearson', 'spearman')

        返回:
            IC值
        """
        # 对齐数据
        aligned_data = pd.DataFrame({'factor': factor_values, 'return': returns}).dropna()

        if len(aligned_data) < 3:
            return np.nan

        if method == 'pearson':
            return aligned_data['factor'].corr(aligned_data['return'])
        elif method == 'spearman':
            return aligned_data['factor'].corr(aligned_data['return'], method='spearman')
        else:
            logger.warning(f"未知的IC计算方法: {method}")
            return np.nan

    @staticmethod
    def analyze_factor_ic(factor_values: pd.Series,
                          returns_df: pd.DataFrame,
                          lookback_periods: List[int] = [5, 10, 20]) -> ICAnalysisResult:
        """
        分析因子的IC表现

        参数:
            factor_values: 因子值序列
            returns_df: 收益率DataFrame (columns为不同周期)
            lookback_periods: 回测周期列表

        返回:
            IC分析结果
        """
        ic_results = []

        for period in lookback_periods:
            if f'return_{period}d' not in returns_df.columns:
                continue

            period_returns = returns_df[f'return_{period}d']

            # 计算IC
            ic = ICAnalyzer.calculate_ic(factor_values, period_returns, method='pearson')
            rank_ic = ICAnalyzer.calculate_ic(factor_values, period_returns, method='spearman')

            ic_results.append({
                'period': period,
                'ic': ic,
                'rank_ic': rank_ic
            })

        if not ic_results:
            return ICAnalysisResult('unknown', 0, 0, 0, 0, 0, 0, 0)

        # 聚合IC统计
        ic_values = [r['ic'] for r in ic_results if not pd.isna(r['ic'])]
        rank_ic_values = [r['rank_ic'] for r in ic_results if not pd.isna(r['rank_ic'])]

        ic_mean = np.mean(ic_values) if ic_values else 0
        ic_std = np.std(ic_values) if ic_values else 1
        ic_positive_ratio = np.mean([1 for ic in ic_values if ic > 0]) if ic_values else 0.5
        ic_ir = ic_mean / ic_std if ic_std > 0 else 0

        rank_ic_mean = np.mean(rank_ic_values) if rank_ic_values else 0
        rank_ic_std = np.std(rank_ic_values) if rank_ic_values else 1
        rank_ic_ir = rank_ic_mean / rank_ic_std if rank_ic_std > 0 else 0

        return ICAnalysisResult(
            factor_name=factor_values.name if hasattr(factor_values, 'name') else 'unknown',
            ic_mean=ic_mean,
            ic_std=ic_std,
            ic_ir=ic_ir,
            ic_positive_ratio=ic_positive_ratio,
            rank_ic_mean=rank_ic_mean,
            rank_ic_std=rank_ic_std,
            rank_ic_ir=rank_ic_ir
        )

    @staticmethod
    def calculate_rolling_ic(factor_values: pd.Series,
                            returns: pd.Series,
                            window: int = 20) -> pd.Series:
        """
        计算滚动IC

        参数:
            factor_values: 因子值序列
            returns: 收益率序列
            window: 滚动窗口

        返回:
            滚动IC序列
        """
        aligned_data = pd.DataFrame({'factor': factor_values, 'return': returns}).dropna()

        if len(aligned_data) < window:
            return pd.Series([np.nan] * len(aligned_data), index=aligned_data.index)

        rolling_ic = aligned_data['factor'].rolling(window=window).corr(aligned_data['return'])

        return rolling_ic


class FactorOrthogonalizer:
    """因子正交化器"""

    def __init__(self, method: str = 'gram-schmidt'):
        """
        初始化因子正交化器

        参数:
            method: 正交化方法 ('gram-schmidt', 'pca', 'regression')
        """
        self.method = method

    def orthogonalize(self, factor_df: pd.DataFrame,
                     reference_factors: List[str] = None) -> pd.DataFrame:
        """
        对因子进行正交化处理

        参数:
            factor_df: 因子DataFrame (columns为因子名, index为股票代码/日期)
            reference_factors: 参考因子列表（正交化到这些因子）

        返回:
            正交化后的因子DataFrame
        """
        factor_df = factor_df.copy()
        factors = factor_df.columns.tolist()

        if reference_factors is None:
            reference_factors = factors[:-1]  # 默认使用前几个因子作为参考

        if self.method == 'pca':
            return self._orthogonalize_pca(factor_df)
        elif self.method == 'regression':
            return self._orthogonalize_regression(factor_df, reference_factors)
        else:
            return self._orthogonalize_gram_schmidt(factor_df, reference_factors)

    def _orthogonalize_gram_schmidt(self, factor_df: pd.DataFrame,
                                   reference_factors: List[str]) -> pd.DataFrame:
        """Gram-Schmidt正交化"""
        result = factor_df.copy()

        # 标准化
        scaler = StandardScaler()
        normalized = pd.DataFrame(
            scaler.fit_transform(factor_df),
            columns=factor_df.columns,
            index=factor_df.index
        )

        # 对每个因子进行正交化
        orthogonalized = normalized.copy()

        for factor in normalized.columns:
            if factor not in reference_factors:
                continue

            # 从剩余因子中移除该因子的线性影响
            for other_factor in normalized.columns:
                if other_factor == factor:
                    continue

                # 计算投影
                dot_product = np.dot(normalized[factor], normalized[other_factor])
                orthogonalized[other_factor] -= dot_product * normalized[factor]

        # 反标准化
        result_ortho = pd.DataFrame(
            scaler.inverse_transform(orthogonalized),
            columns=orthogonalized.columns,
            index=orthogonalized.index
        )

        return result_ortho

    def _orthogonalize_pca(self, factor_df: pd.DataFrame,
                         n_components: int = None) -> pd.DataFrame:
        """PCA正交化"""
        # 移除缺失值
        factor_df_clean = factor_df.dropna()

        if len(factor_df_clean) < 10:
            return factor_df

        # 标准化
        scaler = StandardScaler()
        normalized = scaler.fit_transform(factor_df_clean)

        # PCA变换
        if n_components is None:
            n_components = min(len(factor_df.columns), factor_df_clean.shape[0] - 1)

        pca = PCA(n_components=n_components)
        pca_factors = pca.transform(normalized)

        # 转换回DataFrame
        result_ortho = pd.DataFrame(
            scaler.inverse_transform(pca_factors),
            index=factor_df_clean.index,
            columns=[f'PC_{i+1}' for i in range(n_components)]
        )

        return result_ortho

    def _orthogonalize_regression(self, factor_df: pd.DataFrame,
                                  reference_factors: List[str]) -> pd.DataFrame:
        """回归正交化"""
        from sklearn.linear_model import LinearRegression

        result = factor_df.copy()

        for factor in factor_df.columns:
            if factor in reference_factors:
                continue

            # 准备数据
            X = factor_df[reference_factors].dropna()
            y = factor_df[factor].loc[X.index]

            if len(X) < 3:
                continue

            # 回归
            model = LinearRegression()
            model.fit(X, y)

            # 残差作为正交化后的因子
            residuals = y - model.predict(X)
            result[factor].loc[X.index] = residuals

        return result


class WeightOptimizer:
    """权重优化器"""

    def __init__(self):
        """初始化权重优化器"""
        self.ic_analyzer = ICAnalyzer()

    def optimize_weights_by_ic(self,
                                 factor_ic_results: Dict[str, ICAnalysisResult],
                                 method: str = 'ic_ir') -> Dict[str, float]:
        """
        基于IC分析结果优化因子权重

        参数:
            factor_ic_results: 因子IC分析结果字典
            method: 权重计算方法 ('ic_mean', 'ic_ir', 'positive_ratio')

        返回:
            因子权重字典
        """
        weights = {}

        # 计算每个因子的得分
        factor_scores = {}
        for factor_name, ic_result in factor_ic_results.items():
            if method == 'ic_mean':
                score = abs(ic_result.ic_mean)
            elif method == 'ic_ir':
                score = abs(ic_result.ic_ir)
            elif method == 'positive_ratio':
                score = ic_result.ic_positive_ratio
            else:
                score = abs(ic_result.ic_mean)

            factor_scores[factor_name] = score

        # 归一化为权重
        total_score = sum(factor_scores.values())

        if total_score == 0:
            # 平均权重
            n_factors = len(factor_scores)
            weights = {k: 1.0 / n_factors for k in factor_scores.keys()}
        else:
            weights = {k: v / total_score for k, v in factor_scores.items()}

        return weights

    def optimize_weights_by_backtest(self,
                                     factor_returns: pd.DataFrame,
                                     target: str = 'sharpe') -> Dict[str, float]:
        """
        基于历史回测结果优化因子权重

        参数:
            factor_returns: 因子收益DataFrame (columns为因子名)
            target: 优化目标 ('sharpe', 'return', 'max_drawdown')

        返回:
            因子权重字典
        """
        from scipy.optimize import minimize

        factors = factor_returns.columns.tolist()
        n_factors = len(factors)

        if n_factors == 0:
            return {}

        # 计算因子组合收益
        def calculate_portfolio_metrics(weights):
            """计算组合指标"""
            weights = np.array(weights)
            portfolio_return = np.dot(factor_returns, weights)

            # 计算年化收益
            annual_return = portfolio_return.mean() * 252

            # 计算年化波动率
            annual_vol = portfolio_return.std() * np.sqrt(252)

            # 夏普比率
            sharpe = annual_return / annual_vol if annual_vol > 0 else -np.inf

            # 最大回撤
            cumulative = (1 + portfolio_return).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = drawdown.min()

            if target == 'sharpe':
                return -sharpe
            elif target == 'return':
                return -annual_return
            elif target == 'max_drawdown':
                return max_drawdown
            else:
                return -sharpe

        # 约束条件
        constraints = {
            'type': 'eq',
            'fun': lambda w: np.sum(w) - 1.0  # 权重和为1
        }

        # 权重边界（允许做空，限制在[-0.5, 1.5]）
        bounds = [(-0.5, 1.5) for _ in range(n_factors)]

        # 初始权重（平均分配）
        initial_weights = np.array([1.0 / n_factors] * n_factors)

        # 优化
        result = minimize(
            calculate_portfolio_metrics,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        if result.success:
            optimal_weights = result.x
            weights = {factor: weight for factor, weight in zip(factors, optimal_weights)}
        else:
            logger.warning(f"权重优化失败: {result.message}")
            # 返回平均权重
            weights = {factor: 1.0 / n_factors for factor in factors}

        return weights

    def dynamic_weight_adjustment(self,
                                  base_weights: Dict[str, float],
                                  factor_ic_results: Dict[str, ICAnalysisResult],
                                  decay_factor: float = 0.1) -> Dict[str, float]:
        """
        动态权重调整（基于IC变化）

        参数:
            base_weights: 基础权重
            factor_ic_results: 最新的IC分析结果
            decay_factor: 衰减因子

        返回:
            调整后的权重
        """
        adjusted_weights = {}

        for factor_name, base_weight in base_weights.items():
            if factor_name not in factor_ic_results:
                adjusted_weights[factor_name] = base_weight
                continue

            ic_result = factor_ic_results[factor_name]

            # 根据IC信息比率调整权重
            # IC_IR越高，权重越大
            adjustment = np.tanh(ic_result.ic_ir) * decay_factor

            adjusted_weight = base_weight * (1 + adjustment)

            # 确保权重非负
            adjusted_weights[factor_name] = max(0, adjusted_weight)

        # 归一化
        total_weight = sum(adjusted_weights.values())
        if total_weight > 0:
            adjusted_weights = {k: v / total_weight for k, v in adjusted_weights.items()}

        return adjusted_weights


class FactorWeightManager:
    """因子权重管理器"""

    def __init__(self):
        """初始化权重管理器"""
        self.weight_optimizer = WeightOptimizer()
        self.current_weights: Dict[str, FactorWeight] = {}
        self.history: List[Dict] = []

    def set_weights(self, weights: Dict[str, float],
                   source: str = 'manual',
                   confidence: float = 1.0) -> None:
        """
        设置因子权重

        参数:
            weights: 权重字典
            source: 权重来源
            confidence: 置信度
        """
        self.current_weights = {}

        for factor_name, weight in weights.items():
            self.current_weights[factor_name] = FactorWeight(
                factor_name=factor_name,
                weight=weight,
                weight_source=source,
                confidence=confidence
            )

        # 记录历史
        self.history.append({
            'timestamp': pd.Timestamp.now(),
            'weights': self.current_weights.copy(),
            'source': source
        })

    def get_weights(self) -> Dict[str, float]:
        """
        获取当前权重

        返回:
            权重字典
        """
        return {k: v.weight for k, v in self.current_weights.items()}

    def optimize_weights(self,
                          factor_values: pd.DataFrame,
                          returns_df: pd.DataFrame,
                          method: str = 'ic') -> None:
        """
        优化因子权重

        参数:
            factor_values: 因子值DataFrame
            returns_df: 收益率DataFrame
            method: 优化方法
        """
        if method == 'ic':
            # IC分析
            ic_results = {}

            for factor_name in factor_values.columns:
                ic_result = self.weight_optimizer.ic_analyzer.analyze_factor_ic(
                    factor_values[factor_name],
                    returns_df
                )
                ic_results[factor_name] = ic_result

            # 优化权重
            weights = self.weight_optimizer.optimize_weights_by_ic(
                ic_results,
                method='ic_ir'
            )

            self.set_weights(weights, source='ic', confidence=0.8)

        elif method == 'backtest':
            # 回测优化
            factor_returns = returns_df  # 这里简化处理，实际需要更复杂的回测

            weights = self.weight_optimizer.optimize_weights_by_backtest(
                factor_returns,
                target='sharpe'
            )

            self.set_weights(weights, source='backtest', confidence=0.7)

        elif method == 'dynamic':
            # 动态调整
            base_weights = self.get_weights()

            # IC分析
            ic_results = {}
            for factor_name in factor_values.columns:
                ic_result = self.weight_optimizer.ic_analyzer.analyze_factor_ic(
                    factor_values[factor_name],
                    returns_df
                )
                ic_results[factor_name] = ic_result

            weights = self.weight_optimizer.dynamic_weight_adjustment(
                base_weights,
                ic_results
            )

            self.set_weights(weights, source='dynamic', confidence=0.9)

    def orthogonalize_factors(self, factor_df: pd.DataFrame,
                             method: str = 'gram-schmidt') -> pd.DataFrame:
        """
        正交化因子

        参数:
            factor_df: 因子DataFrame
            method: 正交化方法

        返回:
            正交化后的因子DataFrame
        """
        orthogonalizer = FactorOrthogonalizer(method=method)
        return orthogonalizer.orthogonalize(factor_df)

    def export_weights(self) -> pd.DataFrame:
        """
        导出权重

        返回:
            权重DataFrame
        """
        data = []
        for factor_name, weight_info in self.current_weights.items():
            data.append({
                'factor_name': factor_name,
                'weight': weight_info.weight,
                'source': weight_info.weight_source,
                'confidence': weight_info.confidence
            })

        return pd.DataFrame(data)


def calculate_factor_correlation(factor_df: pd.DataFrame) -> pd.DataFrame:
    """
    计算因子相关性矩阵

    参数:
        factor_df: 因子DataFrame

    返回:
        相关性矩阵
    """
    return factor_df.corr()


def calculate_factor_variance_inflation_factor(factor_df: pd.DataFrame) -> pd.Series:
    """
    计算因子VIF（方差膨胀因子）

    参数:
        factor_df: 因子DataFrame

    返回:
        VIF值Series
    """
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    vif_data = []
    for i in range(factor_df.shape[1]):
        vif = variance_inflation_factor(factor_df.values, i)
        vif_data.append(vif)

    return pd.Series(vif_data, index=factor_df.columns)
