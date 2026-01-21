# 趋势雷达选股系统 - 完整文档

## 📚 文档导航

本目录包含项目的所有文档，按照使用场景分类。

---

## 🚀 快速开始

### [快速开始指南](QUICK_START.md)
- 5分钟快速上手
- 常用命令速查
- 配置Token方法
- 常见问题解答

---

## 📖 项目说明

### [README.md](README.md)
- 项目简介
- 功能特性
- 安装步骤
- 项目结构
- 模块说明
- 代码示例
- 版本历史

---

## 🗂️ 项目结构

### [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)
- 完整目录结构
- 各模块功能说明
- 文件依赖关系
- 扩展开发指南

---

## 核心功能文档

## 1. 选股系统

### 选股逻辑

**候选股票条件（必须满足）：**
1. 收盘价 + 最高价突破60日高位
2. MA20 > MA60 且 MA60斜率向上
3. 成交额/20日均额 >= 1.2倍
4. ADX > 25（强趋势）
5. KDJ-J <= 100（不过热）
6. 价格位置 < 0.9（不在高位）
7. RSI14 <= 80

### 市场环境判断

**多头信号：**
- 均线多头排列 (MA20 > MA60 > MA120)
- MA20斜率向上
- MA60斜率向上
- 近期动量为正

**空头信号：**
- 均线空头排列 (MA20 < MA60)
- MA20斜率向下
- 近期动量为负

**市场环境：**
- **牛市**: 多头信号≥3，空头信号≤1
- **震荡市**: 多头信号≥2，空头信号≤2
- **熊市**: 其他情况

### 止损/止盈

- **硬止损**: -10%
- **ATR止损**: 入场价 - 2.5*ATR（动态追踪）
- **保本止盈**: 盈利10%后，止损价移到成本价
- **止盈**: +25%
- **最大持仓**: 20天

---

## 2. 回测系统

### 回测配置

```python
from analysis.backtest import BacktestEngine, BacktestConfig

config = BacktestConfig(
    start_date='20230101',
    end_date='20231231',
    initial_capital=1000000,
    max_positions=10,
    position_size=0.1,
    slippage=0.001,           # 滑点0.1%
    commission=0.0003,        # 手续费0.03%
    stop_loss=-0.10,
    take_profit=0.25,
    max_holding_days=20,
    rebalance_days=5
)
```

### 向量化回测（高性能）

```python
from analysis.backtest_vectorized import VectorizedBacktestEngine

# 性能提升5-10倍
engine = VectorizedBacktestEngine(config, strategy, fetcher)
result = engine.run()
```

### 并行回测

```python
from analysis.backtest_vectorized import ParallelBacktestRunner

# 并行运行多个参数组合
runner = ParallelBacktestRunner(max_workers=4)
results = runner.run_backtests(param_list)
```

---

## 3. 参数优化

### 网格搜索

```python
from analysis.optimizer import ParameterOptimizer

param_grid = {
    'BREAKOUT_N': [40, 60, 80],
    'MA_FAST': [10, 20, 30],
    'MA_SLOW': [40, 60, 80],
    'VOL_CONFIRM_MULT': [1.2, 1.5, 2.0],
    'RSI_MAX': [70, 75, 80]
}

optimizer = ParameterOptimizer(fetcher, config)
results = optimizer.grid_search(param_grid)
```

### Walk-Forward验证

```python
# 验证参数稳定性
wf_results = optimizer.walk_forward_analysis(
    train_days=252,    # 训练期1年
    test_days=63,      # 测试期3个月
    step_days=63       # 滚动步长
)
```

### 贝叶斯优化

```python
param_bounds = {
    'BREAKOUT_N': (20, 100),
    'MA_FAST': (5, 30),
    'MA_SLOW': (20, 100),
    'VOL_CONFIRM_MULT': (1.0, 3.0),
    'RSI_MAX': (60, 90)
}

best = optimizer.bayesian_optimization(param_bounds, n_iterations=50)
```

---

## 4. 可视化系统

### 股票K线图

```python
from visualization import plot_stock_candlestick

# 基本K线图
fig = plot_stock_candlestick(df, title="股票K线图")

# 带指标的K线图
fig = plot_stock_candlestick(
    df,
    indicators=['ma', 'bollinger', 'volume', 'macd'],
    title="完整分析"
)
```

### 回测结果可视化

```python
from visualization import plot_backtest_results

fig = plot_backtest_results(
    results,
    figsize=(16, 12),
    save_path='reports/backtest_results.png'
)
```

### 参数优化热力图

```python
from visualization import plot_parameter_heatmap

fig = plot_parameter_heatmap(
    optimization_results,
    param1='BREAKOUT_N',
    param2='MA_FAST',
    metric='total_return'
)
```

---

## 5. 缓存和并发

### 优化的缓存管理器

```python
from core.cache_manager import CacheManager

cache = CacheManager(
    cache_dir="./cache",
    memory_cache_size=100,      # 内存缓存容量
    enable_compression=True     # 启用gzip压缩
)

# 查看统计信息
cache.print_cache_stats()
```

### 线程安全的限流器

```python
from core.utils import RateLimiter

# 创建限流器（每分钟最多200次调用）
limiter = RateLimiter(max_calls_per_minute=200)

# 在API调用前等待
limiter.wait_if_needed()
```

---

## 6. 配置管理

### 使用配置文件

```python
from config.config_loader import ConfigLoader

# 加载配置文件
config = ConfigLoader.load('config.yaml')

# 获取配置值
top_n = config.get('TOP_N')
log_level = config.get('LOG_LEVEL')

# 运行时设置（仅内存生效）
config.set('TOP_N', 30)
```

### 配置参数

```python
# 选股参数
TOP_N = 20                      # 返回TopN候选股票
BREAKOUT_N = 60                 # 60日突破
MA_FAST = 20                    # 快速均线
MA_SLOW = 60                    # 慢速均线
VOL_CONFIRM_MULT = 1.2          # 放量确认阈值
RSI_MAX = 80                    # 过热过滤阈值

# 风险参数
MAX_LOSS_PCT = 0.10            # -10%硬止损
ATR_N = 14                      # ATR计算周期
ATR_MULT = 2.5                  # 波动止损倍数

# 日志配置
LOG_LEVEL = "INFO"
LOG_DIR = "./logs"
LOG_CONSOLE_OUTPUT = True
LOG_FILE_OUTPUT = True
```

---

## 7. 日志系统

### 日志级别

- **DEBUG**: 详细的调试信息
- **INFO**: 一般信息（默认级别）
- **WARNING**: 警告信息
- **ERROR**: 错误信息
- **CRITICAL**: 严重错误

### 使用示例

```python
from core.logger import get_logger

# 获取logger实例
logger = get_logger(__name__)

# 记录不同级别的日志
logger.debug("这是调试信息")
logger.info("这是普通信息")
logger.warning("这是警告信息")
logger.error("这是错误信息")
```

---

## 8. 技术指标

系统提供60+种技术指标：

### 趋势类
- SMA/EMA（简单/指数移动平均）
- MACD（移动平均收敛发散）
- ADX（平均趋向指标）
- Parabolic SAR（抛物线转向）
- Vortex（漩涡指标）
- TRIX（三重指数平滑平均线）

### 动量类
- RSI（相对强弱指标）
- KDJ（随机指标）
- Williams %R（威廉指标）
- CCI（顺势指标）
- Momentum（动量）
- ROC（变化率）
- DPO（去趋势价格振荡）

### 成交量类
- OBV（能量潮）
- MFI（资金流量指标）
- VPT（量价趋势）
- VWAP（成交量加权平均价）

### 波动率类
- ATR（平均真实波幅）
- Bollinger Bands（布林带）
- Chandelier Exit（吊灯止损）

---

## 9. 测试指南

### 运行测试

```bash
# 运行所有测试
python run_tests.py

# 运行特定测试
python run_tests.py --type indicators
python run_tests.py --type config
python run_tests.py --type logger

# 生成覆盖率报告
python run_tests.py --coverage
```

### 测试覆盖

- 技术指标测试: 11个测试类
- 配置管理测试: 11个测试类
- 日志系统测试: 1个测试类
- 数据验证测试: 10个测试类
- 可视化测试: 8个测试类
- 扩展指标测试: 7个测试类
- 缓存并发测试: 6个测试类

总计: 200+ 测试用例

---

## 附录

### 常用指数代码

- 000300.SH: 沪深300
- 000905.SH: 中证500
- 399006.SZ: 创业板指
- 000016.SH: 上证50

### 命令行参数速查

| 参数 | 说明 | 默认值 |
|-----|------|--------|
| `--top-n` | 返回Top N股票 | 20 |
| `--index-code` | 指数代码 | 000300.SH |
| `--holding-days` | 持有天数 | 10 |
| `--save-report` | 保存报告 | False |
| `--verbose` | 详细输出 | False |
| `--token` | API Token | 环境变量 |

### 性能指标

- **收益指标**: 总收益率、年化收益率、平均单笔收益
- **风险指标**: 最大回撤、夏普比率、Sortino比率、Calmar比率
- **交易统计**: 胜率、盈亏比、平均盈利、平均亏损
- **稳定性指标**: Walk-Forward相关性、参数稳定性

### 免责声明

本系统仅供学习和研究使用，不构成投资建议。回测结果基于历史数据，实际交易受滑点、手续费、流动性等因素影响，可能与回测结果存在差异。参数优化可能存在过拟合风险，建议使用Walk-Forward分析验证参数稳定性。投资有风险，入市需谨慎。

---

**最后更新**: 2026-01-21
