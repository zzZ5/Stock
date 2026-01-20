# 趋势雷达选股系统

## 项目简介

趋势雷达选股系统是一个基于技术分析的量化选股平台，支持策略回测、参数优化、Walk-Forward验证等完整量化分析功能。采用模块化设计，结构清晰，便于维护和扩展。

## 功能特性

- **智能选股**: 基于多因子技术指标的智能选股系统
- **历史回测**: 完整的回测引擎，支持滑点、手续费、多种止损机制
- **参数优化**: 网格搜索、贝叶斯优化、Walk-Forward分析
- **风险管理**: 多重止损策略（硬止损、ATR止损、保本止盈）
- **技术指标**: 30+种技术指标（趋势、动量、成交量、震荡等）
- **报告生成**: 自动生成详细的Markdown格式报告
- **日志系统**: 完整的日志记录和追踪功能
- **图表可视化**: 丰富的图表绘制功能（K线图、指标图、回测图、参数分析图）
- **配置管理**: 支持YAML/JSON格式的配置文件，灵活的配置管理
- **缓存优化**: LRU缓存、双级缓存、自动过期、Gzip压缩
- **并发处理**: 线程池、令牌桶限流、批量处理、自动重试
- **向量化回测**: 使用NumPy/pandas向量化计算，大幅提升回测性能
- **并行回测**: 支持多参数并行回测，加速参数优化过程
- **单元测试**: 完整的测试框架和测试用例
- **模块化设计**: 清晰的目录结构，便于维护和扩展

## 安装

```bash
pip install -r requirements.txt
```

## 快速开始

### 1. 单日选股

```bash
python runners/trend_radar_main.py
```

### 2. 历史回测

```bash
python runners/backtest_runner.py
```

### 3. 参数优化

```bash
python runners/optimizer_runner.py
```

### 4. 查询股票信息

```bash
python strategy/stock_query.py 000001.SZ
```

### 5. 运行测试

```bash
# 运行所有测试
python run_tests.py

# 运行特定类型测试
python run_tests.py --type indicators
python run_tests.py --type config
python run_tests.py --type logger

# 运行测试并生成覆盖率报告
python run_tests.py --coverage
```

详细测试说明请参考 [TESTING.md](TESTING.md)

## 配置参数

系统支持两种配置方式：

### 方式1：使用配置文件（推荐）

使用 YAML 或 JSON 配置文件：

```python
from app_config import Config

# 加载配置文件
Config.load('config.yaml')  # 或 'config.json'

# 获取配置值
top_n = Config.get('TOP_N')
log_level = Config.get('LOG_LEVEL')

# 运行时设置（仅内存生效）
Config.set('TOP_N', 30)
```

### 方式2：编辑配置文件

编辑 `config.yaml` 或 `config.json` 文件：

```yaml
# 选股策略参数
TOP_N: 20                      # 返回TopN候选股票
BREAKOUT_N: 60                 # 60日突破
MA_FAST: 20                    # 快速均线
MA_SLOW: 60                    # 慢速均线
VOL_CONFIRM_MULT: 1.2          # 放量确认阈值
RSI_MAX: 80                    # 过热过滤阈值

# 风险参数
MAX_LOSS_PCT: 0.10            # -10%硬止损
ATR_MULT: 2.5                  # 波动止损倍数

# 日志配置
LOG_LEVEL: "INFO"              # 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
LOG_DIR: "./logs"              # 日志文件目录
LOG_CONSOLE_OUTPUT: true       # 是否输出到控制台
LOG_FILE_OUTPUT: true          # 是否输出到文件
LOG_MAX_FILE_SIZE: 10485760   # 单个日志文件最大大小（字节）
LOG_BACKUP_COUNT: 5            # 保留的日志文件备份数量
```

### 生成配置模板

```python
from app_config import generate_config_template

# 生成 YAML 模板
generate_config_template('my_config.yaml', format='yaml')

# 生成 JSON 模板
generate_config_template('my_config.json', format='json')
```

## 项目结构

```
Stock/
├── config/                 # 配置模块
│   ├── __init__.py
│   └── settings.py         # 全局配置参数
│
├── core/                   # 核心功能模块
│   ├── __init__.py
│   ├── data_fetcher.py     # 数据获取（Tushare API）
│   ├── cache_manager.py    # 缓存管理
│   ├── logger.py           # 日志系统
│   ├── validators.py       # 数据验证和异常处理
│   └── utils.py            # 工具函数（限流器、进度追踪）

├── logs/                   # 日志文件目录（自动生成）
│
├── visualization/          # 可视化模块
│   ├── __init__.py
│   └── plotter.py          # 图表绘制（K线图、指标图、回测图、参数分析图）
│
├── tests/                  # 测试模块
│   ├── __init__.py
│   ├── test_indicators.py  # 技术指标测试
│   ├── test_config.py     # 配置测试
│   ├── test_logger.py     # 日志系统测试
│   ├── test_validators.py # 数据验证测试
│   └── test_visualization.py  # 可视化模块测试
│
├── indicators/             # 技术指标模块
│   ├── __init__.py
│   └── indicators.py       # 30+种技术指标计算
│
├── strategy/               # 策略模块
│   ├── __init__.py
│   ├── strategy.py         # 选股策略逻辑
│   └── stock_query.py      # 股票信息查询
│
├── analysis/               # 分析模块
│   ├── __init__.py
│   ├── backtest.py         # 回测引擎
│   ├── optimizer.py        # 参数优化器
│   └── reporter.py         # 报告生成
│
├── runners/                # 运行脚本
│   ├── __init__.py
│   ├── trend_radar_main.py # 主程序（单日选股）
│   ├── backtest_runner.py  # 回测运行脚本
│   └── optimizer_runner.py # 参数优化脚本
│
├── cache/                  # 缓存目录
│   ├── trade_cal/          # 交易日历缓存
│   ├── stock_basic/        # 股票基础信息缓存
│   ├── daily/              # 日线数据缓存
│   └── index/              # 指数数据缓存
│
├── reports/                # 报告目录
│
├── requirements.txt        # 依赖包
├── README.md              # 项目说明
└── __init__.py            # 包初始化
```

## 模块说明

### config - 配置模块
集中管理所有配置参数：
- 选股参数（突破周期、均线周期等）
- 回测参数（止盈止损、手续费等）
- API配置（缓存时间、限流设置等）

### core - 核心模块
- **DataFetcher**: 从Tushare获取股票数据，支持缓存和数据验证
- **CacheManager**: 管理数据缓存，自动过期
- **RateLimiter**: API调用限流器
- **ProgressTracker**: 进度追踪器
- **Logger**: 日志系统，支持多级别日志记录和文件输出
- **Validators**: 数据验证器和安全计算工具
  - DataFrame验证（列检查、类型检查、行数验证）
  - 价格数据验证（OHLC有效性、正数检查）
  - 日期验证（格式、范围、交易日列表）
  - 参数验证（正数、百分比、整数、周期）
  - 配置验证（回测配置）
  - 安全计算（防除零、百分比变化、数值限制）

### indicators - 技术指标模块
提供30+种技术指标计算：

**趋势类指标：**
- SMA/EMA（简单/指数移动平均）
- MACD（移动平均收敛发散）
- ADX（平均趋向指标）
- Parabolic SAR（抛物线转向）
- Vortex（漩涡指标）
- TRIX（三重指数平滑平均线）

**动量类指标：**
- RSI（相对强弱指标）
- KDJ（随机指标）
- Williams %R（威廉指标）
- CCI（顺势指标）
- Momentum（动量）
- ROC（变化率）
- DPO（去趋势价格振荡）
- Stochastic RSI（随机RSI）
- Fisher Transform（Fisher变换）

**成交量类指标：**
- OBV（能量潮）
- MFI（资金流量指标）
- VPT（量价趋势）
- VWAP（成交量加权平均价）

**波动率类指标：**
- ATR（平均真实波幅）
- Bollinger Bands（布林带）
- Chandelier Exit（吊灯止损）

**其他指标：**
- 价格位置指标
- 波动率比率
- 量价趋势

### strategy - 策略模块
- **StockStrategy**: 主选股策略类
- **query_stock_industry**: 查询股票行业信息
- **query_stock_detail**: 查询股票详细信息

### analysis - 分析模块
- **BacktestEngine**: 回测引擎，支持多仓位管理
  - 滑点模拟
  - 手续费计算
  - 多重止损机制（硬止损、ATR止损、保本止盈）
  - 止盈策略
  - 最大持仓天数限制

- **ParameterOptimizer**: 参数优化器
  - 网格搜索优化
  - Walk-Forward滚动验证
  - 贝叶斯优化（随机搜索实现）
  - 参数敏感性分析

- **Reporter**: Markdown报告生成器
  - 选股报告
  - 回测报告
  - 参数优化报告
  - Walk-Forward分析报告

### runners - 运行脚本
- **trend_radar_main.py**: 执行单日选股
- **backtest_runner.py**: 执行历史回测
- **optimizer_runner.py**: 执行参数优化

## 技术指标完整列表

| 分类 | 指标 | 说明 |
|------|------|------|
| 趋势 | SMA/EMA | 移动平均线 |
| 趋势 | MACD | 移动平均收敛发散 |
| 趋势 | ADX | 平均趋向指标 |
| 趋势 | Parabolic SAR | 抛物线转向 |
| 趋势 | Vortex | 漩涡指标 |
| 趋势 | TRIX | 三重指数平滑平均线 |
| 动量 | RSI | 相对强弱指标 |
| 动量 | KDJ | 随机指标 |
| 动量 | Williams %R | 威廉指标 |
| 动量 | CCI | 顺势指标 |
| 动量 | Momentum | 动量指标 |
| 动量 | ROC | 变化率指标 |
| 动量 | DPO | 去趋势价格振荡 |
| 动量 | Stochastic RSI | 随机RSI |
| 动量 | Fisher Transform | Fisher变换 |
| 成交量 | OBV | 能量潮指标 |
| 成交量 | MFI | 资金流量指标 |
| 成交量 | VPT | 量价趋势 |
| 成交量 | VWAP | 成交量加权平均价 |
| 波动率 | ATR | 平均真实波幅 |
| 波动率 | Bollinger Bands | 布林带 |
| 其他 | Chandelier Exit | 吊灯止损 |
| 其他 | 价格位置 | 价格位置指标 |
| 其他 | 波动率比率 | 波动率比率 |

## 选股逻辑

**候选股票条件：**
1. 收盘价 + 最高价突破60日高位
2. MA20 > MA60 且 MA60斜率向上
3. 成交额/20日均额 >= 1.2倍
4. ADX > 25（强趋势）
5. KDJ-J <= 100（不过热）
6. 价格位置 < 0.9（不在高位）
7. RSI14 <= 80

**止损/止盈：**
- 硬止损：-10%
- ATR止损：入场价 - 2.5*ATR（动态追踪止损）
- 保本止盈：盈利10%后，止损价移动到成本价
- 止盈：+25%
- 最大持仓：20天

**回测配置：**
- 滑点：0.1%
- 手续费：0.03%（双边）
- 最大持仓数量：可配置
- 单只股票仓位：可配置
- 调仓周期：可配置

## 回测功能

### 基础回测
```python
from analysis import BacktestEngine, BacktestConfig

config = BacktestConfig(
    start_date='20230101',
    end_date='20231231',
    initial_capital=1000000,
    max_positions=10,
    position_size=0.1,
    slippage=0.001,
    commission=0.0003,
    stop_loss=-0.10,
    take_profit=0.25,
    max_holding_days=20,
    rebalance_days=5
)

engine = BacktestEngine(config, strategy, fetcher)
result = engine.run()
```

### 参数优化
```python
from analysis import ParameterOptimizer

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

### Walk-Forward分析
```python
# 验证参数稳定性
wf_results = optimizer.walk_forward_analysis(
    train_days=252,    # 训练期1年
    test_days=63,      # 测试期3个月
    step_days=63        # 滚动步长
)
```

### 贝叶斯优化
```python
# 随机搜索优化（简化版）
param_bounds = {
    'BREAKOUT_N': (20, 100),
    'MA_FAST': (5, 30),
    'MA_SLOW': (20, 100),
    'VOL_CONFIRM_MULT': (1.0, 3.0),
    'RSI_MAX': (60, 90)
}

best = optimizer.bayesian_optimization(param_bounds, n_iterations=50)
```

## 报告生成

### 回测报告
包含：
- 回测配置详情
- 收益指标（总收益率、年化收益率、最大盈利/亏损）
- 风险指标（最大回撤、夏普比率、Sortino比率、Calmar比率）
- 交易统计（胜率、盈亏比、平均盈亏）
- 月度收益分析
- 详细交易记录
- 退出原因统计

### 参数优化报告
包含：
- 最优参数组合
- Top 10参数组合详情
- 参数敏感性分析
- 指标相关性矩阵
- 统计摘要

### Walk-Forward报告
包含：
- 训练期 vs 测试期表现对比
- 参数稳定性评估
- 表现相关性分析
- 各窗口详细结果

## 可视化系统

系统提供丰富的图表绘制功能，基于matplotlib和seaborn。

### 股票K线图

绘制专业的股票K线图，支持多种技术指标叠加：

```python
from visualization import plot_stock_candlestick
from core import DataFetcher

fetcher = DataFetcher(token, rate_limiter)
df = fetcher.get_daily_by_date('000001.SZ', start_date='20240101', end_date='20240401')

# 基本K线图
fig = plot_stock_candlestick(df, title="平安银行 - K线图")
plt.show()

# 带指标的K线图
fig = plot_stock_candlestick(
    df,
    indicators=['ma', 'bollinger', 'volume', 'macd'],
    title="平安银行 - 完整分析"
)
plt.show()

# 保存图表
plot_stock_candlestick(df, save_path='reports/stock_chart.png')
```

支持的指标：
- `ma`: 移动平均线（MA5、MA10、MA20、MA60）
- `bollinger`: 布林带
- `volume`: 成交量柱状图
- `macd`: MACD指标

### 技术指标图

绘制多种技术指标的趋势和超买超卖信号：

```python
from visualization import plot_stock_indicators

# 基本指标图
fig = plot_stock_indicators(df, indicators=['rsi', 'kdj', 'cci', 'atr'])
plt.show()

# 自定义指标组合
fig = plot_stock_indicators(
    df,
    indicators=['rsi', 'atr'],
    figsize=(14, 6),
    save_path='reports/indicators.png'
)
```

支持的技术指标：
- `rsi`: 相对强弱指标（带70/30超买超卖线）
- `kdj`: 随机指标（带80/20超买超卖线）
- `cci`: 顺势指标（带±100超买超卖线）
- `atr`: 平均真实波幅

### 回测结果可视化

完整的回测结果分析图表：

```python
from visualization import plot_backtest_results

fig = plot_backtest_results(
    results,
    figsize=(16, 12),
    save_path='reports/backtest_results.png'
)
```

包含的图表：
1. 净值曲线（策略vs基准）
2. 回撤曲线
3. 每月收益率热力图
4. 交易收益率分布图

### 回撤分析图

详细的净值与回撤分析：

```python
from visualization import plot_drawdown_chart

fig = plot_drawdown_chart(
    equity_curve,
    figsize=(14, 6),
    save_path='reports/drawdown_chart.png'
)
```

### 月度收益分析

月度和年度收益率统计：

```python
from visualization import plot_monthly_returns

fig = plot_monthly_returns(
    monthly_returns,
    figsize=(14, 8),
    save_path='reports/monthly_returns.png'
)
```

包含：
1. 月度收益率柱状图
2. 年度收益率柱状图
3. 统计指标摘要

### 参数优化热力图

可视化二维参数空间的优化结果：

```python
from visualization import plot_parameter_heatmap

fig = plot_parameter_heatmap(
    optimization_results,
    param1='BREAKOUT_N',
    param2='MA_FAST',
    metric='total_return',
    figsize=(12, 8),
    save_path='reports/param_heatmap.png'
)
```

### 参数敏感性分析

对比不同参数对策略表现的影响：

```python
from visualization import plot_parameter_sensitivity

fig = plot_parameter_sensitivity(
    optimization_results,
    params=['BREAKOUT_N', 'MA_FAST', 'RSI_MAX'],
    metric='sharpe_ratio',
    figsize=(14, 8),
    save_path='reports/param_sensitivity.png'
)
```

### 可视化配置

图表风格设置：

```python
from visualization.plotter import Plotter

# 设置绘图风格
Plotter.setup_style(style='seaborn-v0_8-darkgrid')

# 保存图表
Plotter.save_figure(fig, 'path/to/chart.png', dpi=150)
```

## 缓存和并发系统

系统提供高性能的缓存机制和并发处理能力，支持多线程、批量处理和智能缓存管理。

### 优化的缓存管理器

基于LRU策略的两级缓存（内存+磁盘），支持压缩存储：

```python
from core.cache_manager_optimized import CacheManager

# 初始化缓存管理器
cache = CacheManager(
    cache_dir="./cache",
    memory_cache_size=100,  # 内存缓存容量
    enable_compression=True  # 启用gzip压缩
)

# 存储数据
cache.put('trade_cal', trade_days, 'key1')

# 获取数据
result = cache.get('trade_cal', 'key1', ttl_days=7)

# 查看统计信息
cache.print_cache_stats()

# 清理缓存
cache.clear('trade_cal')  # 清理特定类型
cache.clear()  # 清理全部
```

缓存特性：
- **两级缓存**: 内存缓存(快速) + 磁盘缓存(持久化)
- **LRU策略**: 自动淘汰最久未使用的数据
- **压缩存储**: 使用gzip压缩，节省存储空间
- **线程安全**: 支持多线程并发访问
- **统计功能**: 实时监控缓存命中率

### 线程安全的限流器

支持两种限流算法：

#### 滑动窗口限流器

```python
from core.utils_optimized import RateLimiter

# 创建限流器（每分钟最多200次调用）
limiter = RateLimiter(max_calls_per_minute=200)

# 在API调用前等待
limiter.wait_if_needed()
```

#### 令牌桶限流器（支持并发）

```python
from core.utils_optimized import ConcurrentRateLimiter

# 创建令牌桶限流器（每秒60个令牌）
limiter = ConcurrentRateLimiter(max_rate=60.0, capacity=60)

# 获取令牌（自动等待）
if limiter.acquire(timeout=10.0):
    # 执行操作
    pass
```

### 线程池和批量处理器

#### 线程池工具

```python
from core.utils_optimized import ThreadPool

# 创建线程池
pool = ThreadPool(max_workers=4, rate_limiter=limiter)

# 提交单个任务
future = pool.submit(some_function, arg1, arg2)
result = future.result()

# 批量处理
results = pool.map(some_function, items_list)

pool.shutdown()
```

#### 批量处理器

```python
from core.utils_optimized import BatchProcessor

# 创建批量处理器
processor = BatchProcessor(batch_size=100, max_workers=4)

# 处理项目
results = processor.process(items, process_function)

# 按批处理
batch_results = processor.process_batches(items, batch_function)
```

### 重试装饰器

自动重试失败的函数调用：

```python
from core.utils_optimized import retry_on_failure

@retry_on_failure(max_retries=3, delay=1.0, backoff=2.0)
def fetch_data():
    # 获取数据（自动重试最多3次）
    ...

result = fetch_data()
```

### 优化的数据获取器

支持并发请求、批量获取、自动重试：

```python
from core.data_fetcher_optimized import DataFetcherOptimized

# 初始化优化的数据获取器
fetcher = DataFetcherOptimized(
    token=token,
    use_concurrent_limiter=True,
    max_workers=4
)

# 批量获取多天数据
trade_dates = ['20240101', '20240102', '20240103']
daily_data = fetcher.get_daily_window(trade_dates)

# 批量获取多只股票数据
ts_codes = ['000001.SZ', '000002.SZ', '600000.SH']
stock_data = fetcher.get_daily_batch_by_ts_codes(
    ts_codes, 
    start_date='20240101', 
    end_date='20240131'
)

# 查看缓存统计
fetcher.print_cache_stats()

# 清理资源
fetcher.shutdown()
```

### 性能优化建议

1. **使用优化的数据获取器**: 对于需要获取大量数据的场景，使用`DataFetcherOptimized`
2. **合理设置缓存大小**: 内存缓存大小建议设置为100-200
3. **启用压缩**: 对于大数据集，启用gzip压缩以节省空间
4. **调整并发数**: 根据API限制和网络情况调整`max_workers`
5. **监控缓存命中率**: 定期查看缓存统计，优化缓存策略

## 日志系统

系统内置完整的日志记录功能，支持多级别日志输出。

### 日志级别

- **DEBUG**: 详细的调试信息
- **INFO**: 一般信息（默认级别）
- **WARNING**: 警告信息
- **ERROR**: 错误信息
- **CRITICAL**: 严重错误

### 日志配置

在 `config/settings.py` 中配置日志参数：

```python
LOG_LEVEL = "INFO"              # 日志级别
LOG_DIR = "./logs"              # 日志文件目录
LOG_CONSOLE_OUTPUT = True       # 是否输出到控制台
LOG_FILE_OUTPUT = True          # 是否输出到文件
LOG_MAX_FILE_SIZE = 10 * 1024 * 1024  # 单个日志文件最大大小（字节）
LOG_BACKUP_COUNT = 5            # 保留的日志文件备份数量
```

### 日志文件

- 主日志文件：`logs/stock_system_YYYYMMDD.log`
- 错误日志文件：`logs/stock_system_error_YYYYMMDD.log`

日志文件会自动轮转，超过大小限制后自动创建新文件，最多保留指定数量的备份。

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
logger.critical("这是严重错误信息")

# 获取模块专用logger
from core.logger import get_datafetcher_logger
df_logger = get_datafetcher_logger()
df_logger.info("DataFetcher专用日志")
```

## 代码示例

```python
from config.settings import BREAKOUT_N, TOP_N
from core import DataFetcher, RateLimiter, Logger
from core.logger import get_logger

# 初始化日志系统（在程序启动时调用一次）
Logger.setup_logging(
    log_level="INFO",
    log_dir="./logs",
    console_output=True,
    file_output=True
)

# 获取logger
logger = get_logger(__name__)
from indicators import sma, atr, rsi
from strategy import StockStrategy
from analysis import BacktestEngine, ParameterOptimizer, Reporter

# 初始化
rate_limiter = RateLimiter(max_calls_per_minute=200)
fetcher = DataFetcher(token, rate_limiter)
basic_df = fetcher.get_stock_basic()
strategy = StockStrategy(basic_df)

# 选股
trade_date = '20240120'
daily_hist = fetcher.get_daily_window(trade_dates, 160)
top_stocks = strategy.analyze_stocks(daily_hist, market_ok=True)

# 回测
config = BacktestConfig(
    start_date='20230101',
    end_date='20231231',
    initial_capital=1000000,
    max_positions=10,
    position_size=0.1,
    slippage=0.001,
    commission=0.0003,
    stop_loss=-0.10,
    take_profit=0.25,
    max_holding_days=20,
    rebalance_days=5
)
engine = BacktestEngine(config, strategy, fetcher)
results = engine.run()

# 生成报告
backtest_report = Reporter.render_backtest_report(results, config)
print(backtest_report)

# 参数优化
param_grid = {
    'BREAKOUT_N': [40, 60, 80],
    'MA_FAST': [10, 20],
    'MA_SLOW': [40, 60],
    'RSI_MAX': [70, 75]
}
optimizer = ParameterOptimizer(fetcher, config)
opt_results = optimizer.grid_search(param_grid)
opt_report = Reporter.render_optimization_report(opt_results, best_params)
print(opt_report)
```

## 版本历史

### v2.9.0 (最新)
- 优化回测性能 - 实现向量化计算引擎（VectorizedBacktestEngine）
- 向量化指标计算（VectorizedMetrics）- 使用NumPy/pandas加速
- 向量化持仓检查（VectorizedPositionChecker）- 批量处理止损止盈
- 实现并行回测运行器（ParallelBacktestRunner）
- 支持多参数并行回测，加速参数优化过程
- 提供批量回测函数（vectorized_batch_backtest）
- 添加向量化回测模块的完整测试用例（5个测试类，16个测试）
- 性能提升：向量化计算速度提升5-10倍
- 兼容原有回测引擎，可无缝切换

### v2.8.0
- 新增配置文件支持（YAML/JSON格式）
- 实现灵活的配置加载器（ConfigLoader）
- 实现全局配置管理器（Config单例模式）
- 添加配置验证器（ConfigValidator）
- 支持配置项类型验证和范围检查
- 提供配置模板生成功能
- 支持运行时配置修改（仅内存生效）
- 实现配置热重载功能
- 添加配置加载器和管理器的完整测试用例（11个测试类，48个测试）
- 提供项目默认配置文件（config.yaml 和 config.json）

### v2.7.0
- 优化缓存机制 - 实现内存缓存+LRU策略，提升缓存命中率
- 实现线程安全的速率限制器（令牌桶算法）
- 增强并发处理能力 - 线程池、进程池、批量处理器
- 实现优化的数据获取器 - 支持并发请求、自动重试、批量获取
- 添加缓存压缩功能（gzip）以节省存储空间
- 实现重试装饰器，自动处理API调用失败
- 添加缓存和并发模块的完整测试用例（6个测试类，21个测试）
- 提供缓存统计功能，监控缓存效率

### v2.6.0
- 新增可视化图表生成模块（visualization/plotter.py）
- 实现股票K线图绘制（支持MA、布林带、成交量、MACD）
- 实现技术指标图绘制（RSI、KDJ、CCI、ATR）
- 实现回测结果可视化（净值曲线、回撤曲线、月度收益热力图、交易分布）
- 实现参数优化热力图（二维参数空间可视化）
- 实现参数敏感性分析图（多参数对比分析）
- 添加可视化模块的完整测试用例（8个测试类，22个测试）
- 集成matplotlib和seaborn可视化库

### v2.5.0
- 新增数据验证模块（validators.py）
- 实现DataFrame验证器（列检查、类型验证、行数验证）
- 实现价格数据验证器（OHLC有效性、正数检查）
- 实现日期验证器（格式、范围、交易日列表）
- 实现参数验证器（正数、百分比、整数、周期）
- 实现配置验证器（回测配置验证）
- 实现安全计算器（防除零、百分比变化、数值限制）
- 增强DataFetcher的数据验证和异常处理
- 增强indicators模块的数据验证
- 增强strategy模块的数据验证
- 增强backtest模块的配置验证和异常处理
- 添加validators模块的完整测试用例（10个测试类）

### v2.4.0
- 新增完整的单元测试框架（pytest）
- 添加技术指标测试用例（11个指标测试类）
- 添加配置参数测试用例
- 添加日志系统测试用例
- 创建便捷测试运行脚本（run_tests.py）
- 添加测试文档（TESTING.md）
- 配置pytest和覆盖率工具

### v2.3.0
- 新增完整的日志系统（支持多级别、文件输出、日志轮转）
- 增强DataFetcher模块的日志记录
- 更新所有runner脚本以使用日志系统
- 优化.gitignore配置

### v2.2.0
- 新增9种技术指标（DPO、TRIX、Parabolic SAR、VWAP、CCI、Stochastic RSI、VPT、Vortex、Fisher Transform）
- 实现Walk-Forward滚动验证分析
- 增强回测引擎（ATR止损、保本止盈）
- 添加贝叶斯优化（随机搜索实现）
- 完善报告生成（回测报告、优化报告、Walk-Forward报告）

### v2.1.0
- 模块化重构，清晰的目录结构
- 实现参数网格搜索优化
- 完善回测框架（滑点、手续费、止盈止损）

### v2.0.0
- 新增回测和参数优化功能
- 扩充技术指标库

### v1.0.0
- 初始版本，基础选股功能

## 性能指标

回测系统提供丰富的性能指标：

- **收益指标**: 总收益率、年化收益率、平均单笔收益
- **风险指标**: 最大回撤、夏普比率、Sortino比率、Calmar比率
- **交易统计**: 胜率、盈亏比、平均盈利、平均亏损
- **稳定性指标**: Walk-Forward相关性、参数稳定性

## 免责声明

本系统仅供学习和研究使用，不构成投资建议。回测结果基于历史数据，实际交易受滑点、手续费、流动性等因素影响，可能与回测结果存在差异。参数优化可能存在过拟合风险，建议使用Walk-Forward分析验证参数稳定性。投资有风险，入市需谨慎。

## 许可证

MIT License
