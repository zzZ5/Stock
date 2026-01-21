# 趋势雷达选股系统

## 项目简介

趋势雷达选股系统是一个基于技术分析的量化选股平台，支持策略回测、参数优化、Walk-Forward验证等完整量化分析功能。采用模块化设计，结构清晰，便于维护和扩展。

## 功能特性

### 核心功能
- **智能选股**: 基于多因子技术指标的智能选股系统
- **历史回测**: 完整的回测引擎，支持滑点、手续费、多种止损机制
- **参数优化**: 网格搜索、贝叶斯优化、Walk-Forward分析
- **风险管理**: 多重止损策略（硬止损、ATR止损、保本止盈）

### 技术特性
- **技术指标**: 60+种技术指标（趋势、动量、成交量、波动率等）
- **向量化回测**: 使用NumPy/pandas向量化计算，性能提升5-10倍
- **并行回测**: 支持多参数并行回测，加速参数优化过程
- **缓存优化**: LRU缓存、双级缓存、自动过期、Gzip压缩
- **并发处理**: 线程池、令牌桶限流、批量处理、自动重试

### 工程特性
- **配置管理**: 支持YAML/JSON格式的配置文件，灵活的配置管理
- **日志系统**: 完整的日志记录和追踪功能
- **图表可视化**: 丰富的图表绘制功能（K线图、指标图、回测图、参数分析图）
- **报告生成**: 自动生成详细的Markdown格式报告
- **单元测试**: 完整的测试框架和测试用例（200+测试）
- **模块化设计**: 清晰的目录结构，便于维护和扩展

## 安装

```bash
pip install -r requirements.txt
```

## 快速开始

### 方式1：交互式菜单（推荐新手）
```bash
python runners/interactive_menu.py
```

### 方式2：命令行方式
```bash
# 使用默认配置运行
python runners/trend_radar_main.py

# 自定义参数
python runners/trend_radar_main.py --top-n 10 --index-code 000905.SH --holding-days 10
```

### 其他功能
```bash
# 历史回测
python runners/backtest_runner.py

# 参数优化
python runners/optimizer_runner.py
```

## 文档导航

| 文档 | 说明 | 适用人群 |
|-----|------|---------|
| **[QUICK_START.md](QUICK_START.md)** | 5分钟快速上手 | 新手 |
| **[DOCUMENTATION.md](DOCUMENTATION.md)** | 完整文档索引 | 所有用户 |
| **[README.md](README.md)** | 项目详细说明 | 深入了解 |
| **[PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)** | 项目结构说明 | 开发者 |

### 推荐阅读顺序

1. **新手入门**: QUICK_START.md
2. **了解项目**: README.md
3. **深入使用**: DOCUMENTATION.md
4. **扩展开发**: PROJECT_STRUCTURE.md

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
│   ├── backtest_vectorized.py # 向量化回测引擎
│   ├── optimizer.py        # 参数优化器
│   └── reporter.py         # 报告生成
│
├── runners/                # 运行脚本
│   ├── __init__.py
│   ├── interactive_menu.py # 交互式菜单（推荐）
│   ├── trend_radar_main.py # 主程序（整合版）
│   ├── backtest_runner.py  # 回测运行脚本
│   ├── backtest_demo.py    # 回测演示
│   └── optimizer_runner.py # 参数优化脚本
│
├── ARCHIVE/                # 归档文档
│   └── ARCHIVED_DOCS.md    # 已归档文档索引
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
├── README.md              # 项目说明（本文档）
├── DOCUMENTATION.md       # 完整文档索引（推荐查阅）
└── __init__.py            # 包初始化
```

## 模块说明

### config - 配置模块
集中管理所有配置参数，支持YAML/JSON格式：
- 选股参数（突破周期、均线周期等）
- 回测参数（止盈止损、手续费等）
- API配置（缓存时间、限流设置等）

### core - 核心模块
- **DataFetcher**: 从Tushare获取股票数据，支持缓存和数据验证
- **CacheManager**: 管理数据缓存，LRU策略、压缩存储
- **RateLimiter**: API调用限流器
- **ProgressTracker**: 进度追踪器
- **Logger**: 日志系统，支持多级别日志记录和文件输出
- **Validators**: 数据验证器和安全计算工具
- **数据质量**: 数据质量检查和异常检测

### indicators - 技术指标模块
提供60+种技术指标计算：
- **趋势类**: SMA/EMA, MACD, ADX, Parabolic SAR, Vortex, TRIX
- **动量类**: RSI, KDJ, Williams %R, CCI, Momentum, ROC, DPO
- **成交量类**: OBV, MFI, VPT, VWAP
- **波动率类**: ATR, Bollinger Bands, Chandelier Exit
- **其他**: 价格位置、波动率比率、量价趋势

### strategy - 策略模块
- **StockStrategy**: 主选股策略类
- **stock_query**: 股票信息查询

### analysis - 分析模块
- **BacktestEngine**: 回测引擎（基础版）
- **VectorizedBacktestEngine**: 向量化回测引擎（高性能）
- **ParallelBacktestRunner**: 并行回测运行器
- **ParameterOptimizer**: 参数优化器（网格搜索、贝叶斯、Walk-Forward）
- **Reporter**: Markdown报告生成器

### visualization - 可视化模块
- **plot_stock_candlestick**: 股票K线图
- **plot_stock_indicators**: 技术指标图
- **plot_backtest_results**: 回测结果可视化
- **plot_parameter_heatmap**: 参数优化热力图
- **plot_monthly_returns**: 月度收益分析

### runners - 运行脚本
- **interactive_menu.py**: 交互式菜单
- **trend_radar_main.py**: 主程序（整合版）
- **backtest_runner.py**: 回测运行脚本
- **backtest_demo.py**: 回测演示
- **optimizer_runner.py**: 参数优化脚本

### tests - 测试模块
- 技术指标测试
- 配置管理测试
- 日志系统测试
- 数据验证测试
- 可视化测试
- 缓存并发测试
- 扩展指标测试

总计: 200+ 测试用例

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

**详细文档请参考 [DOCUMENTATION.md](DOCUMENTATION.md)**

## 可视化系统

基于matplotlib和seaborn的图表绘制功能：

### 股票K线图
```python
from visualization import plot_stock_candlestick

fig = plot_stock_candlestick(
    df,
    indicators=['ma', 'bollinger', 'volume', 'macd'],
    title="股票分析"
)
```

### 回测结果可视化
```python
from visualization import plot_backtest_results

fig = plot_backtest_results(
    results,
    figsize=(16, 12)
)
```

包含：净值曲线、回撤曲线、月度收益热力图、交易分布

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

**详细文档请参考 [DOCUMENTATION.md](DOCUMENTATION.md)**

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

### v3.0.0 (最新)
- 添加30+种新技术指标，总计60+种指标
- 新增高级移动平均线：WMA、DEMA、TEMA、Hull MA
- 新增趋势指标：SuperTrend、Ichimoku Kinko Hyo、Donchian Channels、Pivot Points、Aroon、Decycler
- 新增包络线指标：Acceleration Bands、Envelope SMA
- 新增背离指标：RSI Divergence
- 新增成交量指标：Volume Weighted MA、Money Flow Ratio、Ease of Movement、Standardized Volume、Volume Profile
- 新增动量指标：Mass Index、Ultimate Oscillator
- 新增形态指标：ZigZag（摆动点检测）
- 新增回归指标：Linear Regression Slope/Intercept
- 新增组合指标：Squeeze Momentum
- 集成scipy科学计算库
- 添加扩展技术指标的完整测试用例（7个测试类，27个测试）

### v2.9.0
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
