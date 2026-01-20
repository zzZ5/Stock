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

## 配置参数

在 `config/settings.py` 中修改策略参数：

```python
# 选股策略参数
TOP_N = 20                      # 返回TopN候选股票
BREAKOUT_N = 60                 # 60日突破
MA_FAST = 20                    # 快速均线
MA_SLOW = 60                    # 慢速均线
VOL_CONFIRM_MULT = 1.2          # 放量确认阈值
RSI_MAX = 80                    # 过热过滤阈值

# 风险参数
MAX_LOSS_PCT = -0.10           # -10%硬止损
ATR_MULT = 2.5                  # 波动止损倍数

# 日志配置
LOG_LEVEL = "INFO"              # 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
LOG_DIR = "./logs"              # 日志文件目录
LOG_CONSOLE_OUTPUT = True       # 是否输出到控制台
LOG_FILE_OUTPUT = True          # 是否输出到文件
LOG_MAX_FILE_SIZE = 10 * 1024 * 1024  # 单个日志文件最大大小（字节）
LOG_BACKUP_COUNT = 5            # 保留的日志文件备份数量
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
│   └── utils.py            # 工具函数（限流器、进度追踪）

├── logs/                   # 日志文件目录（自动生成）
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
- **DataFetcher**: 从Tushare获取股票数据，支持缓存
- **CacheManager**: 管理数据缓存，自动过期
- **RateLimiter**: API调用限流器
- **ProgressTracker**: 进度追踪器
- **Logger**: 日志系统，支持多级别日志记录和文件输出

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

### v2.3.0 (最新)
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
