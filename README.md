# 趋势雷达选股系统

## 项目简介

趋势雷达选股系统是一个基于技术分析的量化选股平台，支持策略回测、参数优化等功能。采用模块化设计，结构清晰，便于维护和扩展。

## 功能特性

- **选股策略**: 基于技术指标的智能选股系统
- **回测系统**: 完整的历史数据回测功能
- **参数优化**: 网格搜索优化策略参数
- **技术指标**: 18+常用技术指标
- **报告生成**: 自动生成Markdown格式报告
- **模块化设计**: 清晰的目录结构，便于维护

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
│   └── utils.py            # 工具函数（限流器、进度追踪）
│
├── indicators/             # 技术指标模块
│   ├── __init__.py
│   └── indicators.py       # 18+种技术指标计算
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

### indicators - 技术指标模块
提供18+种技术指标计算：
- 趋势类：MA、EMA、MACD、ADX
- 动量类：RSI、KDJ、Williams %R、CCI
- 成交量类：OBV、MFI、VPT
- 波动率类：ATR、布林带
- 其他：Chandelier Exit、价格位置、ROC

### strategy - 策略模块
- **StockStrategy**: 主选股策略类
- **query_stock_industry**: 查询股票行业信息
- **query_stock_detail**: 查询股票详细信息

### analysis - 分析模块
- **BacktestEngine**: 回测引擎，支持多仓位管理
- **ParameterOptimizer**: 参数优化器（网格搜索、Walk-Forward）
- **Reporter**: Markdown报告生成器

### runners - 运行脚本
- **trend_radar_main.py**: 执行单日选股
- **backtest_runner.py**: 执行历史回测
- **optimizer_runner.py**: 执行参数优化

## 技术指标

| 指标 | 说明 |
|------|------|
| SMA/EMA | 移动平均线 |
| MACD | 移动平均收敛发散 |
| RSI | 相对强弱指标 |
| ATR | 平均真实波幅 |
| ADX | 平均趋向指标 |
| KDJ | 随机指标 |
| Williams %R | 威廉指标 |
| Bollinger Bands | 布林带 |
| OBV | 能量潮指标 |
| MFI | 资金流量指标 |

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
- ATR止损：入场价 - 2.5*ATR
- 止盈：+25%
- 最大持仓：20天

## 代码示例

```python
from config.settings import BREAKOUT_N, TOP_N
from core import DataFetcher, RateLimiter
from indicators import sma, atr, rsi
from strategy import StockStrategy
from analysis import BacktestEngine, ParameterOptimizer, Reporter

# 初始化
rate_limiter = RateLimiter(max_calls_per_minute=200)
fetcher = DataFetcher(token, rate_limiter)
strategy = StockStrategy(basic_df)

# 选股
top_stocks = strategy.select_top_stocks(df, trade_date, TOP_N)

# 回测
backtest = BacktestEngine(config)
results = backtest.run(trade_dates)

# 优化
optimizer = ParameterOptimizer(fetcher, backtest_config)
best_params = optimizer.grid_search()
```

## 版本历史

- **v2.1.0**: 模块化重构，清晰的目录结构
- **v2.0.0**: 新增回测和参数优化功能
- **v1.0.0**: 初始版本

## 免责声明

本系统仅供学习和研究使用，不构成投资建议。投资有风险，入市需谨慎。

## 许可证

MIT License
