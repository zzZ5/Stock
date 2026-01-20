# 趋势雷达选股系统 - 重构版

## 项目简介

趋势雷达选股系统是一个基于技术分析的量化选股平台，支持策略回测、参数优化等功能。本项目已完成模块化重构，结构更清晰，便于维护和扩展。

## 目录结构

```
Stock/
├── README.md                      # 项目说明
├── requirements.txt               # 依赖包
│
├── config/                       # 【配置层】
│   ├── settings.py               # 主配置（策略参数、缓存等）
│   ├── backtest_config.py        # 回测配置类
│   └── constants.py              # 常量定义
│
├── core/                         # 【核心层】
│   ├── data_manager.py           # 数据获取与管理
│   └── cache_manager.py          # 缓存管理
│
├── indicators/                   # 【指标层】
│   ├── basic.py                 # 基础指标（SMA、EMA、RSI等）
│   ├── trend.py                 # 趋势指标（ADX、KDJ等）
│   ├── oscillator.py            # 震荡指标（Williams %R等）
│   └── volatility.py            # 波动指标（布林带等）
│
├── strategy/                     # 【策略层】
│   ├── base_strategy.py         # 基础策略类
│   ├── signals.py               # 信号生成
│   └── stock_screening.py      # 股票筛选
│
├── backtest/                     # 【回测层】
│   ├── engine.py                # 回测引擎
│   └── runner.py                # 回测运行脚本
│
├── optimization/                 # 【优化层】
│   ├── optimizer.py             # 参数优化器
│   └── runner.py                # 优化运行脚本
│
├── utils/                        # 【工具层】
│   └── helpers.py               # 辅助函数
│
├── output/                       # 【输出层】
│   ├── reporter.py              # 报告生成
│   ├── reports/                 # Markdown报告
│   └── results/                # 回测/优化结果
│
├── cache/                        # 【数据缓存】
│   └── tushare/                # Tushare缓存
│
├── scripts/                      # 【脚本层】
│   ├── main.py                  # 主程序（单日选股）
│   ├── backtest_run.py          # 回测运行
│   └── optimize_run.py          # 优化运行
│
└── docs/                         # 【文档层】
    ├── USAGE_GUIDE.md           # 使用指南
    └── API_REFERENCE.md          # API参考
```

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 设置Token

```bash
# Windows
set TUSHARE_TOKEN=your_token_here

# Linux/Mac
export TUSHARE_TOKEN=your_token_here
```

### 运行程序

```bash
# 1. 单日选股
python scripts/main.py

# 2. 历史回测
python scripts/backtest_run.py

# 3. 参数优化
python scripts/optimize_run.py
```

## 模块说明

### 配置层 (config/)
统一管理所有配置参数，便于维护和修改。

- `settings.py`: 策略参数、缓存参数等
- `backtest_config.py`: 回测配置类
- `constants.py`: 常量定义（指标阈值、市场代码等）

### 核心层 (core/)
提供基础数据和管理功能。

- `data_manager.py`: Tushare API数据获取
- `cache_manager.py`: 数据缓存管理

### 指标层 (indicators/)
技术指标计算库，按类型分类。

- `basic.py`: SMA、EMA、MACD、RSI、ATR等
- `trend.py`: ADX、KDJ等趋势指标
- `oscillator.py`: Williams %R、CCI等震荡指标
- `volatility.py`: 布林带、波动率等

### 策略层 (strategy/)
选股和交易策略实现。

- `base_strategy.py`: 基础策略类
- `stock_screening.py`: 股票筛选功能

### 回测层 (backtest/)
历史回测系统。

- `engine.py`: 完整的回测引擎
- `runner.py`: 回测运行脚本

### 优化层 (optimization/)
参数优化系统。

- `optimizer.py`: 参数优化器（网格搜索、Walk-Forward）

### 工具层 (utils/)
通用工具函数。

- `helpers.py`: 辅助函数、限流器等

### 输出层 (output/)
报告和结果管理。

- `reporter.py`: 报告生成器
- `reports/`: Markdown报告
- `results/`: 回测和优化结果

## Import规则

### 同层导入
```python
from .engine import BacktestEngine
```

### 跨层导入
```python
from config.settings import TOP_N, BREAKOUT_N
from core.data_manager import DataManager
from indicators.basic import sma, rsi
from strategy.base_strategy import StockStrategy
```

## 策略说明

### 选股条件

**候选股票：**
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

## 回测配置

```python
BacktestConfig(
    start_date="20240101",      # 开始日期
    end_date="20241231",        # 结束日期
    initial_capital=1000000.0,  # 初始资金
    max_positions=5,             # 最大持仓数
    position_size=0.15,         # 单股仓位
    slippage=0.001,             # 滑点
    commission=0.0003,         # 手续费
    stop_loss=-0.10,           # 止损
    take_profit=0.25,          # 止盈
    max_holding_days=20,        # 最大持仓天数
    rebalance_days=5           # 选股间隔
)
```

## 技术指标

| 指标 | 模块 | 说明 |
|------|------|------|
| SMA | basic | 简单移动平均 |
| EMA | basic | 指数移动平均 |
| MACD | basic | 移动平均收敛发散 |
| RSI | basic | 相对强弱指标 |
| ATR | basic | 平均真实波幅 |
| ADX | trend | 平均趋向指标 |
| KDJ | trend | 随机指标 |
| Williams %R | oscillator | 威廉指标 |
| Bollinger Bands | volatility | 布林带 |

## 回测指标

| 指标 | 说明 | 优秀值 |
|------|------|--------|
| 年化收益 | 年化回报率 | > 20% |
| 夏普比率 | 风险调整收益 | > 1.5 |
| 最大回撤 | 最大亏损 | < 20% |
| 胜率 | 盈利交易占比 | > 50% |
| 盈亏比 | 盈利/亏损 | > 2.0 |

## 开发指南

### 添加新指标

1. 在`indicators/`对应分类下添加函数
2. 在`indicators/__init__.py`中导出
3. 在策略中使用

### 添加新策略

1. 在`strategy/`目录下创建新文件
2. 继承或参考`base_strategy.py`
3. 在脚本中使用

### 添加新回测

1. 修改`backtest/engine.py`或创建新引擎
2. 在`scripts/`下创建运行脚本

## 注意事项

1. **API限流**：Tushare有调用限制，已内置限流
2. **缓存机制**：数据缓存到`cache/tushare/`
3. **首次运行**：需要下载历史数据
4. **实盘风险**：本系统仅供学习，实盘需谨慎

## 参考文档

- [项目结构说明](PROJECT_STRUCTURE.md) - 详细的目录结构说明
- [重构方案](REFACTOR_PLAN.md) - Import路径更新规则
- [使用指南](docs/USAGE_GUIDE.md) - 详细使用方法

## 更新日志

### v2.0.0 (重构版)
- 完成模块化重构
- 目录结构按功能和业务逻辑分类
- 新增回测和参数优化功能
- 扩展技术指标库
- 完善文档

### v1.0.0 (原始版)
- 基础选股功能
- 技术指标计算
- 报告生成

## 许可证

MIT License
