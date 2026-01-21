# 趋势雷达选股系统 - 完整文档索引

## 📚 文档导航

本目录包含项目的所有文档，按照使用场景分类。

---

## 🚀 快速开始（新手必读）

### [快速开始指南](#快速开始指南)
- **5分钟快速上手**
- 常见使用场景
- 故障排除

### [用户手册](#用户手册)
- 完整功能介绍
- 配置参数说明
- 命令行参数详解

---

## 📖 核心功能文档

### [趋势雷达选股系统](#趋势雷达选股系统)
- 选股逻辑详解
- 评分系统说明
- 市场环境判断

### [回测系统指南](#回测系统指南)
- 回测引擎使用
- 参数优化方法
- Walk-Forward验证

### [演示指南](#演示指南)
- 功能演示脚本
- 使用示例代码

---

## 🔧 优化与进阶

### [系统优化说明](#系统优化说明)
- 性能优化方案
- 向量化计算
- 缓存优化

### [趋势雷达优化](#趋势雷达优化)
- 选股逻辑改进
- 多因子评分
- 动态参数调整

---

## 🧪 测试文档

### [测试指南](#测试指南)
- 单元测试说明
- 测试覆盖率
- 运行测试

---

## 📖 快速开始指南

本文档帮助你快速开始使用趋势雷达选股系统。

## 🚀 一键运行

### 最简单的方式
```bash
# 使用交互式菜单（推荐新手）
python runners/interactive_menu.py
```

### 命令行方式
```bash
# 使用默认配置运行
python runners/trend_radar_user_friendly.py
```

## 📋 常用命令

### 选股
```bash
# 默认配置
python runners/trend_radar_user_friendly.py

# 自定义Top N
python runners/trend_radar_user_friendly.py --top-n 10

# 使用不同指数
python runners/trend_radar_user_friendly.py --index-code 000905.SH
```

### 回测
```bash
# 运行回测
python runners/backtest_runner.py

# 运行演示
python runners/backtest_demo.py
```

### 参数优化
```bash
# 运行优化
python runners/optimizer_runner.py
```

## 🎨 输出说明

### 彩色标识
- ✓ <span style="color:green">绿色</span> - 成功/完成
- ⚠ <span style="color:orange">黄色</span> - 警告/注意
- ✗ <span style="color:red">红色</span> - 错误/失败
- ℹ <span style="color:blue">蓝色</span> - 信息/提示
- → <span style="color:cyan">青色</span> - 进度/进行中

## 💡 快速技巧

1. **首次使用**: 先运行 `python runners/interactive_menu.py` 熟悉系统
2. **测试选股**: 使用 `--top-n 5` 快速查看结果
3. **定时运行**: 使用 `--quiet` 模式配合定时任务
4. **查看报告**: 报告保存在 `reports/` 目录

## ⚠️ 注意事项

1. **API限流**: 默认每分钟200次调用，勿频繁运行
2. **数据延迟**: 免费版可能有1-2天延迟
3. **风险提示**: 选股结果仅供参考，不构成投资建议

## 🆘 常见问题

### Q: 为什么没有选中股票？
A: 可能原因：
- 市场环境较差（熊市）
- 选股标准过高
- 数据不足或异常

### Q: 如何调整选股标准？
A: 修改 `config/settings.py` 中的参数

### Q: 报告保存在哪里？
A: 默认保存在 `./reports/` 目录

### Q: Token从哪里获取？
A: 访问 https://tushare.pro 注册获取

---

## 📖 用户手册

本文档提供完整的系统使用说明。

## 🎯 功能概述

### 1. 智能选股
基于多因子技术指标的智能选股系统
- 8大信号因子
- 4大类评分系统
- 动态市场环境判断

### 2. 历史回测
完整的回测引擎
- 滑点模拟
- 手续费计算
- 多重止损机制
- 向量化计算

### 3. 参数优化
- 网格搜索
- 贝叶斯优化
- Walk-Forward验证

### 4. 风险管理
- 硬止损
- ATR止损
- 保本止盈
- 三层止损体系

## ⚙️ 配置参数

### 选股参数
```python
TOP_N = 20                      # 返回TopN候选股票
BREAKOUT_N = 60                 # 60日突破
MA_FAST = 20                    # 快速均线
MA_SLOW = 60                    # 慢速均线
VOL_CONFIRM_MULT = 1.2          # 放量确认阈值
RSI_MAX = 80                    # 过热过滤阈值
```

### 风险参数
```python
MAX_LOSS_PCT = 0.10            # -10%硬止损
ATR_N = 14                      # ATR计算周期
ATR_MULT = 2.5                  # 波动止损倍数
```

### 回测参数
```python
DEFAULT_HOLDING_DAYS = 10       # 默认持仓天数
```

## 📊 评分系统

### 趋势因子 (35%)
- 突破强度 (12分)
- 均线结构 (8分)
- ADX强度 (8分)
- MACD信号 (7分)

### 动量因子 (25%)
- 5日动量 (10分)
- 20日动量 (10分)
- OBV量价 (5分)

### 量能因子 (20%)
- 量能确认 (12分)
- 量能强度 (8分)

### 风险因子 (20%)
- RSI位置 (6分)
- 价格位置 (6分)
- 波动率 (4分)
- 止损距离 (4分)

## 🎯 选股条件

### 候选股（必须满足）
- 至少5个信号因子
- RSI不超过80
- 不处于过高位置
- 波动率不过大

### 观察股（接近候选）
- 至少4个信号因子
- 距突破价≤2%
- 未过热

## 📖 趋势雷达选股系统

本文档详细介绍趋势雷达选股系统的选股逻辑。

## 🎯 选股核心逻辑

### 市场环境判断

系统根据以下指标判断市场环境：

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

### 8大信号因子

1. **突破信号**: 收盘价+最高价突破60日高位
2. **趋势结构**: MA20 > MA60 且 MA60斜率向上
3. **量能确认**: 成交额/20日均额 >= 阈值
4. **趋势强度**: ADX >= 25（强趋势）
5. **动量强劲**: 5日和20日动量均上涨
6. **MACD金叉**: MACD和MACD柱状图均>0
7. **量能持续性**: 3日内有2天放量
8. **价格位置**: 在布林带上轨附近

### 止损体系

**三层止损：**

1. **硬止损**: -10% （最后一道防线）
2. **ATR止损**: 入场价 - 2.5*ATR（动态追踪）
3. **布林带止损**: 布林带下轨（趋势支撑）

**止盈策略：**
- 保本止盈: 盈利10%后，止损价移至成本价
- 止盈: +25%

## 📖 回测系统指南

本文档详细介绍回测系统的使用方法。

## 🎯 回测引擎

### 基础回测

```python
from analysis.backtest import BacktestEngine, BacktestConfig

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

### 向量化回测（高性能）

```python
from analysis.backtest_vectorized import VectorizedBacktestEngine

# 性能提升5-10倍
engine = VectorizedBacktestEngine(config, strategy, fetcher)
result = engine.run()
```

### 并行回测（多参数）

```python
from analysis.backtest_vectorized import ParallelBacktestRunner

# 并行运行多个参数组合
runner = ParallelBacktestRunner(max_workers=4)
results = runner.run_backtests(param_list)
```

## 🎯 参数优化

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
    step_days=63        # 滚动步长
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

## 📖 演示指南

本文档提供回测系统的功能演示。

## 🎯 快速演示

### 运行一键演示

```bash
python runners/backtest_demo.py
```

演示内容：
1. 交易成本模型演示
2. 蒙特卡洛模拟演示
3. 压力测试演示
4. 绩效评估演示
5. 综合应用演示

## 📊 演示功能

### 1. 交易成本模型

演示不同交易规模的成本计算：
- 小额交易（1万元）
- 中额交易（50万元）
- 大额交易（500万元）

### 2. 蒙特卡洛模拟

模拟200次未来表现：
- VaR（风险价值）
- CVaR（条件风险价值）
- 收益率分布
- 盈利概率

### 3. 压力测试

模拟极端市场情况：
- 市场暴跌场景
- 波动率飙升场景
- 相关性危机场景

### 4. 绩效评估

计算20+个绩效指标：
- 基础指标（收益率、回撤）
- 风险调整收益（夏普、Sortino、Calmar）
- 基准比较（Alpha、Beta、信息比率）

## 📖 系统优化说明

本文档说明系统性能优化方案。

## 🚀 性能优化

### 1. 数据类型优化

将 float64 改为 float32，节省 40-50% 内存：

```python
# 优化前
close = stock_data["close"].astype(float)

# 优化后
close = stock_data["close"].astype(np.float32)
```

### 2. 向量化计算

使用 NumPy/pandas 向量化操作，避免循环：

```python
# 优化前
for code in unique_codes:
    result = calculate_indicator(code)

# 优化后
results = df.groupby('code').apply(calculate_indicator)
```

### 3. 缓存优化

两级缓存（内存+磁盘）：

```python
from core.cache_manager_optimized import CacheManager

cache = CacheManager(
    cache_dir="./cache",
    memory_cache_size=100,
    enable_compression=True
)
```

## 📈 性能提升

| 优化项 | 性能提升 | 内存优化 |
|-------|---------|---------|
| 数据类型优化 | 10-15% | 40-50% |
| 向量化计算 | 3-5倍 | 20% |
| 缓存优化 | 30-40% | - |
| 并行回测 | 4倍（4核） | - |

## 📖 趋势雷达优化

本文档说明趋势雷达选股系统的优化改进。

## 🎯 优化内容

### 1. 多因子评分

从4个因子扩展到8个因子：
- 突破信号
- 趋势结构
- 量能确认
- 趋势强度
- 动量强劲（新增）
- MACD金叉（新增）
- 量能持续性（新增）
- 价格位置（新增）

### 2. 动态参数调整

根据市场环境动态调整选股标准：
- **牛市**: 降低阈值，更积极
- **震荡市**: 标准阈值，精选标的
- **熊市**: 提高阈值，更保守

### 3. 增强风险管理

三层止损体系：
- 硬止损（-10%）
- ATR止损（动态）
- 布林带止损（趋势）

## 📖 测试指南

本文档说明如何运行项目测试。

## 🧪 运行测试

### 运行所有测试

```bash
python run_tests.py
```

### 运行特定测试

```bash
# 指标测试
python run_tests.py --type indicators

# 配置测试
python run_tests.py --type config

# 日志测试
python run_tests.py --type logger
```

### 生成覆盖率报告

```bash
python run_tests.py --coverage
```

## 📊 测试覆盖

- 技术指标测试: 11个测试类
- 配置管理测试: 11个测试类
- 日志系统测试: 1个测试类
- 数据验证测试: 10个测试类
- 可视化测试: 8个测试类
- 扩展指标测试: 7个测试类
- 缓存并发测试: 6个测试类

总计: 200+ 测试用例

---

## 📖 附录

### 命令行参数速查

| 参数 | 说明 | 默认值 |
|-----|------|--------|
| `--top-n` | 返回TopN股票 | 20 |
| `--index-code` | 指数代码 | 000300.SH |
| `--no-report` | 不保存报告 | False |
| `--token` | API Token | 环境变量 |
| `--quiet` | 静默模式 | False |

### 常用指数代码

- 000300.SH: 沪深300
- 000905.SH: 中证500
- 399006.SZ: 创业板指
- 000016.SH: 上证50

### 项目结构

```
Stock/
├── config/          # 配置模块
├── core/            # 核心功能
├── indicators/      # 技术指标
├── strategy/        # 策略模块
├── analysis/        # 分析模块
├── visualization/   # 可视化
├── runners/         # 运行脚本
├── tests/           # 测试模块
├── cache/           # 缓存目录
├── logs/            # 日志目录
└── reports/         # 报告目录
```

### 技术支持

如有问题，请查看：
- GitHub Issues
- 项目文档
- 测试用例

---

**最后更新**: 2026-01-21
