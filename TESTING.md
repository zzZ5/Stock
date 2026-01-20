# 趋势雷达选股系统 - 测试指南

## 安装测试依赖

在运行测试之前，需要安装测试相关的依赖包：

```bash
# 安装所有依赖（包括测试依赖）
pip install -r requirements.txt

# 或者只安装测试依赖
pip install pytest pytest-cov
```

## 运行测试

### 运行所有测试

```bash
# 使用 pytest
python -m pytest tests/ -v

# 或者使用便捷脚本
python run_tests.py
```

### 运行特定测试

```bash
# 运行特定测试文件
python -m pytest tests/test_indicators.py -v

# 运行特定测试类
python -m pytest tests/test_indicators.py::TestSMA -v

# 运行特定测试方法
python -m pytest tests/test_indicators.py::TestSMA::test_sma_basic -v

# 使用便捷脚本
python run_tests.py --type indicators
python run_tests.py --type config
python run_tests.py --type logger
python run_tests.py --type validators
python run_tests.py --type visualization
python run_tests.py --type cache_concurrent
```

### 运行测试并生成覆盖率报告

```bash
# 使用 pytest
python -m pytest --cov=. --cov-report=html --cov-report=term-missing -v tests/

# 或者使用便捷脚本
python run_tests.py --coverage
```

覆盖率报告会生成在 `htmlcov/` 目录中，在浏览器中打开 `htmlcov/index.html` 查看详细报告。

## 测试文件说明

### tests/test_indicators.py
测试所有技术指标计算函数的正确性：
- **TestSMA**: 测试简单移动平均线
- **TestEMA**: 测试指数移动平均线
- **TestATR**: 测试平均真实波幅
- **TestRSI**: 测试相对强弱指标
- **TestMACD**: 测试MACD指标
- **TestKDJ**: 测试KDJ指标
- **TestWilliamsR**: 测试威廉指标
- **TestPricePosition**: 测试价格位置指标
- **TestADX**: 测试ADX指标
- **TestBollingerBands**: 测试布林带
- **TestOBV**: 测试能量潮指标

### tests/test_config.py
测试配置参数的合法性和有效性：
- **TestConfigParameters**: 测试所有配置参数的合理性

### tests/test_logger.py
测试日志系统的功能：
- **TestLoggerSetup**: 测试日志系统初始化
- **TestGetLogger**: 测试获取logger实例
- **TestLoggerOutput**: 测试日志输出
- **TestLoggerLevels**: 测试日志级别

### tests/test_validators.py
测试数据验证器的功能：
- **TestDataFrameValidator**: 测试DataFrame验证
- **TestPriceValidator**: 测试价格数据验证
- **TestDateValidator**: 测试日期验证
- **TestParameterValidator**: 测试参数验证
- **TestConfigValidator**: 测试配置验证
- **TestSafeCalculator**: 测试安全计算

### tests/test_visualization.py
测试可视化模块的功能：
- **TestPlotter**: 测试绘图器基础功能
- **TestStockCandlestick**: 测试K线图绘制
- **TestStockIndicators**: 测试技术指标图绘制
- **TestBacktestResults**: 测试回测结果可视化
- **TestDrawdownChart**: 测试回撤图绘制
- **TestMonthlyReturns**: 测试月度收益图绘制
- **TestParameterHeatmap**: 测试参数热力图
- **TestParameterSensitivity**: 测试参数敏感性分析

### tests/test_cache_concurrent.py
测试缓存和并发模块：
- **TestLRUCache**: 测试LRU缓存
- **TestCacheManager**: 测试优化的缓存管理器
- **TestRateLimiter**: 测试速率限制器
- **TestConcurrentRateLimiter**: 测试并发限流器
- **TestProgressTracker**: 测试进度追踪器
- **TestThreadPool**: 测试线程池
- **TestBatchProcessor**: 测试批量处理器
- **TestRetryDecorator**: 测试重试装饰器

### tests/test_config_loader.py
测试配置文件加载器功能：
- **TestConfigValidator**: 测试配置验证器（策略参数、缓存参数、日志参数等）
- **TestConfigLoader**: 测试配置加载器（YAML/JSON加载、配置合并、模板生成）
- **TestConvenienceFunctions**: 测试便捷函数
- **TestDefaultConfig**: 测试默认配置的完整性和类型
- **TestProjectConfigFiles**: 测试项目配置文件的存在和加载

### tests/test_config_manager.py
测试全局配置管理器功能：
- **TestConfig**: 测试Config单例模式（加载、重载、获取、设置）
- **TestConfigGetters**: 测试配置项getter函数（策略、缓存、报告、回测、指数、日志参数）
- **TestConfigCustomization**: 测试配置自定义功能

## pytest 配置

`pytest.ini` 文件包含 pytest 的配置：

```ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    -v
    --strict-markers
    --tb=short
```

## 编写新测试

### 测试文件命名规范

- 测试文件：`test_<module_name>.py`
- 测试类：`Test<ClassName>`
- 测试函数：`test_<function_name>`

### 测试示例

```python
import pytest
import pandas as pd

def test_example():
    """测试示例"""
    # 准备测试数据
    data = pd.Series([1, 2, 3, 4, 5])
    
    # 调用被测试的函数
    result = some_function(data)
    
    # 断言结果
    assert result == 15
    assert isinstance(result, int)
```

### 使用 Fixture

```python
@pytest.fixture
def sample_data():
    """创建测试数据"""
    return pd.DataFrame({
        'open': [10, 11, 12],
        'close': [10.5, 11.5, 12.5]
    })

def test_with_fixture(sample_data):
    """使用fixture的测试"""
    result = some_function(sample_data)
    assert result is not None
```

## 持续集成

### GitHub Actions 示例

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
      - name: Run tests
        run: |
          python -m pytest tests/ -v --cov=. --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

## 常见问题

### Q: 如何跳过某个测试？

```bash
# 跳过特定测试
python -m pytest tests/test_indicators.py::TestSMA::test_sma_basic -v -k "not test_sma_basic"

# 使用标记跳过
@ pytest.mark.skip("暂时跳过")
def test_example():
    pass
```

### Q: 如何只运行失败的测试？

```bash
python -m pytest tests/ --lf
```

### Q: 如何在失败时停止测试？

```bash
python -m pytest tests/ -x
```

### Q: 如何显示详细的输出？

```bash
python -m pytest tests/ -vv -s
```

## 测试覆盖率目标

- **整体覆盖率**: 目标 > 80%
- **核心模块**: 目标 > 90%
  - `indicators/`: 目标 > 90%
  - `core/`: 目标 > 85%
  - `strategy/`: 目标 > 85%

## 下一步计划

- [ ] 添加回测引擎的测试用例
- [ ] 添加参数优化器的测试用例
- [ ] 添加报告生成的测试用例
- [ ] 添加集成测试
- [ ] 提高测试覆盖率到 > 80%
