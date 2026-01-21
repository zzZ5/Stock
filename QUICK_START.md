# 趋势雷达选股系统 - 快速开始

## 🚀 5分钟快速上手

### 方式1：交互式菜单（最简单）⭐⭐⭐

```bash
python runners/interactive_menu.py
```

然后按照菜单提示操作即可！

### 方式2：命令行（最灵活）⭐⭐

```bash
# 默认配置
python runners/trend_radar_main.py

# 自定义参数
python runners/trend_radar_main.py --top-n 10 --index-code 000905.SH
```

## 📋 常用命令

| 场景 | 命令 |
|-----|------|
| 新手入门 | `python runners/interactive_menu.py` |
| 日常使用 | `python runners/trend_radar_main.py` |
| 快速测试 | `python runners/trend_radar_main.py --top-n 5` |
| 不同指数 | `python runners/trend_radar_main.py --index-code 000905.SH` |
| 历史回测 | `python runners/backtest_runner.py` |
| 参数优化 | `python runners/optimizer_runner.py` |

## 🎯 命令行参数

| 参数 | 说明 | 默认值 |
|-----|------|--------|
| `--top-n` | 返回Top N股票 | 20 |
| `--index-code` | 指数代码 | 000300.SH |
| `--holding-days` | 持有天数 | 10 |
| `--save-report` | 保存报告 | False |
| `--verbose` | 详细输出 | False |
| `--token` | API Token | 环境变量 |

## ⚙️ 配置Token

### 方式1：环境变量（推荐）
```bash
# Linux/Mac
export TUSHARE_TOKEN="your_token_here"

# Windows
set TUSHARE_TOKEN=your_token_here
```

### 方式2：命令行参数
```bash
python runners/trend_radar_main.py --token your_token_here
```

### 方式3：配置文件
编辑 `config.yaml` 或 `config.json`：
```yaml
tushare:
  token: "your_token_here"
```

### 获取Token
1. 访问 [Tushare官网](https://tushare.pro/)
2. 注册并登录
3. 在个人中心获取API Token

## 🎯 首次使用

1. **运行交互式菜单**
   ```bash
   python runners/interactive_menu.py
   ```

2. **选择选项 1** - 运行选股系统（默认配置）

3. **查看结果**
   - 控制台会显示选中的股票
   - 报告保存在 `reports/` 目录

## 📖 更多文档

- **[README.md](README.md)** - 项目详细说明
- **[DOCUMENTATION.md](DOCUMENTATION.md)** - 完整文档索引
- **[PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)** - 项目结构说明

## ⚠️ 注意事项

1. 需要配置 TuShare API Token
2. 免费版数据可能有1-2天延迟
3. 选股结果仅供参考，不构成投资建议
4. API有调用限制，建议间隔运行

## 🆘 常见问题

### Q: 运行时提示"未获取到交易日历"？

**A:** 检查以下几点：
- Token是否正确
- 网络连接是否正常
- Tushare API是否可用

### Q: 如何修改选股参数？

**A:** 编辑 `config/settings.py` 文件，或使用命令行参数：
```bash
python runners/trend_radar_main.py --top-n 30 --holding-days 15
```

### Q: 选股报告保存在哪里？

**A:** 默认保存在 `reports/` 目录，文件名格式：`trend_radar_YYYYMMDD.md`

### Q: 为什么没有选中股票？

**A:** 可能原因：
- 市场环境较差（熊市）
- 选股标准过高
- 数据不足或异常
- 尝试调整 `--top-n` 参数查看更多结果

### Q: 如何查看沪深500最强股票？

**A:** 使用中证500指数代码：
```bash
python runners/trend_radar_main.py --index-code 000905.SH
```

---

**开始你的量化之旅！** 🎉
