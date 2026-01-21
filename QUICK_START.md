# 趋势雷达选股系统 - 快速开始

## 🚀 5分钟快速上手

### 方式1：交互式菜单（最简单）⭐

```bash
python runners/interactive_menu.py
```

然后按照菜单提示操作即可！

### 方式2：命令行（最灵活）

```bash
# 默认配置
python runners/trend_radar_user_friendly.py

# 自定义参数
python runners/trend_radar_user_friendly.py --top-n 10 --index-code 000905.SH
```

## 📋 常用命令

| 场景 | 命令 |
|-----|------|
| 新手入门 | `python runners/interactive_menu.py` |
| 日常使用 | `python runners/trend_radar_user_friendly.py` |
| 快速测试 | `python runners/trend_radar_user_friendly.py --top-n 5` |
| 定时任务 | `python runners/trend_radar_user_friendly.py --quiet` |
| 回测演示 | `python runners/backtest_demo.py` |

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

- **[DOCUMENTATION.md](DOCUMENTATION.md)** - 完整文档索引
- **[README.md](README.md)** - 项目详细说明

## ⚠️ 注意事项

1. 需要配置 TuShare API Token
2. 免费版数据可能有1-2天延迟
3. 选股结果仅供参考，不构成投资建议

## 🆘 遇到问题？

1. 查看 [DOCUMENTATION.md](DOCUMENTATION.md) 中的常见问题
2. 检查 Token 是否正确配置
3. 查看日志文件 `logs/` 目录

---

**开始你的量化之旅！** 🎉
