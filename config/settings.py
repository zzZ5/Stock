"""
趋势雷达选股系统 - 配置参数模块
包含所有可配置的策略参数、缓存参数等
"""

# =========================
# 选股策略参数
# =========================
TOP_N = 20                      # 返回TopN候选股票

# 趋势信号参数
BREAKOUT_N = 60                 # 60日突破（最高收盘价）
MA_FAST = 20                    # 快速均线
MA_SLOW = 60                    # 慢速均线
VOL_LOOKBACK = 20               # 量能回看天数
VOL_CONFIRM_MULT = 1.2          # 放量确认阈值（当前成交额 / 20日均额）
RSI_MAX = 80                    # 过热过滤阈值

# 风险参数
MAX_LOSS_PCT = 0.10            # -10% 硬止损
ATR_N = 14                      # ATR计算周期
ATR_MULT = 2.5                  # 波动止损倍数（入场价 - ATR_MULT*ATR）

# 过滤参数
MIN_LIST_DAYS = 120             # 剔除上市不足 120 个交易日
MIN_PRICE = 2.0                 # 剔除低价票
MIN_AVG_AMOUNT_20D = 30000      # 近20日平均成交额阈值（TuShare amount 通常为"千元"）
EXCLUDE_ONE_WORD_LIMITUP = True  # 排除一字涨停板

# =========================
# 缓存与限流参数
# =========================
CACHE_DIR = "./tushare_cache"         # 缓存目录
SLEEP_PER_CALL = 0.25                 # 每次API调用后的延迟时间
MAX_CALLS_PER_MINUTE = 200            # API每分钟最大调用次数

STOCK_BASIC_TTL_DAYS = 7        # stock_basic 缓存有效期（天）
TRADE_CAL_TTL_DAYS = 30         # trade_cal 缓存有效期（天）

# =========================
# 报告输出参数
# =========================
SAVE_REPORT = True              # 是否保存报告到本地
REPORT_DIR = "./reports"         # 报告保存目录

# =========================
# 回测参数
# =========================
DEFAULT_HOLDING_DAYS = 10       # 默认持仓天数（用于回测）

# =========================
# 指数参数
# =========================
INDEX_CODE = "000300.SH"        # 沪深300指数代码

# =========================
# 日志配置
# =========================
LOG_LEVEL = "INFO"              # 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
LOG_DIR = "./logs"              # 日志文件目录
LOG_CONSOLE_OUTPUT = True       # 是否输出到控制台
LOG_FILE_OUTPUT = True          # 是否输出到文件
LOG_MAX_FILE_SIZE = 10 * 1024 * 1024  # 单个日志文件最大大小（字节），默认10MB
LOG_BACKUP_COUNT = 5            # 保留的日志文件备份数量
