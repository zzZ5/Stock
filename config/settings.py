"""
趋势雷达选股系统 - 配置参数模块
包含所有可配置的策略参数、缓存参数等
"""

class Settings:
    """配置类，封装所有可配置参数"""
    def __init__(self):
        # =========================
        # 选股策略参数
        # =========================
        self.TOP_N = 20                      # 返回TopN候选股票

        # 多周期突破模式
        self.MULTI_TIMEFRAME_MODE = True     # 是否启用多周期突破（日周月）

        # 趋势信号参数
        self.BREAKOUT_N = 60                 # 60日突破（最高收盘价）
        self.WEEKLY_BREAKOUT_N = 12          # 12周突破（周线）
        self.MONTHLY_BREAKOUT_N = 6          # 6月突破（月线）
        self.MA_FAST = 20                    # 快速均线
        self.MA_SLOW = 60                    # 慢速均线
        self.VOL_LOOKBACK = 20               # 量能回看天数
        self.VOL_CONFIRM_MULT = 1.2          # 放量确认阈值（当前成交额 / 20日均额）
        self.RSI_MAX = 80                    # 过热过滤阈值

        # 风险参数
        self.MAX_LOSS_PCT = 0.10            # -10% 硬止损
        self.ATR_N = 14                      # ATR计算周期
        self.ATR_MULT = 2.5                  # 波动止损倍数（入场价 - ATR_MULT*ATR）

        # 过滤参数
        self.MIN_LIST_DAYS = 120             # 剔除上市不足 120 个交易日
        self.MIN_PRICE = 2.0                 # 剔除低价票
        self.MIN_AVG_AMOUNT_20D = 30000      # 近20日平均成交额阈值（TuShare amount 通常为"千元"）
        self.EXCLUDE_ONE_WORD_LIMITUP = True  # 排除一字涨停板

        # =========================
        # 缓存与限流参数
        # =========================
        self.CACHE_DIR = "./tushare_cache"         # 缓存目录
        self.SLEEP_PER_CALL = 0.25                 # 每次API调用后的延迟时间
        self.MAX_CALLS_PER_MINUTE = 200            # API每分钟最大调用次数

        self.STOCK_BASIC_TTL_DAYS = 7        # stock_basic 缓存有效期（天）
        self.TRADE_CAL_TTL_DAYS = 30         # trade_cal 缓存有效期（天）

        # =========================
        # 报告输出参数
        # =========================
        self.SAVE_REPORT = True              # 是否保存报告到本地
        self.REPORT_DIR = "./reports"         # 报告保存目录

        # =========================
        # 回测参数
        # =========================
        self.DEFAULT_HOLDING_DAYS = 10       # 默认持仓天数（用于回测）

        # =========================
        # 指数参数
        # =========================
        self.INDEX_CODE = "000300.SH"        # 沪深300指数代码

        # =========================
        # 日志配置
        # =========================
        self.LOG_LEVEL = "INFO"              # 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        self.LOG_DIR = "./logs"              # 日志文件目录
        self.LOG_CONSOLE_OUTPUT = True       # 是否输出到控制台
        self.LOG_FILE_OUTPUT = True          # 是否输出到文件
        self.LOG_MAX_FILE_SIZE = 10 * 1024 * 1024  # 单个日志文件最大大小（字节），默认10MB
        self.LOG_BACKUP_COUNT = 5            # 保留的日志文件备份数量


# 创建全局 settings 对象
settings = Settings()
