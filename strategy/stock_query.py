"""
趋势雷达选股系统 - 股票查询模块
提供股票信息查询功能
"""
import pandas as pd
from core.data_fetcher import DataFetcher
from core.utils import RateLimiter


def query_stock_industry(stock_codes: str | list[str]) -> None:
    """
    查询股票所属行业板块信息

    参数:
        stock_codes: 股票代码，可以是单个代码或代码列表
                  例如: "000001.SZ" 或 ["000001.SZ", "600000.SH"]
    """
    # 初始化数据获取器
    token = "706b1dbca05800fea1d77c3a727f6ad5e0b3a1d0687f8a4e3266fe9c"
    rate_limiter = RateLimiter(max_calls_per_minute=200)
    fetcher = DataFetcher(token, rate_limiter)

    # 获取股票基础信息
    basic_df = fetcher.get_stock_basic()

    if basic_df.empty:
        print("获取股票基础信息失败")
        return

    # 查询股票信息
    if isinstance(stock_codes, str):
        stock_codes = [stock_codes]

    results = []
    for code in stock_codes:
        match = basic_df[basic_df["ts_code"] == code]
        if not match.empty:
            row = match.iloc[0]
            industry = row.get("industry", "未知")

            # 行业分类
            industry_map = {
                "银行": "金融", "证券": "金融", "保险": "金融", "信托": "金融",
                "计算机": "科技", "通信": "科技", "半导体": "科技", "电子": "科技",
                "医药生物": "医药", "医疗器械": "医药", "医疗服务": "医药",
                "食品饮料": "消费", "纺织服装": "消费", "家用电器": "消费",
                "石油石化": "能源", "煤炭": "能源", "电力": "能源",
                "有色金属": "材料", "钢铁": "材料", "基础化工": "材料",
                "机械设备": "制造业", "电气设备": "制造业", "汽车": "制造业",
                "建筑装饰": "基建", "建筑材料": "基建", "房地产": "基建",
                "农林牧渔": "农业"
            }

            # 查找匹配的行业大类
            sector = "其他"
            for key, value in industry_map.items():
                if key in industry:
                    sector = value
                    break

            results.append({
                "代码": code,
                "名称": row.get("name", ""),
                "行业": industry,
                "板块": sector
            })

    # 输出结果
    if results:
        print("\n股票行业信息:")
        print("-" * 60)
        print(f"{'代码':<12} {'名称':<12} {'行业':<20} {'板块':<10}")
        print("-" * 60)
        for r in results:
            print(f"{r['代码']:<12} {r['名称']:<12} {r['行业']:<20} {r['板块']:<10}")
        print("-" * 60)
    else:
        print(f"未找到股票: {stock_codes}")


def query_stock_detail(stock_code: str) -> None:
    """
    查询单只股票的详细信息

    参数:
        stock_code: 股票代码，如 "000001.SZ"
    """
    token = "706b1dbca05800fea1d77c3a727f6ad5e0b3a1d0687f8a4e3266fe9c"
    rate_limiter = RateLimiter(max_calls_per_minute=200)
    fetcher = DataFetcher(token, rate_limiter)

    # 获取基础信息
    basic_df = fetcher.get_stock_basic()
    match = basic_df[basic_df["ts_code"] == stock_code]

    if match.empty:
        print(f"未找到股票: {stock_code}")
        return

    row = match.iloc[0]
    print(f"\n股票详细信息:")
    print("-" * 50)
    print(f"代码: {row['ts_code']}")
    print(f"名称: {row['name']}")
    print(f"行业: {row.get('industry', '未知')}")
    print(f"上市日期: {row['list_date']}")
    print("-" * 50)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("使用方法:")
        print("  python stock_query.py <股票代码>")
        print("  例如: python stock_query.py 000001.SZ")
    else:
        stock_code = sys.argv[1]
        query_stock_industry(stock_code)
        query_stock_detail(stock_code)
