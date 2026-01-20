"""
趋势雷达选股系统 - 测试运行脚本
运行所有单元测试
"""
import sys
import os
import subprocess

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_tests(test_type="all"):
    """
    运行测试

    参数:
        test_type: 测试类型 (all, unit, indicators, config, logger)
    """
    base_cmd = ["pytest", "-v", "--tb=short"]
    
    if test_type == "all":
        cmd = base_cmd + ["tests/"]
    elif test_type == "indicators":
        cmd = base_cmd + ["tests/test_indicators.py"]
    elif test_type == "config":
        cmd = base_cmd + ["tests/test_config.py"]
    elif test_type == "logger":
        cmd = base_cmd + ["tests/test_logger.py"]
    else:
        print(f"未知的测试类型: {test_type}")
        return False
    
    print(f"运行测试: {test_type}")
    print("=" * 70)
    
    result = subprocess.run(cmd, cwd=os.path.dirname(__file__))
    
    return result.returncode == 0


def run_with_coverage():
    """运行测试并生成覆盖率报告"""
    cmd = [
        "pytest",
        "--cov=.",
        "--cov-report=html",
        "--cov-report=term-missing",
        "-v",
        "tests/"
    ]
    
    print("运行测试并生成覆盖率报告...")
    print("=" * 70)
    
    result = subprocess.run(cmd, cwd=os.path.dirname(__file__))
    
    if result.returncode == 0:
        print("\n覆盖率报告已生成在 htmlcov/ 目录")
        print("在浏览器中打开 htmlcov/index.html 查看详细报告")
    
    return result.returncode == 0


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="运行趋势雷达选股系统的测试")
    parser.add_argument(
        "--type",
        choices=["all", "indicators", "config", "logger"],
        default="all",
        help="测试类型"
    )
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="生成代码覆盖率报告"
    )
    
    args = parser.parse_args()
    
    if args.coverage:
        success = run_with_coverage()
    else:
        success = run_tests(args.type)
    
    if success:
        print("\n" + "=" * 70)
        print("测试通过！")
        print("=" * 70)
        sys.exit(0)
    else:
        print("\n" + "=" * 70)
        print("测试失败！")
        print("=" * 70)
        sys.exit(1)


if __name__ == "__main__":
    main()
