"""
趋势雷达选股系统 - Web界面启动器
提供简单的交互式菜单选择
"""
import os
import sys

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def clear_screen():
    """清屏"""
    os.system('cls' if os.name == 'nt' else 'clear')


def print_header():
    """打印标题"""
    print()
    print("╔" + "═"*68 + "╗")
    print("║" + " "*68 + "║")
    print("║" + "    趋势雷达选股系统".center(68) + "║")
    print("║" + "    Trend Radar Stock Selection".center(68) + "║")
    print("║" + " "*68 + "║")
    print("╚" + "═"*68 + "╝")
    print()


def print_menu():
    """打印菜单"""
    print("请选择操作：")
    print()
    print("  [1] 运行选股系统（默认配置-多周期）")
    print("  [2] 运行选股系统（自定义参数）")
    print("  [3] 运行回测演示")
    print("  [4] 运行参数优化")
    print("  [5] 查看使用指南")
    print("  [6] 退出")
    print()


def run_stock_selection(top_n=None, multi_tf=None):
    """运行选股系统"""
    clear_screen()
    print_header()
    print("正在运行选股系统...")
    print()

    from runners.trend_radar_main import main as trend_main

    # 修改sys.argv传入参数
    original_argv = sys.argv.copy()

    if top_n:
        if multi_tf is not None:
            if multi_tf:
                sys.argv = ['trend_radar_main.py', '--top-n', str(top_n), '--multi-tf']
            else:
                sys.argv = ['trend_radar_main.py', '--top-n', str(top_n), '--daily-only']
        else:
            sys.argv = ['trend_radar_main.py', '--top-n', str(top_n)]
    else:
        if multi_tf is not None:
            if multi_tf:
                sys.argv = ['trend_radar_main.py', '--multi-tf']
            else:
                sys.argv = ['trend_radar_main.py', '--daily-only']
        else:
            sys.argv = ['trend_radar_main.py']

    try:
        trend_main()
    except SystemExit:
        pass
    finally:
        sys.argv = original_argv


def run_backtest_demo():
    """运行回测演示"""
    clear_screen()
    print_header()
    print("正在运行回测演示...")
    print()

    from runners.backtest_demo import main as demo_main

    original_argv = sys.argv.copy()
    sys.argv = ['backtest_demo.py']

    try:
        demo_main()
    except SystemExit:
        pass
    finally:
        sys.argv = original_argv


def run_optimizer():
    """运行参数优化"""
    clear_screen()
    print_header()
    print("正在运行参数优化...")
    print()

    from runners.optimizer_runner import main as opt_main

    original_argv = sys.argv.copy()
    sys.argv = ['optimizer_runner.py']

    try:
        opt_main()
    except SystemExit:
        pass
    finally:
        sys.argv = original_argv


def show_guide():
    """显示使用指南"""
    clear_screen()
    print_header()
    print("使用指南")
    print()
    print("="*70)
    print()
    print("📚 完整文档：")
    print("  • QUICK_START.md - 5分钟快速上手指南")
    print("  • README.md - 项目详细说明")
    print("  • DOCUMENTATION.md - 完整文档索引")
    print("  • PROJECT_STRUCTURE.md - 项目结构说明")
    print()
    print("🚀 命令行使用：")
    print("  python runners/trend_radar_main.py")
    print("  python runners/trend_radar_main.py --top-n 10")
    print("  python runners/trend_radar_main.py --multi-tf")
    print("  python runners/trend_radar_main.py --daily-only")
    print("  python runners/trend_radar_main.py --index-code 000905.SH")
    print("  python runners/backtest_demo.py")
    print("  python runners/optimizer_runner.py")
    print()
    print("⚙️  配置文件：")
    print("  config/settings.py - 主要配置参数")
    print("  config.yaml - YAML格式配置（推荐使用）")
    print()
    print("💡 常用参数：")
    print("  --top-n N          设置返回Top N股票（默认20）")
    print("  --multi-tf         启用多周期模式（日+周+月）")
    print("  --daily-only       仅使用日线突破")
    print("  --index-code CODE  设置指数代码（默认000300.SH）")
    print("  --holding-days N   设置持有天数（默认10）")
    print("  --save-report      保存报告")
    print("  --verbose         详细输出")
    print()
    print("📊 多周期突破说明：")
    print("  日突破: 股价突破近N日高点")
    print("  周突破: 股价突破近M周高点")
    print("  月突破: 股价突破近K月高点")
    print("  共振突破: 多周期同时突破，信号更强")
    print()
    print("="*70)
    print()
    input("按回车键继续...")


def get_top_n():
    """获取自定义Top N"""
    while True:
        try:
            top_n = input("请输入Top N数量（5-50，默认20）：").strip()

            if not top_n:
                return 20

            top_n = int(top_n)

            if 5 <= top_n <= 50:
                return top_n
            else:
                print("请输入5-50之间的数字！")
        except ValueError:
            print("请输入有效的数字！")


def get_multi_timeframe():
    """获取多周期模式"""
    while True:
        choice = input("选择周期模式 [1-3]：").strip()
        if choice == '1':
            return True  # 多周期（日+周+月）
        elif choice == '2':
            return False  # 仅日线
        elif choice == '3':
            return None  # 使用默认配置
        else:
            print("无效选项，请输入1/2/3！")


def run_custom_selection():
    """运行自定义参数选股"""
    clear_screen()
    print_header()
    print("自定义参数设置")
    print()

    top_n = get_top_n()

    print()
    print("请选择突破周期模式：")
    print("  [1] 多周期模式（日+周+月突破）")
    print("  [2] 仅日线突破")
    print("  [3] 使用默认配置")
    multi_tf = get_multi_timeframe()

    run_stock_selection(top_n=top_n, multi_tf=multi_tf)


def main_menu():
    """主菜单"""
    while True:
        clear_screen()
        print_header()
        print_menu()

        choice = input("请输入选项 [1-6]: ").strip()

        if choice == '1':
            run_stock_selection()  # 使用默认多周期模式
        elif choice == '2':
            run_custom_selection()
        elif choice == '3':
            run_backtest_demo()
        elif choice == '4':
            run_optimizer()
        elif choice == '5':
            show_guide()
        elif choice == '6':
            print()
            print("感谢使用趋势雷达选股系统！")
            print()
            break
        else:
            print()
            print("无效选项，请重新选择！")
            input("按回车键继续...")


if __name__ == "__main__":
    try:
        main_menu()
    except KeyboardInterrupt:
        print("\n\n用户中断")
        sys.exit(0)
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
