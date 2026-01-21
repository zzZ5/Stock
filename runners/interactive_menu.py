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
    print("  [1] 运行选股系统（默认配置）")
    print("  [2] 运行选股系统（自定义Top N）")
    print("  [3] 运行回测演示")
    print("  [4] 查看使用指南")
    print("  [5] 退出")
    print()


def run_stock_selection(top_n=None):
    """运行选股系统"""
    clear_screen()
    print_header()
    print("正在运行选股系统...")
    print()

    from runners.trend_radar_user_friendly import main

    # 修改sys.argv传入参数
    original_argv = sys.argv.copy()

    if top_n:
        sys.argv = ['trend_radar_user_friendly.py', f'--top-n={top_n}']
    else:
        sys.argv = ['trend_radar_user_friendly.py']

    try:
        main()
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

    from runners.backtest_demo import main

    original_argv = sys.argv.copy()
    sys.argv = ['backtest_demo.py']

    try:
        main()
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
    print("  • USER_GUIDE.md - 用户快速开始指南")
    print("  • BACKTEST_GUIDE.md - 回测系统详细指南")
    print("  • OPTIMIZATION_SUMMARY.md - 系统优化说明")
    print()
    print("🚀 命令行使用：")
    print("  python runners/trend_radar_user_friendly.py")
    print("  python runners/trend_radar_user_friendly.py --top-n 10")
    print("  python runners/backtest_demo.py")
    print()
    print("⚙️  配置文件：")
    print("  config/settings.py - 主要配置参数")
    print()
    print("💡 常用参数：")
    print("  --top-n N          设置返回Top N股票（默认20）")
    print("  --index-code CODE  设置指数代码（默认000300.SH）")
    print("  --no-report        不保存报告")
    print("  --quiet            静默模式")
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


def main_menu():
    """主菜单"""
    while True:
        clear_screen()
        print_header()
        print_menu()

        choice = input("请输入选项 [1-5]: ").strip()

        if choice == '1':
            run_stock_selection()
        elif choice == '2':
            top_n = get_top_n()
            run_stock_selection(top_n=top_n)
        elif choice == '3':
            run_backtest_demo()
        elif choice == '4':
            show_guide()
        elif choice == '5':
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
