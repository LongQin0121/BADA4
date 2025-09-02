#!/usr/bin/env python3
"""
Stockholm ATC Simulator - 依赖安装脚本
自动安装所需的Python包
"""

import subprocess
import sys
import importlib

def check_package(package_name):
    """检查包是否已安装"""
    try:
        importlib.import_module(package_name)
        return True
    except ImportError:
        return False

def install_package(package_name):
    """安装包"""
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package_name])
        return True
    except subprocess.CalledProcessError:
        return False

def main():
    """主函数"""
    print("🛩️ Stockholm ATC Simulator - 依赖检查和安装")
    print("=" * 50)
    
    # 必需的包
    required_packages = {
        'flask': 'Flask',
        'flask_socketio': 'Flask-SocketIO',
        'matplotlib': 'matplotlib',
        'numpy': 'numpy',
        'requests': 'requests'
    }
    
    # 可选的包
    optional_packages = {
        'eventlet': 'eventlet',  # 提升WebSocket性能
        'pandas': 'pandas',      # 数据分析
        'plotly': 'plotly'       # 高级可视化
    }
    
    print("检查必需的包...")
    missing_required = []
    
    for module_name, package_name in required_packages.items():
        if check_package(module_name):
            print(f"✅ {package_name} - 已安装")
        else:
            print(f"❌ {package_name} - 未安装")
            missing_required.append(package_name)
    
    if missing_required:
        print(f"\n正在安装缺失的必需包: {', '.join(missing_required)}")
        
        for package in missing_required:
            print(f"安装 {package}...")
            if install_package(package):
                print(f"✅ {package} 安装成功")
            else:
                print(f"❌ {package} 安装失败")
                return False
    
    print("\n检查可选的包...")
    missing_optional = []
    
    for module_name, package_name in optional_packages.items():
        if check_package(module_name):
            print(f"✅ {package_name} - 已安装")
        else:
            print(f"⚠️  {package_name} - 未安装 (可选)")
            missing_optional.append(package_name)
    
    if missing_optional:
        install_optional = input(f"\n是否安装可选包? {', '.join(missing_optional)} [y/N]: ")
        if install_optional.lower() in ['y', 'yes']:
            for package in missing_optional:
                print(f"安装 {package}...")
                if install_package(package):
                    print(f"✅ {package} 安装成功")
                else:
                    print(f"⚠️  {package} 安装失败 (可选包，可忽略)")
    
    print("\n" + "=" * 50)
    print("✅ 依赖检查完成!")
    print("\n🚀 可以开始使用模拟器:")
    print("   python run_simulator.py")
    print("\n📖 详细说明:")
    print("   cat README.md")
    
    return True

if __name__ == "__main__":
    main()