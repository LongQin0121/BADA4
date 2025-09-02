print("Hello, Python in VS Code!")

packages = {
    "numpy": "np",
    "pandas": "pd",
    "sklearn": "sk",
    "matplotlib.pyplot": "plt",
    "seaborn": "sns",
    "torch": "torch",
    "tensorflow": "tf"
}

print("🔍 正在检测包是否成功安装...\n")

for package, alias in packages.items():
    try:
        exec(f"import {package} as {alias}")
        print(f"✅ 成功导入：{package}")
    except ImportError:
        print(f"❌ 未安装：{package}")

print("\n✅ 检查完毕！")


