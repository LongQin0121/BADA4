# pyBADA - Base of Aircraft Data for Python

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://python.org)
[![BADA Version](https://img.shields.io/badge/BADA-4.2-green.svg)](https://eurocontrol.int)
[![License](https://img.shields.io/badge/license-EUPL-orange.svg)](https://joinup.ec.europa.eu/collection/eupl)

**完整的EUROCONTROL BADA (Base of Aircraft Data) Python实现，用于飞机性能建模、轨迹预测和优化。**

---

## 🚀 快速开始

### 安装要求

```bash
# Python环境
python >= 3.8
pandas >= 1.0.0
numpy >= 1.18.0
xml.etree.ElementTree (标准库)

# 安装pyBADA
pip install pyBADA
```

### 数据文件配置

⚠️ **重要**: pyBADA需要EUROCONTROL的BADA数据文件才能正常工作。

```bash
# 1. 从EUROCONTROL获取BADA数据文件
# BADA4.2: https://amelia.upc.edu:31416/sharing/5qXZ46B5H
# BADA3.15: https://amelia.upc.edu:31416/sharing/q8f50KNyP

# 2. 解压并安装数据文件
7z x BADA_4.2_L06514UPC.7z -o/tmp/bada_main
unzip BADA_4.2_Release_files.zip -d /tmp/bada_release

# 3. 复制到pyBADA目录
cp -r /tmp/bada_main/* /path/to/site-packages/pyBADA/aircraft/BADA4/
```

## 📖 基本使用

### 导入模块

```python
from pyBADA.bada4 import BADA4, Airplane, Parser
from pyBADA.aircraft import Bada
import pandas as pd
```

### 核心使用模式

#### 方法1: Airplane + BADA4 对象 ⭐⭐⭐⭐⭐

```python
# 创建飞机对象
airplane = Airplane()                    # 无参数构造
airplane.aircraft_type = 'B738W26'       # 设置飞机代码
airplane.code = 'B738W26'

# 创建BADA4计算对象
bada4 = BADA4(airplane)                  # 传入airplane对象

# 可用的计算方法 (28个)
methods = ['CD', 'CL', 'CF', 'CT', 'D', 'L', 'GS', 'CP', 'CLmax', ...]

# 示例计算
drag_coeff = bada4.CD(HLid=0, LG=0, CL=0.5, M=0.78)
lift_coeff = bada4.CL(delta=1.0, mass=70000, M=0.78)
fuel_flow = bada4.CF(delta=1.0, theta=1.0, DeltaTemp=0)
```

#### 方法2: 直接XML解析 ⭐⭐⭐⭐⭐

```python
import xml.etree.ElementTree as ET

# 解析特定飞机的XML文件
aircraft_code = 'B738W26'
xml_path = f"/path/to/pyBADA/aircraft/BADA4/Models/{aircraft_code}/{aircraft_code}.xml"

tree = ET.parse(xml_path)
root = tree.getroot()

# 提取参数
parameters = {}
for elem in root.iter():
    if elem.text and elem.text.strip():
        try:
            parameters[elem.tag.split('}')[-1]] = float(elem.text.strip())
        except ValueError:
            parameters[elem.tag.split('}')[-1]] = elem.text.strip()
```

#### 方法3: Parser类解析

```python
# 创建解析器
parser = Parser()

# 可用方法
parser.parseXML(xml_path, aircraft_name)      # 解析单个XML
parser.parseAll(directory, aircraft_name)     # 解析目录
parser.parseGPF(gpf_path, aircraft_name)      # 解析全局性能文件
```

## ✈️ 支持的飞机型号

### Boeing 系列 (共31个型号)
```python
boeing_aircraft = [
    # 737 系列
    'B73215', 'B73320', 'B73423', 'B73518', 'B73622',
    'B737W24', 'B738W26', 'B739ERW26',
    
    # 747 系列  
    'B742RR', 'B743PW', 'B744ERGE', 'B744GE', 'B748F',
    
    # 757/767 系列
    'B752WRR40', 'B753RR', 'B762ERPW56', 'B762GE50', 
    'B763ERGE61', 'B763PW60', 'B764ER',
    
    # 777 系列
    'B772LR', 'B772RR92', 'B773ERGE115B', 'B773RR92',
    
    # 787 系列
    'B788GE67', 'B788GE70', 'B788RR53', 'B788RR64', 
    'B788RR67', 'B788RR70', 'B789GE75', 'B789RR64', 'B789RR74'
]
```

### Airbus 系列 (共32个型号)
```python
airbus_aircraft = [
    # A300/A310 系列
    'A300B4-203', 'A300B4-601', 'A300B4-608ST', 'A300B4-622',
    'A310-204', 'A310-222', 'A310-308', 'A310-322', 'A310-324',
    
    # A320 系列
    'A318-112', 'A319-114', 'A319-131', 
    'A320-212', 'A320-214', 'A320-231', 'A320-232',
    'A321-111', 'A321-131',
    
    # A330 系列
    'A330-203', 'A330-223', 'A330-243', 'A330-301', 'A330-321', 'A330-341',
    
    # A340 系列
    'A340-213', 'A340-313', 'A340-541', 'A340-642',
    
    # A350/A380 系列
    'A350-941', 'A380-841', 'A380-861'
]
```

### 其他制造商 (共41个型号)
```python
other_aircraft = [
    # ATR系列
    'ATR42-300', 'ATR42-320', 'ATR42-400', 'ATR42-500',
    'ATR72-200', 'ATR72-210', 'ATR72-500', 'ATR72-600',
    
    # Embraer系列  
    'EMB-135ER', 'EMB-145ER', 'EMB-170AR', 'EMB-175LR', 
    'EMB-190STD', 'EMB-195AR', 'EMB-500', 'EMB-505',
    
    # Fokker系列
    'F70-620', 'F100-620', 'F100-650',
    
    # McDonnell Douglas系列
    'MD808120', 'MD808221', 'MD808321', 'MD808720', 'MD808821'
]
```

## 📊 参数说明

### Boeing 737-800 (B738W26) 参数示例

```python
# 基本信息
aircraft_info = {
    'icao_code': 'B738',
    'name': 'Boeing 737-800',
    'engine': 'CFM56-7B26/27',
    'wtc': 'M'  # Medium
}

# 几何参数
geometry = {
    'length': 39.47,        # 机身长度 (m)
    'wingspan': 34.32,      # 翼展 (m) [估算]
    'wing_area': 124.58,    # 翼面积 (m²)
    'height': 12.5          # 机身高度 (m) [估算]
}

# 重量参数
mass = {
    'mtow': 79016,          # 最大起飞重量 (kg)
    'mlw': 66361,           # 最大着陆重量 (kg)
    'mzfw': 62732           # 最大零燃油重量 (kg)
}

# 性能参数
performance = {
    'v_fe': 162,            # 襟翼极限速度 (kt)
    'mach_max': 0.82,       # 最大马赫数
    'cas_descent': 290,     # 下降指示空速 (kt)
    'mach_descent': 0.78    # 下降马赫数
}

# 跑道性能
runway = {
    'tol': 2200,            # 起飞距离 (m)
    'ldl': 1500             # 着陆距离 (m)
}
```

## 🧮 BADA4计算方法

### 空气动力学计算

```python
# 阻力系数计算
CD = bada4.CD(HLid, LG, CL, M)
# 参数:
# HLid: 高升力装置ID (0=收起, 1-7=不同襟翼位置)
# LG: 起落架状态 (0=收起, 1=放下)  
# CL: 升力系数
# M: 马赫数

# 升力系数计算
CL = bada4.CL(delta, mass, M)
# 参数:
# delta: 大气密度比 (ρ/ρ₀)
# mass: 飞机质量 (kg)
# M: 马赫数

# 最大升力系数
CLmax = bada4.CLmax(HLid, LG, M)
```

### 推力和燃油计算

```python
# 燃油流量系数
CF = bada4.CF(delta, theta, DeltaTemp)
# 参数:
# delta: 大气密度比
# theta: 大气温度比 (T/T₀)
# DeltaTemp: 温度偏差 (K)

# 推力系数
CT = bada4.CT(...)  # 需要更多参数

# 慢车推力燃油流量
CF_idle = bada4.CF_idle(delta, theta)
```

### 飞行包线计算

```python
# 地速计算
GS = bada4.GS(TAS, wind_speed, wind_direction, track)

# 阻力计算
D = bada4.D(CD, S, rho, TAS)

# 升力计算  
L = bada4.L(CL, S, rho, TAS)
```

## 🔧 实用示例

### 完整的飞机参数获取器

```python
class AircraftDataExtractor:
    def __init__(self, aircraft_code):
        self.aircraft_code = aircraft_code
        self.airplane = Airplane()
        self.airplane.aircraft_type = aircraft_code
        self.bada4 = BADA4(self.airplane)
        
    def get_all_parameters(self):
        # XML参数提取
        xml_params = self.extract_xml_params()
        
        # BADA4对象信息
        bada4_info = {
            'methods_count': len([m for m in dir(self.bada4) if not m.startswith('_')]),
            'available_methods': [m for m in dir(self.bada4) if not m.startswith('_')][:10]
        }
        
        return {
            'aircraft_code': self.aircraft_code,
            'xml_parameters': xml_params,
            'bada4_info': bada4_info
        }
    
    def extract_xml_params(self):
        import xml.etree.ElementTree as ET
        xml_path = f"/path/to/BADA4/Models/{self.aircraft_code}/{self.aircraft_code}.xml"
        
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            params = {}
            for elem in root.iter():
                if elem.text and elem.text.strip() and len(elem.text.strip()) < 100:
                    tag = elem.tag.split('}')[-1]  # 去除命名空间
                    try:
                        params[tag] = float(elem.text.strip())
                    except ValueError:
                        params[tag] = elem.text.strip()
            
            return params
        except Exception as e:
            return {'error': str(e)}

# 使用示例
extractor = AircraftDataExtractor('B738W26')
all_data = extractor.get_all_parameters()
print(f"提取到 {len(all_data['xml_parameters'])} 个参数")
```

### 性能计算示例

```python
def calculate_cruise_performance(aircraft_code, altitude_ft=35000, mach=0.78):
    """计算巡航性能"""
    
    # 初始化
    airplane = Airplane()
    airplane.aircraft_type = aircraft_code
    bada4 = BADA4(airplane)
    
    # 大气参数计算 (ISA标准大气)
    altitude_m = altitude_ft * 0.3048
    delta = (1 - 0.0065 * altitude_m / 288.15) ** 5.256  # 密度比
    theta = (1 - 0.0065 * altitude_m / 288.15)           # 温度比
    
    try:
        # 假设质量为MTOW的80%
        mass = 79016 * 0.8  # Boeing 737-800
        
        # 计算升力系数 (简化)
        rho = 1.225 * delta  # 密度
        v_tas = mach * 340 * (theta ** 0.5)  # 真空速
        S = 124.58  # 翼面积
        
        CL = (mass * 9.81) / (0.5 * rho * v_tas**2 * S)
        
        # 计算阻力系数
        CD = bada4.CD(HLid=0, LG=0, CL=CL, M=mach)
        
        # 计算燃油流量
        CF = bada4.CF(delta=delta, theta=theta, DeltaTemp=0)
        
        return {
            'altitude_ft': altitude_ft,
            'mach': mach,
            'lift_coefficient': CL,
            'drag_coefficient': CD,
            'fuel_flow_coefficient': CF,
            'true_airspeed_ms': v_tas,
            'mass_kg': mass
        }
        
    except Exception as e:
        return {'error': str(e)}

# 使用示例
performance = calculate_cruise_performance('B738W26')
print(performance)
```

## 📁 项目结构

```
pyBADA/
├── aircraft/
│   ├── BADA3/          # BADA 3.x 数据文件
│   ├── BADA4/          # BADA 4.x 数据文件
│   │   ├── Models/     # 飞机型号数据
│   │   │   ├── B738W26/
│   │   │   │   └── B738W26.xml
│   │   │   ├── A320-212/
│   │   │   │   └── A320-212.xml
│   │   │   └── ...
│   │   ├── GPF.xml     # 全局性能文件
│   │   └── release.csv # 发布信息
│   └── BADAH/          # 直升机数据
├── bada3.py            # BADA3 模块
├── bada4.py            # BADA4 模块 ⭐
├── aircraft.py         # 飞机基础类 ⭐
└── ...
```

## 🛠️ 故障排除

### 常见错误及解决方案

#### 1. 数据文件缺失
```
错误: DUMMY文件或空目录
解决: 确保已正确安装BADA数据文件
```

#### 2. API调用错误
```python
# ❌ 错误方式
airplane = Airplane('B738W26')  # 会报错

# ✅ 正确方式  
airplane = Airplane()
airplane.aircraft_type = 'B738W26'
bada4 = BADA4(airplane)
```

#### 3. 计算方法参数错误
```python
# ❌ 错误调用
CD = bada4.CD()  # 缺少必需参数

# ✅ 正确调用
CD = bada4.CD(HLid=0, LG=0, CL=0.5, M=0.78)
```

#### 4. XML解析失败
```python
# 检查文件是否存在
from pathlib import Path
xml_path = f"/path/to/Models/{aircraft_code}/{aircraft_code}.xml"
if not Path(xml_path).exists():
    print(f"文件不存在: {xml_path}")
```

### 诊断脚本

```python
def diagnose_pybada():
    """诊断pyBADA安装和配置"""
    
    print("🔍 pyBADA诊断")
    print("-" * 40)
    
    # 检查导入
    try:
        from pyBADA.bada4 import BADA4, Airplane, Parser
        print("✅ 模块导入成功")
    except ImportError as e:
        print(f"❌ 模块导入失败: {e}")
        return
    
    # 检查数据文件
    bada4_path = "/path/to/pyBADA/aircraft/BADA4"
    models_path = f"{bada4_path}/Models"
    
    if Path(models_path).exists():
        aircraft_count = len([d for d in Path(models_path).iterdir() if d.is_dir()])
        print(f"✅ 找到 {aircraft_count} 个飞机型号")
    else:
        print("❌ Models目录不存在")
        return
    
    # 测试基本功能
    try:
        airplane = Airplane()
        airplane.aircraft_type = 'B738W26'
        bada4 = BADA4(airplane)
        method_count = len([m for m in dir(bada4) if not m.startswith('_')])
        print(f"✅ BADA4对象创建成功，{method_count}个方法可用")
    except Exception as e:
        print(f"❌ BADA4测试失败: {e}")
    
    print("✅ 诊断完成")

# 运行诊断
diagnose_pybada()
```

## 📚 参考资源

### 官方资源
- **EUROCONTROL BADA**: https://www.eurocontrol.int/model/bada
- **GitHub仓库**: https://github.com/eurocontrol-bada/pybada
- **BADA用户手册**: 随数据包提供

### 学术文献
- BADA 4.0 Model Specification
- Aircraft Performance Modelling with BADA
- pyBADA: Easy BADA integration in Python

### 数据格式参考
- **XML Schema**: ACM_BADA4.xsd, GPF_BADA4.xsd
- **参数定义**: BADA User Manual Section 3
- **单位说明**: SI单位制 (kg, m, s, K)

## 📝 许可协议

pyBADA在欧盟公共许可证 v1.2 (EUPL-1.2) 下发布。

BADA数据受EUROCONTROL版权保护，使用需要许可协议。

## 🤝 贡献

欢迎提交Issue和Pull Request到官方GitHub仓库。

## 📞 支持

- **GitHub Issues**: 技术问题和bug报告
- **EUROCONTROL**: BADA数据相关问题
- **社区论坛**: 用户经验分享

---

**版本**: 基于 pyBADA 0.1.5 和 BADA 4.2  
**更新时间**: 2025年8月  
**适用于**: Python 3.8+ 环境

## 📚 参考资源

- **官方GitHub**: https://github.com/eurocontrol-bada/pybada
- **BADA用户手册**: EUROCONTROL官网
- **数据位置**: `/home/longqin/anaconda3/envs/atc_sim/lib/python3.12/site-packages/pyBADA/aircraft/BADA4`
- **支持的飞机**: 104种不同型号

---

*这份文档基于pyBADA 0.1.5版本和BADA 4.2数据集编写*