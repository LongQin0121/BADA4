# pyBADA 完整参考指南和即用代码

## 📖 目录
1. [快速开始](#快速开始)
2. [通用飞机数据提取器](#通用飞机数据提取器)
3. [参数名称映射表](#参数名称映射表)
4. [常用飞机列表](#常用飞机列表)
5. [BADA4计算方法详解](#bada4计算方法详解)
6. [性能计算示例](#性能计算示例)
7. [故障排除指南](#故障排除指南)

---

## 🚀 快速开始

### 安装和路径设置
```python
import pyBADA
from pathlib import Path

# 获取BADA数据路径
BADA4_PATH = Path(pyBADA.__file__).parent / "aircraft" / "BADA4"
MODELS_PATH = BADA4_PATH / "Models"

print(f"BADA4路径: {BADA4_PATH}")
print(f"飞机模型路径: {MODELS_PATH}")
```

### 基本使用模板
```python
from pyBADA.bada4 import BADA4, Airplane
import xml.etree.ElementTree as ET

def get_aircraft_data(aircraft_code):
    """通用飞机数据获取模板"""
    
    # 1. 解析XML参数
    xml_path = MODELS_PATH / aircraft_code / f"{aircraft_code}.xml"
    params = parse_aircraft_xml(xml_path)
    
    # 2. 创建配置的飞机对象
    airplane = create_airplane_object(aircraft_code, params)
    
    # 3. 执行性能计算
    results = perform_calculations(airplane, params)
    
    return airplane, params, results
```

---

## 🔧 通用飞机数据提取器

### 完整的即用代码
```python
#!/usr/bin/env python3
"""
pyBADA 通用飞机数据提取器 - 即用版本
支持所有104种飞机型号的数据提取和性能计算
"""

import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import pyBADA
from pyBADA.bada4 import BADA4, Airplane

class AircraftDataExtractor:
    """通用飞机数据提取器"""
    
    def __init__(self):
        self.bada4_path = Path(pyBADA.__file__).parent / "aircraft" / "BADA4"
        self.models_path = self.bada4_path / "Models"
        self.available_aircraft = self._get_available_aircraft()
    
    def _get_available_aircraft(self):
        """获取所有可用的飞机型号"""
        if not self.models_path.exists():
            return []
        return [d.name for d in self.models_path.iterdir() if d.is_dir()]
    
    def get_aircraft_data(self, aircraft_code):
        """获取指定飞机的完整数据"""
        
        if aircraft_code not in self.available_aircraft:
            raise ValueError(f"飞机型号 {aircraft_code} 不存在。可用型号: {self.available_aircraft[:10]}...")
        
        print(f"🛩️ 获取 {aircraft_code} 数据...")
        
        # 1. 解析XML参数
        params = self._parse_xml_parameters(aircraft_code)
        
        # 2. 创建配置的飞机对象
        airplane = self._create_airplane_object(aircraft_code, params)
        
        # 3. 显示飞机规格
        self._display_specifications(aircraft_code, params)
        
        # 4. 执行性能计算
        results = self._perform_calculations(airplane, params)
        
        return {
            'aircraft_code': aircraft_code,
            'airplane_object': airplane,
            'parameters': params,
            'performance_results': results,
            'specifications': self._get_key_specifications(params)
        }
    
    def _parse_xml_parameters(self, aircraft_code):
        """解析XML参数"""
        
        xml_path = self.models_path / aircraft_code / f"{aircraft_code}.xml"
        
        if not xml_path.exists():
            raise FileNotFoundError(f"XML文件不存在: {xml_path}")
        
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        params = {}
        for elem in root.iter():
            if elem.text and elem.text.strip():
                tag = elem.tag.split('}')[-1]
                text = elem.text.strip()
                
                try:
                    if '.' in text or 'e' in text.lower():
                        params[tag] = float(text)
                    else:
                        params[tag] = int(text)
                except ValueError:
                    params[tag] = text
        
        return params
    
    def _create_airplane_object(self, aircraft_code, params):
        """创建配置的飞机对象"""
        
        airplane = Airplane()
        airplane.aircraft_type = aircraft_code
        airplane.code = aircraft_code
        
        # 使用标准参数映射
        param_mapping = self._get_parameter_mapping()
        
        for airplane_attr, xml_keys in param_mapping.items():
            if isinstance(xml_keys, list):
                # 尝试多个可能的键名
                for xml_key in xml_keys:
                    if xml_key in params:
                        setattr(airplane, airplane_attr, params[xml_key])
                        break
            else:
                # 单个键名
                if xml_keys in params:
                    setattr(airplane, airplane_attr, params[xml_keys])
        
        # 设置默认值
        if not hasattr(airplane, 'engineType') or airplane.engineType is None:
            airplane.engineType = params.get('engineModel', 'Unknown')
        
        return airplane
    
    def _get_parameter_mapping(self):
        """获取参数映射表 - 基于实际发现的映射关系"""
        
        return {
            # 几何参数
            'S': ['S', 'wingArea'],
            'span': ['span', 'wingspan'],
            'length': ['length', 'fuselageLength'],
            
            # 重量参数 (多种可能的名称)
            'mRef': ['MREF', 'mRef', 'referenceWeight'],
            'mMin': ['MZFW', 'mMin', 'minimumWeight'],
            'mMax': ['MTOW', 'mMax', 'maximumWeight'],
            'mPyld': ['MPL', 'mPyld', 'payloadWeight'],
            
            # 性能参数
            'VMO': ['VMO', 'maxOperatingSpeed'],
            'MMO': ['MMO', 'maxOperatingMach'],
            'hMO': ['hMO', 'maxOperatingAltitude'],
            'hMax': ['hMax', 'maxAltitude', 'serviceCeiling'],
            
            # 空气动力学参数
            'CL_max': ['CL_max', 'CLmax', 'maxLiftCoeff'],
            'CD0': ['CD0', 'zeroLiftDrag'],
            'CD2': ['CD2', 'inducedDragCoeff'],
            
            # 发动机参数
            'engineType': ['engineModel', 'engineType'],
            'nEng': ['nEng', 'numberOfEngines'],
            
            # 跑道性能
            'TOL': ['TOL', 'takeoffLength'],
            'LDL': ['LDL', 'landingLength'],
        }
    
    def _display_specifications(self, aircraft_code, params):
        """显示飞机规格"""
        
        print(f"\n📋 {aircraft_code} 技术规格")
        print("-" * 50)
        
        # 基本信息
        basic_info = {
            'ICAO代码': aircraft_code[:4],
            '制造商': self._get_manufacturer(aircraft_code),
            '飞机类型': params.get('type', 'Unknown'),
            '发动机': params.get('engineModel', 'Unknown'),
        }
        
        print("🔧 基本信息:")
        for key, value in basic_info.items():
            print(f"  {key}: {value}")
        
        # 重量参数
        weight_params = ['MTOW', 'MREF', 'MLW', 'MZFW', 'MPL', 'MFL']
        print(f"\n⚖️ 重量参数 (kg):")
        for param in weight_params:
            if param in params and params[param] > 0:
                print(f"  {param}: {params[param]:,}")
        
        # 几何参数
        geo_params = [('S', 'm²'), ('span', 'm'), ('length', 'm')]
        print(f"\n📐 几何参数:")
        for param, unit in geo_params:
            if param in params and params[param] > 0:
                print(f"  {param}: {params[param]:.2f} {unit}")
        
        # 性能参数
        perf_params = [('VMO', 'kt'), ('MMO', 'Mach'), ('hMax', 'ft')]
        print(f"\n🚀 性能参数:")
        for param, unit in perf_params:
            if param in params and params[param] > 0:
                if 'Mach' in unit:
                    print(f"  {param}: {params[param]:.3f} {unit}")
                else:
                    print(f"  {param}: {params[param]:,} {unit}")
    
    def _get_manufacturer(self, aircraft_code):
        """根据飞机代码识别制造商"""
        
        if aircraft_code.startswith(('A3', 'A2')):
            return 'Airbus'
        elif aircraft_code.startswith('B'):
            return 'Boeing'
        elif aircraft_code.startswith('ATR'):
            return 'ATR'
        elif aircraft_code.startswith('EMB'):
            return 'Embraer'
        elif aircraft_code.startswith('F'):
            return 'Fokker'
        elif aircraft_code.startswith('MD'):
            return 'McDonnell Douglas'
        else:
            return 'Unknown'
    
    def _perform_calculations(self, airplane, params):
        """执行性能计算"""
        
        try:
            bada4 = BADA4(airplane)
            
            # 标准计算条件
            conditions = [
                {'name': '海平面', 'alt_ft': 0, 'mach': 0.3},
                {'name': 'FL250', 'alt_ft': 25000, 'mach': 0.7},
                {'name': 'FL350', 'alt_ft': 35000, 'mach': 0.78},
                {'name': 'FL390', 'alt_ft': 39000, 'mach': 0.82},
            ]
            
            results = []
            
            for condition in conditions:
                # 计算大气参数
                alt_m = condition['alt_ft'] * 0.3048
                if alt_m <= 11000:
                    delta = (1 - 0.0065 * alt_m / 288.15) ** 5.256
                    theta = (1 - 0.0065 * alt_m / 288.15)
                else:
                    temp_11km = 288.15 * (1 - 0.0065 * 11000 / 288.15)
                    delta_11km = (temp_11km / 288.15) ** 5.256
                    delta = delta_11km * np.exp(-(alt_m - 11000) / 6341.62)
                    theta = 216.65 / 288.15
                
                # 质量假设
                mass = params.get('MREF', params.get('MTOW', 70000))
                
                # 执行计算
                calculations = {}
                calc_methods = [
                    ('CDClean', lambda: bada4.CDClean(M=condition['mach'])),
                    ('CLmax', lambda: bada4.CLmax(HLid=0, LG=0, M=condition['mach'])),
                    ('CD', lambda: bada4.CD(HLid=0, LG=0, CL=0.5, M=condition['mach'])),
                    ('CL', lambda: bada4.CL(delta=delta, mass=mass, M=condition['mach'])),
                ]
                
                for calc_name, calc_func in calc_methods:
                    try:
                        calculations[calc_name] = calc_func()
                    except:
                        calculations[calc_name] = None
                
                results.append({
                    'condition': condition,
                    'atmosphere': {'delta': delta, 'theta': theta},
                    'mass': mass,
                    'calculations': calculations
                })
            
            return results
            
        except Exception as e:
            print(f"⚠️ 性能计算失败: {e}")
            return []
    
    def _get_key_specifications(self, params):
        """提取关键规格用于快速参考"""
        
        specs = {}
        
        # 重量规格
        weight_keys = ['MTOW', 'MREF', 'MLW', 'MZFW']
        for key in weight_keys:
            if key in params:
                specs[f'weight_{key.lower()}'] = params[key]
        
        # 几何规格
        geo_keys = ['S', 'span', 'length']
        for key in geo_keys:
            if key in params:
                specs[f'geometry_{key}'] = params[key]
        
        # 性能规格
        perf_keys = ['VMO', 'MMO', 'hMax']
        for key in perf_keys:
            if key in params:
                specs[f'performance_{key.lower()}'] = params[key]
        
        return specs
    
    def list_available_aircraft(self, manufacturer=None):
        """列出可用的飞机型号"""
        
        aircraft_list = self.available_aircraft
        
        if manufacturer:
            aircraft_list = [ac for ac in aircraft_list 
                           if self._get_manufacturer(ac).lower() == manufacturer.lower()]
        
        return sorted(aircraft_list)
    
    def get_manufacturer_aircraft(self):
        """按制造商分组的飞机列表"""
        
        manufacturers = {}
        for aircraft in self.available_aircraft:
            mfr = self._get_manufacturer(aircraft)
            if mfr not in manufacturers:
                manufacturers[mfr] = []
            manufacturers[mfr].append(aircraft)
        
        # 排序
        for mfr in manufacturers:
            manufacturers[mfr] = sorted(manufacturers[mfr])
        
        return manufacturers

# 使用示例
def main():
    """主函数 - 使用示例"""
    
    print("🚀 pyBADA 通用飞机数据提取器")
    print("=" * 50)
    
    # 创建提取器
    extractor = AircraftDataExtractor()
    
    # 显示可用飞机
    print(f"📊 总共可用飞机数量: {len(extractor.available_aircraft)}")
    
    # 按制造商显示
    manufacturers = extractor.get_manufacturer_aircraft()
    for mfr, aircraft_list in manufacturers.items():
        print(f"  {mfr}: {len(aircraft_list)} 架")
    
    # 获取A320-232数据（示例）
    aircraft_code = 'A320-232'
    if aircraft_code in extractor.available_aircraft:
        data = extractor.get_aircraft_data(aircraft_code)
        
        print(f"\n✅ {aircraft_code} 数据获取完成")
        print(f"📊 参数数量: {len(data['parameters'])}")
        print(f"🧮 计算结果: {len(data['performance_results'])} 个条件")
        
        # 显示关键规格
        specs = data['specifications']
        print(f"\n🎯 关键规格:")
        for key, value in specs.items():
            print(f"  {key}: {value}")

if __name__ == "__main__":
    main()
```

---

## 📊 参数名称映射表

### 重量参数映射
| 标准名称 | XML中的实际名称 | 描述 |
|---------|---------------|------|
| mRef | **MREF** | 参考重量 |
| mMax | **MTOW** | 最大起飞重量 |
| mMin | **MZFW** | 最大零燃油重量 |
| mPyld | **MPL** | 最大载荷重量 |
| - | **MLW** | 最大着陆重量 |
| - | **MFL** | 最大燃油重量 |

### 性能参数映射
| 标准名称 | XML中的实际名称 | 描述 |
|---------|---------------|------|
| VMO | **VMO** | 最大操作速度 (kt) |
| MMO | **MMO** | 最大操作马赫数 |
| hMax | **hMax** | 最大高度 (ft) |
| hMO | **hMO** | 最大操作高度 (ft) |

### 几何参数映射
| 标准名称 | XML中的实际名称 | 描述 |
|---------|---------------|------|
| S | **S** | 翼面积 (m²) |
| span | **span** | 翼展 (m) |
| length | **length** | 机身长度 (m) |

### 空气动力学参数映射
| 标准名称 | XML中的实际名称 | 描述 |
|---------|---------------|------|
| CL_max | **CL_max** | 最大升力系数 |
| CD0 | **CD0** | 零升阻力系数 |
| CD2 | **CD2** | 诱导阻力系数 |

---

## ✈️ 常用飞机列表

### Airbus系列 (32种)
```python
airbus_aircraft = [
    # A320系列
    'A320-212', 'A320-214', 'A320-231', 'A320-232',
    'A318-112', 'A319-114', 'A319-131',
    'A321-111', 'A321-131',
    
    # A330系列  
    'A330-203', 'A330-223', 'A330-243', 
    'A330-301', 'A330-321', 'A330-341',
    
    # A340系列
    'A340-213', 'A340-313', 'A340-541', 'A340-642',
    
    # A350/A380系列
    'A350-941', 'A380-841', 'A380-861'
]
```

### Boeing系列 (31种)
```python
boeing_aircraft = [
    # 737系列
    'B737W24', 'B738W26', 'B739ERW26',
    
    # 747系列
    'B744GE', 'B744ERGE', 'B748F',
    
    # 777系列
    'B772LR', 'B772RR92', 'B773ERGE115B',
    
    # 787系列
    'B788GE67', 'B788RR53', 'B789GE75'
]
```

---

## 🧮 BADA4计算方法详解

### 主要计算方法
```python
# 创建BADA4对象
bada4 = BADA4(airplane)

# 空气动力学计算
CD = bada4.CD(HLid=0, LG=0, CL=0.5, M=0.78)          # 阻力系数
CL = bada4.CL(delta=0.3, mass=70000, M=0.78)         # 升力系数
CDClean = bada4.CDClean(M=0.78)                      # 清洁构型阻力系数
CLmax = bada4.CLmax(HLid=0, LG=0, M=0.78)           # 最大升力系数

# 燃油和推力计算
CF = bada4.CF(delta=0.3, theta=0.76, DeltaTemp=0)    # 燃油流量系数
CF_idle = bada4.CF_idle(delta=0.3, theta=0.76)       # 慢车燃油流量
```

### 参数说明
- **HLid**: 高升力装置ID (0=收起, 1-7=不同襟翼位置)
- **LG**: 起落架状态 (0=收起, 1=放下)
- **M**: 马赫数
- **delta**: 大气密度比 (ρ/ρ₀)
- **theta**: 大气温度比 (T/T₀)
- **mass**: 飞机质量 (kg)
- **CL**: 升力系数

---

## 🔧 性能计算示例

### 完整的性能计算模板
```python
def calculate_aircraft_performance(aircraft_code, altitude_ft=35000, mach=0.78):
    """通用飞机性能计算模板"""
    
    # 1. 获取飞机数据
    extractor = AircraftDataExtractor()
    data = extractor.get_aircraft_data(aircraft_code)
    
    airplane = data['airplane_object']
    params = data['parameters']
    
    # 2. 创建BADA4对象
    bada4 = BADA4(airplane)
    
    # 3. 计算大气参数
    alt_m = altitude_ft * 0.3048
    if alt_m <= 11000:
        delta = (1 - 0.0065 * alt_m / 288.15) ** 5.256
        theta = (1 - 0.0065 * alt_m / 288.15)
        temp = 288.15 * theta
    else:
        temp_11km = 288.15 * (1 - 0.0065 * 11000 / 288.15)
        delta_11km = (temp_11km / 288.15) ** 5.256
        delta = delta_11km * np.exp(-(alt_m - 11000) / 6341.62)
        theta = 216.65 / 288.15
        temp = 216.65
    
    # 4. 设置质量
    mass = params.get('MREF', params.get('MTOW', 70000))
    
    # 5. 执行计算
    results = {}
    
    try:
        # 基本空气动力学
        results['CD_clean'] = bada4.CDClean(M=mach)
        results['CL_max'] = bada4.CLmax(HLid=0, LG=0, M=mach)
        results['CD'] = bada4.CD(HLid=0, LG=0, CL=0.5, M=mach)
        results['CL'] = bada4.CL(delta=delta, mass=mass, M=mach)
        
        # 燃油流量
        results['CF'] = bada4.CF(delta=delta, theta=theta, DeltaTemp=0)
        
        # 计算实用参数
        S = params.get('S', 120)  # 翼面积
        rho = 1.225 * delta
        sound_speed = np.sqrt(1.4 * 287.053 * temp)
        tas = mach * sound_speed
        
        # 失速速度
        if results['CL_max']:
            v_stall = np.sqrt(2 * mass * 9.81 / (rho * S * results['CL_max']))
            results['stall_speed_ms'] = v_stall
            results['stall_speed_kt'] = v_stall * 1.94384
        
        # 阻力
        if results['CD']:
            drag_force = 0.5 * rho * tas**2 * S * results['CD']
            results['drag_force_kN'] = drag_force / 1000
        
        # 升力
        if results['CL']:
            lift_force = 0.5 * rho * tas**2 * S * results['CL']
            results['lift_force_kN'] = lift_force / 1000
        
    except Exception as e:
        results['error'] = str(e)
    
    return {
        'aircraft': aircraft_code,
        'conditions': {
            'altitude_ft': altitude_ft,
            'mach': mach,
            'delta': delta,
            'theta': theta,
            'mass': mass
        },
        'results': results
    }

# 使用示例
performance = calculate_aircraft_performance('A320-232', 35000, 0.78)
print(f"失速速度: {performance['results'].get('stall_speed_kt', 'N/A')} kt")
print(f"阻力: {performance['results'].get('drag_force_kN', 'N/A')} kN")
```

---

## 🛠️ 故障排除指南

### 常见问题和解决方案

#### 1. 找不到飞机型号
```python
# 问题：ValueError: 飞机型号 XXX 不存在
# 解决：
extractor = AircraftDataExtractor()
available = extractor.list_available_aircraft()
print("可用飞机:", available[:10])
```

#### 2. XML文件解析失败
```python
# 问题：FileNotFoundError 或解析错误
# 解决：检查BADA数据文件安装
import pyBADA
bada_path = Path(pyBADA.__file__).parent / "aircraft" / "BADA4"
print(f"BADA路径存在: {bada_path.exists()}")
print(f"Models目录存在: {(bada_path / 'Models').exists()}")
```

#### 3. BADA4计算失败
```python
# 问题：计算方法调用失败
# 解决：检查Airplane对象的关键属性
def diagnose_airplane(airplane):
    critical_attrs = ['S', 'mRef', 'engineType']
    for attr in critical_attrs:
        if hasattr(airplane, attr):
            print(f"✅ {attr}: {getattr(airplane, attr)}")
        else:
            print(f"❌ 缺少: {attr}")
```

#### 4. 参数映射错误
```python
# 问题：参数名称不匹配
# 解决：使用参数检查器
def check_aircraft_parameters(aircraft_code):
    extractor = AircraftDataExtractor()
    data = extractor.get_aircraft_data(aircraft_code)
    params = data['parameters']
    
    print("所有参数:")
    for key, value in sorted(params.items()):
        print(f"  {key}: {value}")
```

---

## 📝 快速参考卡

### 一键获取飞机数据
```python
# 导入
from aircraft_data_extractor import AircraftDataExtractor

# 创建提取器
extractor = AircraftDataExtractor()

# 获取数据
data = extractor.get_aircraft_data('A320-232')

# 访问数据
airplane = data['airplane_object']
params = data['parameters'] 
results = data['performance_results']
specs = data['specifications']
```

### 常用飞机快速列表
```python
# Airbus
'A320-232', 'A330-223', 'A350-941', 'A380-841'

# Boeing  
'B738W26', 'B772RR92', 'B788GE67', 'B744GE'

# 其他
'ATR72-600', 'EMB-190STD', 'F100-620'
```

### 标准性能计算
```python
# FL350, M0.78巡航条件
performance = calculate_aircraft_performance('A320-232', 35000, 0.78)

# 获取关键参数
stall_speed = performance['results']['stall_speed_kt']
drag_force = performance['results']['drag_force_kN']
```

---

## 💡 最佳实践

1. **始终检查数据可用性**：使用 `list_available_aircraft()` 确认型号存在
2. **处理异常情况**：包装计算代码在 try-except 块中
3. **验证结果合理性**：检查计算结果是否在预期范围内
4. **缓存常用数据**：避免重复解析同一飞机的XML文件
5. **文档化参数映射**：为新发现的参数映射做记录

---

*此文档基于pyBADA 0.1.5和BADA 4.2创建，涵盖104种飞机型号的完整数据提取方法。*