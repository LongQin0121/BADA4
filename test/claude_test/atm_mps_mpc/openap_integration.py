#!/usr/bin/env python3
"""
OpenAP 航空器性能模型集成
作为BADA的开源替代方案

OpenAP GitHub: https://github.com/TUDelft-CNS-ATM/openap
文档: https://openap.dev/
"""

import requests
import json
import os
from urllib.parse import urljoin
import subprocess
import sys

class OpenAPIntegration:
    def __init__(self):
        self.base_url = "https://raw.githubusercontent.com/TUDelft-CNS-ATM/openap/master/"
        self.data_dir = "openap_data"
        self.aircraft_types = [
            'A319', 'A320', 'A321', 'A332', 'A333', 'A343', 'A359', 'A388',
            'B737', 'B738', 'B739', 'B744', 'B748', 'B752', 'B763', 'B772',
            'B773', 'B777', 'B787', 'B788', 'B789', 'CRJ2', 'CRJ9', 'E145',
            'E170', 'E190', 'E195'
        ]

    def setup_directory(self):
        """创建数据目录"""
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            print(f"Created directory: {self.data_dir}")

    def download_aircraft_data(self):
        """下载航空器性能数据"""
        self.setup_directory()
        
        # 下载航空器数据文件
        data_files = [
            'openap/data/aircraft/aircraft.txt',
            'openap/data/engine/engines.txt'
        ]
        
        for file_path in data_files:
            url = urljoin(self.base_url, file_path)
            local_path = os.path.join(self.data_dir, os.path.basename(file_path))
            
            try:
                response = requests.get(url)
                response.raise_for_status()
                
                with open(local_path, 'w') as f:
                    f.write(response.text)
                print(f"Downloaded: {local_path}")
                
            except requests.RequestException as e:
                print(f"Failed to download {file_path}: {e}")

    def download_performance_models(self):
        """下载性能模型文件"""
        models_dir = os.path.join(self.data_dir, 'models')
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)

        for aircraft in self.aircraft_types:
            # 下载拖阻模型
            drag_url = f"{self.base_url}openap/data/dragpolar/{aircraft.lower()}.txt"
            drag_path = os.path.join(models_dir, f"{aircraft}_drag.txt")
            
            # 下载燃油模型
            fuel_url = f"{self.base_url}openap/data/fuel/{aircraft.lower()}.txt"
            fuel_path = os.path.join(models_dir, f"{aircraft}_fuel.txt")
            
            for url, path in [(drag_url, drag_path), (fuel_url, fuel_path)]:
                try:
                    response = requests.get(url)
                    if response.status_code == 200:
                        with open(path, 'w') as f:
                            f.write(response.text)
                        print(f"Downloaded: {path}")
                except requests.RequestException:
                    print(f"Failed to download: {url}")

    def create_simple_performance_calculator(self):
        """创建简单的性能计算器"""
        calculator_code = '''
import math
import os

class SimpleAircraftPerformance:
    """简化的航空器性能计算器（基于OpenAP原理）"""
    
    def __init__(self):
        # 常见机型的基本参数 (基于OpenAP数据)
        self.aircraft_db = {
            'A320': {
                'mtow': 78000,      # kg
                'max_alt': 39000,   # ft
                'cruise_speed': 447, # kts
                'fuel_capacity': 24210, # kg
                'engine_type': 'CFM56',
                'wingspan': 35.8,   # m
                'length': 37.6      # m
            },
            'B738': {
                'mtow': 79016,
                'max_alt': 41000,
                'cruise_speed': 453,
                'fuel_capacity': 26020,
                'engine_type': 'CFM56',
                'wingspan': 35.8,
                'length': 39.5
            },
            'A333': {
                'mtow': 242000,
                'max_alt': 41000,
                'cruise_speed': 470,
                'fuel_capacity': 97530,
                'engine_type': 'CF6',
                'wingspan': 60.3,
                'length': 63.7
            }
        }
    
    def get_aircraft_data(self, aircraft_type):
        """获取航空器基本数据"""
        return self.aircraft_db.get(aircraft_type.upper(), None)
    
    def calculate_climb_performance(self, aircraft_type, weight_kg, altitude_ft, temperature_c=15):
        """计算爬升性能"""
        aircraft = self.get_aircraft_data(aircraft_type)
        if not aircraft:
            return None
        
        # 简化的爬升率计算 (基于推重比和空气密度)
        weight_ratio = weight_kg / aircraft['mtow']
        altitude_ratio = altitude_ft / aircraft['max_alt']
        
        # 标准大气密度比
        density_ratio = (1 - 0.0065 * altitude_ft * 0.3048 / 288.15) ** 4.256
        
        # 基础爬升率 (ft/min)
        base_climb_rate = 2500 * (1 - weight_ratio) * density_ratio
        
        return max(base_climb_rate, 500)  # 最小爬升率 500 ft/min
    
    def calculate_fuel_flow(self, aircraft_type, altitude_ft, speed_kts, phase='cruise'):
        """计算燃油流量"""
        aircraft = self.get_aircraft_data(aircraft_type)
        if not aircraft:
            return None
        
        # 基础燃油流量 (kg/h)
        if phase == 'cruise':
            base_flow = 2500 if 'A33' in aircraft_type else 1800
        elif phase == 'climb':
            base_flow = 3000 if 'A33' in aircraft_type else 2200
        elif phase == 'descent':
            base_flow = 800 if 'A33' in aircraft_type else 600
        else:
            base_flow = 2000
        
        # 高度和速度修正
        altitude_factor = 1 - (altitude_ft / 100000)  # 高度越高燃油流量越低
        speed_factor = (speed_kts / aircraft['cruise_speed']) ** 1.5
        
        return base_flow * altitude_factor * speed_factor
    
    def calculate_range(self, aircraft_type, fuel_kg, cruise_alt=35000, cruise_speed=None):
        """计算航程"""
        aircraft = self.get_aircraft_data(aircraft_type)
        if not aircraft:
            return None
        
        if cruise_speed is None:
            cruise_speed = aircraft['cruise_speed']
        
        # 计算巡航燃油流量
        fuel_flow = self.calculate_fuel_flow(aircraft_type, cruise_alt, cruise_speed, 'cruise')
        
        # 考虑起飞降落燃油消耗
        taxi_takeoff_fuel = 500  # kg
        landing_fuel = 300       # kg
        reserve_fuel = fuel_kg * 0.05  # 5% 备用燃油
        
        cruise_fuel = fuel_kg - taxi_takeoff_fuel - landing_fuel - reserve_fuel
        
        if cruise_fuel <= 0:
            return 0
        
        # 航程计算 (海里)
        flight_time_hours = cruise_fuel / fuel_flow
        range_nm = flight_time_hours * cruise_speed
        
        return range_nm

# 使用示例
if __name__ == "__main__":
    perf = SimpleAircraftPerformance()
    
    # 测试A320性能
    aircraft_type = 'A320'
    print(f"=== {aircraft_type} 性能计算 ===")
    
    # 基本参数
    data = perf.get_aircraft_data(aircraft_type)
    print(f"最大起飞重量: {data['mtow']:,} kg")
    print(f"最大飞行高度: {data['max_alt']:,} ft")
    print(f"巡航速度: {data['cruise_speed']} kts")
    
    # 爬升性能
    climb_rate = perf.calculate_climb_performance(aircraft_type, 70000, 25000)
    print(f"25,000ft 爬升率: {climb_rate:.0f} ft/min")
    
    # 燃油流量
    fuel_flow = perf.calculate_fuel_flow(aircraft_type, 35000, 450, 'cruise')
    print(f"巡航燃油流量: {fuel_flow:.0f} kg/h")
    
    # 航程
    range_nm = perf.calculate_range(aircraft_type, 20000)
    print(f"20吨燃油航程: {range_nm:.0f} 海里")
'''
        
        with open(os.path.join(self.data_dir, 'aircraft_performance.py'), 'w') as f:
            f.write(calculator_code)
        
        print("Created simple aircraft performance calculator")

    def install_openap_if_possible(self):
        """尝试安装OpenAP包"""
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'openap'])
            print("Successfully installed OpenAP package")
            return True
        except subprocess.CalledProcessError:
            print("Failed to install OpenAP package - using manual implementation")
            return False

    def create_usage_example(self):
        """创建使用示例"""
        example_code = '''
#!/usr/bin/env python3
"""
OpenAP 使用示例
"""

try:
    # 尝试使用官方OpenAP包
    import openap
    from openap import prop, WRAP, FuelFlow, Emission
    
    def openap_official_example():
        print("=== OpenAP 官方包示例 ===")
        
        # 获取航空器属性
        aircraft = prop.aircraft('A320')
        print(f"A320 翼展: {aircraft['wing']['span']} m")
        print(f"A320 最大起飞重量: {aircraft['limits']['MTOW']} kg")
        
        # 计算燃油流量
        fuelflow = FuelFlow(ac='A320')
        FF = fuelflow.enroute(mass=60000, tas=230, alt=32000)
        print(f"燃油流量: {FF:.2f} kg/s")
        
        # 计算排放
        emission = Emission(ac='A320')
        CO2 = emission.co2(FF)
        print(f"CO2排放: {CO2:.2f} kg/s")

    openap_official_example()
    
except ImportError:
    print("OpenAP包未安装，使用简化版本")
    
    # 使用我们的简化实现
    import sys
    import os
    sys.path.append('openap_data')
    
    from aircraft_performance import SimpleAircraftPerformance
    
    def simplified_example():
        print("=== 简化航空器性能计算示例 ===")
        
        perf = SimpleAircraftPerformance()
        
        aircraft_types = ['A320', 'B738', 'A333']
        
        for ac_type in aircraft_types:
            print(f"\\n--- {ac_type} ---")
            data = perf.get_aircraft_data(ac_type)
            if data:
                print(f"MTOW: {data['mtow']:,} kg")
                print(f"巡航速度: {data['cruise_speed']} kts")
                
                # 性能计算
                climb = perf.calculate_climb_performance(ac_type, data['mtow']*0.8, 25000)
                fuel_flow = perf.calculate_fuel_flow(ac_type, 35000, data['cruise_speed'])
                range_nm = perf.calculate_range(ac_type, data['fuel_capacity']*0.8)
                
                print(f"爬升率 (25000ft): {climb:.0f} ft/min")
                print(f"巡航油耗: {fuel_flow:.0f} kg/h")
                print(f"航程: {range_nm:.0f} nm")
    
    simplified_example()
'''
        
        with open('openap_example.py', 'w') as f:
            f.write(example_code)
        
        print("Created usage example: openap_example.py")

    def run_integration(self):
        """运行完整的集成过程"""
        print("=== OpenAP 集成开始 ===")
        
        # 尝试安装官方包
        success = self.install_openap_if_possible()
        
        # 下载数据
        print("\\n下载OpenAP数据...")
        self.download_aircraft_data()
        self.download_performance_models()
        
        # 创建简化实现
        print("\\n创建简化性能计算器...")
        self.create_simple_performance_calculator()
        
        # 创建使用示例
        print("\\n创建使用示例...")
        self.create_usage_example()
        
        print("\\n=== OpenAP 集成完成 ===")
        print(f"数据目录: {self.data_dir}")
        print("运行示例: python openap_example.py")

if __name__ == "__main__":
    integration = OpenAPIntegration()
    integration.run_integration()