
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
         python openap_data/aircraft_performance.py
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
