
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
            print(f"\n--- {ac_type} ---")
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
