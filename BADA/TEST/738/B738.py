#!/usr/bin/env python3
"""
Boeing 737-800 (B738W26) 飞行包线绘制
基于成功获取的BADA参数
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pyBADA.bada4 import BADA4, Airplane

# 设置matplotlib中文显示
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

def create_b738_flight_envelope():
    """创建B738W26飞行包线"""
    
    print("🛩️ 生成Boeing 737-800飞行包线...")
    
    # 基于你成功获取的BADA参数
    aircraft_params = {
        'mtow': 79016,          # kg - 最大起飞重量
        'mlw': 66361,           # kg - 最大着陆重量  
        'wing_area': 124.58,    # m² - 翼面积
        'max_mach': 0.82,       # 最大马赫数
        'service_ceiling': 41000, # ft - 实用升限
        'length': 39.47,        # m - 机身长度
        'vfe': 162,             # kt - 襟翼限制速度
    }
    
    # 高度范围 (0 to 41000 ft)
    altitudes_ft = np.linspace(0, 41000, 100)
    altitudes_m = altitudes_ft * 0.3048
    
    # 不同重量配置
    weight_configs = {
        'MTOW (79,016 kg)': 1.0,
        'Typical (63,213 kg)': 0.8, 
        'Light (47,410 kg)': 0.6
    }
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # 颜色设置
    colors = ['#FF4444', '#44AA44', '#4444FF']
    
    for i, (config_name, weight_ratio) in enumerate(weight_configs.items()):
        
        stall_speeds = []
        max_speeds = []
        
        for alt_ft, alt_m in zip(altitudes_ft, altitudes_m):
            
            # === 大气参数计算 (ISA标准大气) ===
            if alt_m <= 11000:  # 对流层
                temp_ratio = 1 - 0.0065 * alt_m / 288.15
                density_ratio = temp_ratio ** 4.256
                temp = 288.15 * temp_ratio
            else:  # 平流层
                temp_ratio_11km = 1 - 0.0065 * 11000 / 288.15
                density_ratio_11km = temp_ratio_11km ** 4.256
                density_ratio = density_ratio_11km * np.exp(-(alt_m - 11000) / 6341.62)
                temp = 216.65
            
            # === 失速速度计算 ===
            mass = aircraft_params['mtow'] * weight_ratio
            
            # 最大升力系数 (估算值，基于Boeing 737)
            if weight_ratio == 1.0:  # MTOW
                cl_max = 1.6  # 清洁构型
            else:
                cl_max = 1.8  # 较轻重量时可以更高的迎角
            
            # 失速速度: V_stall = sqrt(2*W/(rho*S*CL_max))
            v_stall_ms = np.sqrt(2 * mass * 9.81 / 
                               (1.225 * density_ratio * aircraft_params['wing_area'] * cl_max))
            v_stall_kt = v_stall_ms * 1.94384  # 转换为节
            
            # === 最大速度计算 ===
            # 声速
            a = np.sqrt(1.4 * 287.053 * temp)
            
            # 马赫数限制速度
            v_mach_limit_ms = aircraft_params['max_mach'] * a
            v_mach_limit_kt = v_mach_limit_ms * 1.94384
            
            # 结构限制速度 (VMO/MMO)
            if alt_ft < 28000:
                v_structural_kt = 340  # VMO (340 KIAS below FL280)
            else:
                v_structural_kt = v_mach_limit_kt  # 高空以马赫数为准
            
            # 取较小值作为最大速度
            v_max_kt = min(v_mach_limit_kt, v_structural_kt)
            
            # 数据验证
            if v_stall_kt > 0 and v_max_kt > v_stall_kt:
                stall_speeds.append(v_stall_kt)
                max_speeds.append(v_max_kt)
            else:
                stall_speeds.append(np.nan)
                max_speeds.append(np.nan)
        
        # 绘制包线
        color = colors[i]
        
        # 失速边界
        ax.plot(stall_speeds, altitudes_ft, 
               color=color, linewidth=2.5, linestyle='--',
               label=f'{config_name} - Stall', alpha=0.8)
        
        # 最大速度边界  
        ax.plot(max_speeds, altitudes_ft,
               color=color, linewidth=2.5, linestyle='-',
               label=f'{config_name} - Max Speed', alpha=0.8)
        
        # 填充操作区域
        ax.fill_betweenx(altitudes_ft, stall_speeds, max_speeds,
                        color=color, alpha=0.15, label=f'{config_name} - Operating Area')
    
    # === 添加重要参考线 ===
    
    # 标准巡航高度
    cruise_levels = [25000, 30000, 35000, 39000, 41000]
    for fl in cruise_levels:
        ax.axhline(y=fl, color='gray', linestyle=':', alpha=0.5, linewidth=1)
        ax.text(480, fl+300, f'FL{fl//100}', fontsize=9, alpha=0.7, 
               bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.7))
    
    # 马赫0.82限制线
    mach_limit_speeds = []
    mach_altitudes = []
    for alt_ft in altitudes_ft:
        alt_m = alt_ft * 0.3048
        if alt_m <= 11000:
            temp = 288.15 * (1 - 0.0065 * alt_m / 288.15)
        else:
            temp = 216.65
        
        a = np.sqrt(1.4 * 287.053 * temp)
        v_mach_kt = 0.82 * a * 1.94384
        
        if v_mach_kt < 500:  # 合理范围内
            mach_limit_speeds.append(v_mach_kt)
            mach_altitudes.append(alt_ft)
    
    ax.plot(mach_limit_speeds, mach_altitudes, 'k-', linewidth=3, 
           alpha=0.8, label='Mach 0.82 Limit')
    
    # === 图形美化 ===
    ax.set_xlabel('Indicated Airspeed (knots)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Altitude (feet)', fontsize=14, fontweight='bold')
    ax.set_title('Boeing 737-800 (B738W26) Flight Envelope\n' + 
                'Based on EUROCONTROL BADA 4.2 Data', 
                fontsize=16, fontweight='bold', pad=20)
    
    # 网格
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # 坐标轴范围
    ax.set_xlim(0, 520)
    ax.set_ylim(0, 43000)
    
    # 图例
    ax.legend(loc='center right', fontsize=11, framealpha=0.9,
             bbox_to_anchor=(1.0, 0.5))
    
    # === 添加参数信息框 ===
    info_text = f"""AIRCRAFT SPECIFICATIONS:
    
• ICAO Code: B738
• Aircraft: Boeing 737-800
• Engine: CFM56-7B26/27
    
• MTOW: {aircraft_params['mtow']:,} kg ({aircraft_params['mtow']*2.20462:,.0f} lbs)
• MLW: {aircraft_params['mlw']:,} kg ({aircraft_params['mlw']*2.20462:,.0f} lbs)
• Wing Area: {aircraft_params['wing_area']:.1f} m² ({aircraft_params['wing_area']*10.764:.0f} ft²)
• Length: {aircraft_params['length']:.1f} m ({aircraft_params['length']*3.28084:.1f} ft)
    
• Max Mach: {aircraft_params['max_mach']}
• VFE: {aircraft_params['vfe']} kt
• Service Ceiling: {aircraft_params['service_ceiling']:,} ft
    
• Wing Loading: {aircraft_params['mtow']/aircraft_params['wing_area']:.0f} kg/m²"""
    
    # 信息框
    props = dict(boxstyle='round,pad=0.8', facecolor='lightblue', alpha=0.85)
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', fontfamily='monospace', bbox=props)
    
    # === 性能数据输出 ===
    print("\n📊 Boeing 737-800 Flight Envelope Summary")
    print("=" * 60)
    
    # 计算关键性能点
    sea_level_stall_mtow = np.sqrt(2 * aircraft_params['mtow'] * 9.81 / 
                                  (1.225 * aircraft_params['wing_area'] * 1.6)) * 1.94384
    
    cruise_alt_m = 35000 * 0.3048
    cruise_density_ratio = (1 - 0.0065 * cruise_alt_m / 288.15) ** 4.256
    cruise_temp = 288.15 * (1 - 0.0065 * cruise_alt_m / 288.15)
    cruise_stall_typical = np.sqrt(2 * aircraft_params['mtow'] * 0.8 * 9.81 / 
                                  (1.225 * cruise_density_ratio * aircraft_params['wing_area'] * 1.8)) * 1.94384
    
    cruise_sound_speed = np.sqrt(1.4 * 287.053 * cruise_temp)
    cruise_max_speed = 0.82 * cruise_sound_speed * 1.94384
    
    print(f"🌊 Sea Level Performance (MTOW):")
    print(f"   Stall Speed: {sea_level_stall_mtow:.0f} kt")
    print(f"   Max Speed: 340 kt (VMO)")
    
    print(f"\n✈️ Cruise Performance (FL350, Typical Weight):")
    print(f"   Stall Speed: {cruise_stall_typical:.0f} kt")
    print(f"   Max Speed: {cruise_max_speed:.0f} kt (Mach 0.82)")
    
    print(f"\n📐 Key Parameters:")
    print(f"   Wing Loading (MTOW): {aircraft_params['mtow']/aircraft_params['wing_area']:.0f} kg/m²")
    print(f"   Power Loading: ~4.0 kg/kN (estimated)")
    print(f"   Aspect Ratio: ~9.4 (estimated)")
    
    # 保存图形
    plt.tight_layout()
    plt.savefig('B738W26_Flight_Envelope.png', dpi=300, bbox_inches='tight',
               facecolor='white', edgecolor='none')
    
    print(f"\n✅ Flight envelope plot saved as: B738W26_Flight_Envelope.png")
    plt.show()
    
    return fig, ax

def create_bada4_enhanced_envelope():
    """使用BADA4对象增强的飞行包线"""
    
    print("🔧 尝试使用BADA4对象增强计算...")
    
    try:
        # 初始化BADA4对象
        airplane = Airplane()
        airplane.aircraft_type = 'B738W26'
        bada4 = BADA4(airplane)
        
        print("✅ BADA4对象创建成功")
        
        # 获取可用方法
        methods = [m for m in dir(bada4) if not m.startswith('_')]
        print(f"📋 可用BADA4方法: {len(methods)}个")
        
        # 测试计算示例
        try:
            # 尝试计算巡航条件下的参数
            altitude_m = 35000 * 0.3048
            
            # 大气参数
            delta = (1 - 0.0065 * altitude_m / 288.15) ** 5.256
            theta = (1 - 0.0065 * altitude_m / 288.15)
            
            # 尝试计算升力系数
            cl = bada4.CL(delta=delta, mass=63213, M=0.78)  # 典型巡航条件
            print(f"✅ 巡航升力系数: {cl:.3f}")
            
            # 尝试计算阻力系数
            cd = bada4.CD(HLid=0, LG=0, CL=cl, M=0.78)
            print(f"✅ 巡航阻力系数: {cd:.4f}")
            
            # 尝试计算燃油消耗
            cf = bada4.CF(delta=delta, theta=theta, DeltaTemp=0)
            print(f"✅ 燃油流量系数: {cf:.6f}")
            
        except Exception as e:
            print(f"⚠️ BADA4计算示例失败: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ BADA4增强失败: {e}")
        return False

if __name__ == "__main__":
    print("🛩️ Boeing 737-800 (B738W26) Flight Envelope Generator")
    print("=" * 65)
    
    # 尝试BADA4增强
    bada4_available = create_bada4_enhanced_envelope()
    
    print(f"\n🎨 生成飞行包线图...")
    
    # 生成飞行包线
    fig, ax = create_b738_flight_envelope()
    
    print(f"\n🎉 Complete!")
    print(f"📊 Flight envelope analysis finished")
    print(f"📁 High-resolution plot saved")
    print(f"💡 Use this data for flight planning and performance analysis")