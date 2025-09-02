#!/usr/bin/env python3
"""
Boeing 737-800 简化飞行包线
80%全重配置，双轴显示
"""

import numpy as np
import matplotlib.pyplot as plt
from pyBADA.bada4 import BADA4, Airplane

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

def create_simple_flight_envelope():
    """创建简化的B738飞行包线"""
    
    print("🛩️ Boeing 737-800 飞行包线")
    print("📊 配置：80%全重 (63,213 kg)")
    print("=" * 40)
    
    # 飞机参数
    aircraft = {
        'mtow': 79016,         # kg
        'operating_weight': 79016 * 0.8,  # 80%全重
        'wing_area': 124.58,   # m²
        'max_mach': 0.82,
        'service_ceiling': 41000,  # ft
    }
    
    # 高度范围
    altitudes_ft = np.linspace(0, 41000, 100)
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(14, 9))
    
    # 计算飞行包线
    stall_speeds, max_speeds, valid_altitudes = calculate_flight_envelope(
        altitudes_ft, aircraft)
    
    # 绘制包线
    ax.fill_betweenx(valid_altitudes, stall_speeds, max_speeds,
                    color='lightblue', alpha=0.3, label='Operating Envelope')
    
    ax.plot(stall_speeds, valid_altitudes, 'r-', linewidth=3,
           label='Stall Speed')
    
    ax.plot(max_speeds, valid_altitudes, 'b-', linewidth=3,
           label='Maximum Speed')
    
    # 绘制马赫数等值线
    draw_mach_lines(ax, altitudes_ft)
    
    # 设置双轴
    setup_dual_axis(ax)
    
    # 添加参考线
    add_reference_lines(ax)
    
    # 美化
    ax.set_xlabel('Indicated Airspeed (knots)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Altitude (feet)', fontsize=14, fontweight='bold')
    ax.set_title('Boeing 737-800 Flight Envelope\nOperating Weight: 63,213 kg (80% MTOW)', 
                fontsize=16, fontweight='bold', pad=20)
    
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    ax.set_xlim(50, 500)
    ax.set_ylim(0, 43000)
    
    # 图例
    ax.legend(loc='upper right', fontsize=12, framealpha=0.9)
    
    # 性能信息
    add_performance_info(ax, aircraft, stall_speeds, max_speeds, valid_altitudes)
    
    # 保存和显示
    plt.tight_layout()
    plt.savefig('B738_Simple_Envelope.png', dpi=300, bbox_inches='tight')
    
    print(f"\n✅ 飞行包线已保存: B738_Simple_Envelope.png")
    plt.show()
    
    # 输出关键数据
    print_key_performance(aircraft, stall_speeds, max_speeds, valid_altitudes)
    
    return fig, ax

def calculate_flight_envelope(altitudes_ft, aircraft):
    """计算飞行包线"""
    
    stall_speeds = []
    max_speeds = []
    valid_altitudes = []
    
    print("🔧 计算飞行包线数据...")
    
    for alt_ft in altitudes_ft:
        alt_m = alt_ft * 0.3048
        
        # ISA标准大气
        if alt_m <= 11000:  # 对流层
            temp = 288.15 * (1 - 0.0065 * alt_m / 288.15)
            density_ratio = (temp / 288.15) ** 4.256
        else:  # 平流层
            temp = 216.65
            temp_11km = 288.15 * (1 - 0.0065 * 11000 / 288.15)
            density_ratio_11km = (temp_11km / 288.15) ** 4.256
            density_ratio = density_ratio_11km * np.exp(-(alt_m - 11000) / 6341.62)
        
        # 失速速度计算
        mass = aircraft['operating_weight']
        cl_max = 1.6  # 清洁构型最大升力系数
        
        v_stall_ms = np.sqrt(2 * mass * 9.81 / 
                           (1.225 * density_ratio * aircraft['wing_area'] * cl_max))
        v_stall_kt = v_stall_ms * 1.94384
        
        # 最大速度计算
        # 声速
        a = np.sqrt(1.4 * 287.053 * temp)
        
        # 马赫数限制
        v_mach_ms = aircraft['max_mach'] * a
        v_mach_kt = v_mach_ms * 1.94384
        
        # 结构限制
        if alt_ft < 28000:
            v_max_kt = 340  # VMO
        else:
            # 高空以马赫数限制为准，转换为等效指示空速
            v_max_kt = v_mach_ms * np.sqrt(density_ratio) * 1.94384
        
        # 数据验证
        if (v_stall_kt > 80 and v_stall_kt < 200 and
            v_max_kt > v_stall_kt + 50 and v_max_kt < 450 and
            alt_ft <= aircraft['service_ceiling']):
            
            stall_speeds.append(v_stall_kt)
            max_speeds.append(v_max_kt)
            valid_altitudes.append(alt_ft)
    
    print(f"✅ 计算完成: {len(valid_altitudes)} 个数据点")
    
    return np.array(stall_speeds), np.array(max_speeds), np.array(valid_altitudes)

def draw_mach_lines(ax, altitudes_ft):
    """绘制真正的速度限制线：VMO + MMO + 最小马赫数"""
    
    print("🔧 添加速度限制线...")
    
    # MMO限制：Mach 0.82（最大马赫数）
    mmo_speeds = []
    mmo_altitudes = []
    
    # 最小马赫数限制（高空低速限制）
    min_mach_speeds = []
    min_mach_altitudes = []
    
    # VMO/MMO转换高度
    transition_altitude = 28000  # ft
    
    for alt_ft in altitudes_ft:
        alt_m = alt_ft * 0.3048
        
        # 计算温度和密度
        if alt_m <= 11000:
            temp = 288.15 * (1 - 0.0065 * alt_m / 288.15)
            density_ratio = (temp / 288.15) ** 4.256
        else:
            temp = 216.65
            temp_11km = 288.15 * (1 - 0.0065 * 11000 / 288.15)
            density_ratio = (temp_11km / 288.15) ** 4.256 * np.exp(-(alt_m - 11000) / 6341.62)
        
        # 声速
        a = np.sqrt(1.4 * 287.053 * temp)
        
        # MMO: Mach 0.82的指示空速
        v_mmo_true = 0.82 * a
        v_mmo_indicated = v_mmo_true * np.sqrt(density_ratio) * 1.94384
        
        if 100 < v_mmo_indicated < 500 and alt_ft <= 41000:
            mmo_speeds.append(v_mmo_indicated)
            mmo_altitudes.append(alt_ft)
        
        # 最小马赫数限制（主要在高空）
        if alt_ft > 20000:  # 20,000 ft以上才有最小马赫数限制
            # 最小马赫数随高度增加：从Mach 0.3到Mach 0.5
            min_mach = 0.3 + (alt_ft - 20000) / (41000 - 20000) * 0.2
            min_mach = min(min_mach, 0.5)  # 不超过Mach 0.5
            
            v_min_mach_true = min_mach * a
            v_min_mach_indicated = v_min_mach_true * np.sqrt(density_ratio) * 1.94384
            
            if 80 < v_min_mach_indicated < 400:
                min_mach_speeds.append(v_min_mach_indicated)
                min_mach_altitudes.append(alt_ft)
    
    # 绘制MMO限制线
    if len(mmo_speeds) > 0:
        ax.plot(mmo_speeds, mmo_altitudes, 'darkred', linewidth=3,
               alpha=0.9, label='MMO (Mach 0.82 limit)')
    
    # 绘制最小马赫数限制线
    if len(min_mach_speeds) > 0:
        ax.plot(min_mach_speeds, min_mach_altitudes, 'orange', linewidth=3,
               alpha=0.9, label='Minimum Mach limit')
        
        # 填充马赫数操作区域
        if len(mmo_speeds) > 0:
            # 找到重叠的高度范围
            common_altitudes = []
            mmo_speeds_interp = []
            min_speeds_interp = []
            
            for alt in min_mach_altitudes:
                if alt <= max(mmo_altitudes):
                    # 插值获取MMO在该高度的速度
                    mmo_speed_at_alt = np.interp(alt, mmo_altitudes, mmo_speeds)
                    min_speed_at_alt = np.interp(alt, min_mach_altitudes, min_mach_speeds)
                    
                    if mmo_speed_at_alt > min_speed_at_alt:
                        common_altitudes.append(alt)
                        mmo_speeds_interp.append(mmo_speed_at_alt)
                        min_speeds_interp.append(min_speed_at_alt)
            
            if len(common_altitudes) > 0:
                ax.fill_betweenx(common_altitudes, min_speeds_interp, mmo_speeds_interp,
                               color='yellow', alpha=0.15, 
                               label='Mach operating envelope')
    
    # VMO限制线 - 340 kt 到转换高度
    ax.axvline(x=340, ymin=0, ymax=transition_altitude/43000, 
              color='purple', linestyle='-', linewidth=3, alpha=0.9, 
              label='VMO (340 kt limit)')
    
    # VMO/MMO转换点标注
    ax.plot(340, transition_altitude, 'ko', markersize=10, 
           markerfacecolor='yellow', markeredgecolor='black', linewidth=2,
           label='VMO/MMO transition')
    
    ax.annotate('VMO/MMO\nTransition\nFL280', 
               xy=(340, transition_altitude), 
               xytext=(380, transition_altitude-2000),
               arrowprops=dict(arrowstyle='->', color='black', lw=2),
               fontsize=11, ha='left', color='black', fontweight='bold',
               bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.8))
    
    print(f"✅ 添加了完整的速度限制：VMO + MMO + 最小马赫数")
    print(f"   - VMO: 340 kt (sea level to FL280)")
    print(f"   - MMO: Mach 0.82 (above FL280)")
    print(f"   - Min Mach: 0.3-0.5 (high altitude)")

def setup_dual_axis(ax):
    """设置双轴显示"""
    
    # 创建顶部马赫数轴
    ax2 = ax.twiny()
    
    # 使用FL350作为参考高度计算马赫数对应的指示空速
    ref_alt_m = 35000 * 0.3048
    ref_temp = 288.15 * (1 - 0.0065 * ref_alt_m / 288.15)
    ref_density_ratio = (ref_temp / 288.15) ** 4.256
    ref_sound_speed = np.sqrt(1.4 * 287.053 * ref_temp)
    
    # 马赫数刻度
    mach_ticks = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.82]
    indicated_speeds = []
    
    for mach in mach_ticks:
        true_speed = mach * ref_sound_speed
        indicated_speed = true_speed * np.sqrt(ref_density_ratio) * 1.94384
        indicated_speeds.append(indicated_speed)
    
    # 设置马赫数轴
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(indicated_speeds)
    ax2.set_xticklabels([f'{m:.2f}' for m in mach_ticks])
    ax2.set_xlabel('Mach Number (FL350 reference)', fontsize=12, 
                   fontweight='bold', color='navy')
    ax2.tick_params(axis='x', colors='navy')
    
    # 添加马赫数网格
    for speed in indicated_speeds:
        if ax.get_xlim()[0] <= speed <= ax.get_xlim()[1]:
            ax.axvline(x=speed, color='navy', linestyle=':', alpha=0.3, linewidth=1)
    
    print("✅ 添加了双轴显示")

def add_reference_lines(ax):
    """添加参考线"""
    
    # 重要巡航高度
    cruise_levels = [25000, 30000, 35000, 39000, 41000]
    for fl in cruise_levels:
        ax.axhline(y=fl, color='gray', linestyle=':', alpha=0.4, linewidth=1)
        ax.text(460, fl + 300, f'FL{fl//100}', fontsize=10, alpha=0.7,
               bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))

def add_performance_info(ax, aircraft, stall_speeds, max_speeds, valid_altitudes):
    """添加性能信息框"""
    
    info_text = f"""BOEING 737-800 PERFORMANCE

Configuration: 
• Weight: {aircraft['operating_weight']:,.0f} kg (80% MTOW)
• Wing Area: {aircraft['wing_area']:.1f} m²
• Max Mach: {aircraft['max_mach']}

Speed Limits:
• VMO: 340 kt (below FL280)
• MMO: Mach 0.82 (above FL280) 
• Service Ceiling: {aircraft['service_ceiling']:,} ft

Key Speeds (Sea Level):
• Stall: {stall_speeds[0]:.0f} kt
• Max: {max_speeds[0]:.0f} kt

Key Speeds (FL350):
• Stall: {stall_speeds[len(stall_speeds)//2]:.0f} kt
• Max: {max_speeds[len(max_speeds)//2]:.0f} kt

Wing Loading: {aircraft['operating_weight']/aircraft['wing_area']:.0f} kg/m²"""
    
    props = dict(boxstyle='round,pad=0.8', facecolor='lightcyan', alpha=0.9)
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', fontfamily='monospace', bbox=props)

def print_key_performance(aircraft, stall_speeds, max_speeds, valid_altitudes):
    """输出关键性能数据"""
    
    print(f"\n📊 Boeing 737-800 关键性能 (80% MTOW)")
    print("=" * 50)
    print(f"📈 海平面:")
    print(f"   失速速度: {stall_speeds[0]:.0f} kt")
    print(f"   最大速度: {max_speeds[0]:.0f} kt")
    
    # FL350性能
    fl350_idx = np.argmin(np.abs(valid_altitudes - 35000))
    print(f"📈 FL350 (35,000 ft):")
    print(f"   失速速度: {stall_speeds[fl350_idx]:.0f} kt")
    print(f"   最大速度: {max_speeds[fl350_idx]:.0f} kt")
    
    print(f"📈 实用升限:")
    print(f"   最大高度: {valid_altitudes[-1]:.0f} ft")
    print(f"   失速速度: {stall_speeds[-1]:.0f} kt")
    
    print(f"📈 飞行包线范围:")
    print(f"   速度范围: {stall_speeds.min():.0f} - {max_speeds.max():.0f} kt")
    print(f"   高度范围: 0 - {valid_altitudes.max():.0f} ft")

if __name__ == "__main__":
    print("🛩️ Boeing 737-800 简化飞行包线生成器")
    print("🎯 单一配置：80%全重，双轴显示")
    print("=" * 50)
    
    fig, ax = create_simple_flight_envelope()
    
    print(f"\n🎉 完成!")
    print(f"💡 简洁清晰的飞行包线图已生成")