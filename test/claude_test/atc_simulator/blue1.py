# bluesky_test.py
import bluesky as bs
import time
import matplotlib.pyplot as plt
import numpy as np

# 连接到 BlueSky
print("初始化 BlueSky...")

# 创建飞机并获取反馈
def create_and_track_aircraft():
    # 创建飞机
    bs.stack.stack("CRE KLM001,B737,52.3,4.8,0,250,35000")
    bs.stack.stack("CRE EZY123,A320,52.5,4.9,90,280,37000")
    
    print("已创建飞机:")
    print(f"飞机数量: {bs.traf.ntraf}")
    
    if bs.traf.ntraf > 0:
        for i in range(bs.traf.ntraf):
            print(f"  {bs.traf.id[i]}: {bs.traf.type[i]} at ({bs.traf.lat[i]:.2f}, {bs.traf.lon[i]:.2f})")
    
    # 开始模拟
    bs.stack.stack("OP")
    
    # 记录轨迹
    positions = {'KLM001': {'lat': [], 'lon': []}, 'EZY123': {'lat': [], 'lon': []}}
    
    print("\n开始模拟，记录轨迹...")
    for step in range(60):  # 运行60步
        bs.sim.step()
        
        # 记录位置
        for i in range(bs.traf.ntraf):
            callsign = bs.traf.id[i]
            if callsign in positions:
                positions[callsign]['lat'].append(bs.traf.lat[i])
                positions[callsign]['lon'].append(bs.traf.lon[i])
        
        if step % 10 == 0:
            print(f"步骤 {step}: 模拟时间 {bs.sim.simt:.1f}秒")
            for i in range(bs.traf.ntraf):
                print(f"  {bs.traf.id[i]}: ({bs.traf.lat[i]:.4f}, {bs.traf.lon[i]:.4f}) "
                      f"高度: {bs.traf.alt[i]:.0f}ft 速度: {bs.traf.tas[i]:.0f}kts")
    
    # 绘制轨迹图
    plt.figure(figsize=(10, 8))
    for callsign, pos in positions.items():
        if pos['lat']:
            plt.plot(pos['lon'], pos['lat'], 'o-', label=callsign, markersize=3)
            plt.plot(pos['lon'][0], pos['lat'][0], 's', markersize=8, label=f'{callsign} 起点')
    
    plt.xlabel('经度')
    plt.ylabel('纬度')
    plt.title('飞机轨迹图')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()
    
    print(f"\n模拟完成！总共运行了 {bs.sim.simt:.1f} 秒")

if __name__ == "__main__":
    create_and_track_aircraft()