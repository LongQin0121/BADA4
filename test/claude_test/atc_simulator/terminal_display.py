#!/usr/bin/env python3
"""
Stockholm ATC Simulator - Terminal Display
终端实时显示和绘图工具
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from matplotlib.patches import Circle, Rectangle
import requests
import time
import threading
import os
import sys
from datetime import datetime

class TerminalDisplay:
    """终端显示类"""
    
    def __init__(self, api_url="http://localhost:8080"):
        self.api_url = api_url
        self.fig, (self.ax_map, self.ax_stats) = plt.subplots(1, 2, figsize=(16, 8))
        self.aircraft_plots = {}
        self.waypoint_plots = {}
        self.setup_plot()
        
        # 数据存储
        self.current_data = None
        self.stats_history = {
            'time': [],
            'active_aircraft': [],
            'total_aircraft': []
        }
        
    def setup_plot(self):
        """设置绘图区域"""
        # 地图区域设置
        self.ax_map.set_xlim(17.2, 18.7)
        self.ax_map.set_ylim(59.2, 60.1)
        self.ax_map.set_aspect('equal')
        self.ax_map.grid(True, alpha=0.3)
        self.ax_map.set_title('Stockholm TMA - Real-time Aircraft Tracking', fontsize=14, fontweight='bold')
        self.ax_map.set_xlabel('Longitude (°E)')
        self.ax_map.set_ylabel('Latitude (°N)')
        
        # 添加机场标记
        airport_lat, airport_lon = 59.651111, 17.918611
        self.ax_map.plot(airport_lon, airport_lat, 'ks', markersize=12, label='ESSA Arlanda')
        
        # TMA边界
        tma_boundary = [(59.2, 17.2), (59.2, 18.7), (60.1, 18.7), (60.1, 17.2), (59.2, 17.2)]
        tma_lats = [point[0] for point in tma_boundary]
        tma_lons = [point[1] for point in tma_boundary]
        self.ax_map.plot(tma_lons, tma_lats, 'b-', linewidth=2, alpha=0.5, label='TMA Boundary')
        
        # 控制区域
        ctr_circle = Circle((airport_lon, airport_lat), 0.15, fill=False, color='red', linewidth=2, label='CTR')
        self.ax_map.add_patch(ctr_circle)
        
        self.ax_map.legend(loc='upper right')
        
        # 统计图表设置
        self.ax_stats.set_title('Statistics', fontsize=14, fontweight='bold')
        self.ax_stats.set_xlabel('Time (seconds)')
        self.ax_stats.set_ylabel('Count')
        self.ax_stats.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
    def fetch_simulation_data(self):
        """获取模拟数据"""
        try:
            response = requests.get(f"{self.api_url}/api/simulation/state", timeout=2)
            if response.status_code == 200:
                return response.json()
        except requests.RequestException as e:
            print(f"Failed to fetch data: {e}")
        return None
    
    def update_plot(self, frame):
        """更新绘图"""
        # 获取最新数据
        data = self.fetch_simulation_data()
        if not data:
            return
        
        self.current_data = data
        
        # 清除旧的航空器标记
        for plot in self.aircraft_plots.values():
            plot.remove()
        self.aircraft_plots.clear()
        
        # 绘制航空器
        for callsign, aircraft in data['aircraft'].items():
            # 根据高度设置颜色
            if aircraft['altitude'] < 5000:
                color = 'red'    # 低空
            elif aircraft['altitude'] < 15000:
                color = 'orange' # 中空
            else:
                color = 'green'  # 高空
            
            # 绘制航空器位置
            plot = self.ax_map.plot(aircraft['lon'], aircraft['lat'], 'o', 
                                  color=color, markersize=8, alpha=0.8)[0]
            self.aircraft_plots[callsign] = plot
            
            # 添加标签
            self.ax_map.annotate(f"{callsign}\n{aircraft['altitude']}ft", 
                               (aircraft['lon'], aircraft['lat']),
                               xytext=(5, 5), textcoords='offset points',
                               fontsize=8, alpha=0.8,
                               bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))
        
        # 绘制航路点（只绘制一次）
        if not self.waypoint_plots and 'waypoints' in data:
            for name, waypoint in data['waypoints'].items():
                color = 'purple' if waypoint['type'] == 'entry_point' else 'blue'
                plot = self.ax_map.plot(waypoint['lon'], waypoint['lat'], '^', 
                                      color=color, markersize=6, alpha=0.6)[0]
                self.waypoint_plots[name] = plot
                
                self.ax_map.annotate(name, (waypoint['lon'], waypoint['lat']),
                                   xytext=(3, 3), textcoords='offset points',
                                   fontsize=7, alpha=0.7)
        
        # 更新统计数据
        current_time = len(self.stats_history['time'])
        self.stats_history['time'].append(current_time)
        self.stats_history['active_aircraft'].append(data['stats']['active_aircraft'])
        self.stats_history['total_aircraft'].append(data['stats']['total_aircraft'])
        
        # 保持最近100个数据点
        if len(self.stats_history['time']) > 100:
            for key in self.stats_history:
                self.stats_history[key] = self.stats_history[key][-100:]
        
        # 更新统计图表
        self.ax_stats.clear()
        self.ax_stats.plot(self.stats_history['time'], self.stats_history['active_aircraft'], 
                         'b-', label='Active Aircraft', linewidth=2)
        self.ax_stats.plot(self.stats_history['time'], self.stats_history['total_aircraft'], 
                         'g--', label='Total Aircraft', linewidth=2)
        self.ax_stats.set_title('Statistics', fontsize=14, fontweight='bold')
        self.ax_stats.set_xlabel('Time (seconds)')
        self.ax_stats.set_ylabel('Count')
        self.ax_stats.grid(True, alpha=0.3)
        self.ax_stats.legend()
        
        # 更新标题显示实时信息
        if data:
            self.ax_map.set_title(f'Stockholm TMA - Active: {data["stats"]["active_aircraft"]} | ' +
                                f'Total: {data["stats"]["total_aircraft"]} | ' +
                                f'Running: {"Yes" if data["is_running"] else "No"}',
                                fontsize=12, fontweight='bold')
    
    def start_display(self):
        """启动显示"""
        print("=== Stockholm ATC Simulator - Terminal Display ===")
        print(f"Connecting to: {self.api_url}")
        print("Features:")
        print("- Real-time aircraft tracking")
        print("- TMA boundary visualization")
        print("- Statistics monitoring")
        print("- Color-coded altitude display")
        print("\nPress Ctrl+C to stop")
        
        # 创建动画
        self.anim = animation.FuncAnimation(self.fig, self.update_plot, 
                                          interval=1000, blit=False, cache_frame_data=False)
        
        plt.show()

class TerminalTextDisplay:
    """终端文本显示类"""
    
    def __init__(self, api_url="http://localhost:8080"):
        self.api_url = api_url
        self.running = True
        
    def clear_screen(self):
        """清屏"""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def fetch_data(self):
        """获取数据"""
        try:
            response = requests.get(f"{self.api_url}/api/simulation/state", timeout=2)
            if response.status_code == 200:
                return response.json()
        except requests.RequestException:
            pass
        return None
    
    def display_ascii_map(self, data):
        """显示ASCII地图"""
        if not data or 'aircraft' not in data:
            return
        
        print("📍 Stockholm TMA ASCII Map")
        print("=" * 60)
        
        # 简化的ASCII地图
        map_grid = [['.' for _ in range(40)] for _ in range(20)]
        
        # 标记机场
        airport_x, airport_y = 20, 10
        map_grid[airport_y][airport_x] = '🛩'
        
        # 标记航空器
        for callsign, aircraft in data['aircraft'].items():
            # 简单的坐标转换
            x = int((aircraft['lon'] - 17.2) / (18.7 - 17.2) * 39)
            y = int((60.1 - aircraft['lat']) / (60.1 - 59.2) * 19)
            
            if 0 <= x < 40 and 0 <= y < 20:
                map_grid[y][x] = '✈'
        
        # 打印地图
        for row in map_grid:
            print(''.join(row))
        
        print("Legend: 🛩 = Airport, ✈ = Aircraft")
    
    def display_aircraft_table(self, data):
        """显示航空器表格"""
        if not data or 'aircraft' not in data:
            return
        
        print("\n✈️  Active Aircraft")
        print("=" * 80)
        print(f"{'Callsign':<10} {'Type':<6} {'Alt(ft)':<8} {'Spd(kts)':<9} {'Hdg':<5} {'Waypoint':<10} {'Procedure'}")
        print("-" * 80)
        
        for callsign, aircraft in data['aircraft'].items():
            procedure = aircraft.get('sid', aircraft.get('star', ''))
            waypoint = aircraft.get('current_waypoint', '')
            
            print(f"{callsign:<10} {aircraft['aircraft_type']:<6} {aircraft['altitude']:<8} " +
                  f"{aircraft['speed']:<9} {aircraft['heading']:<5} {waypoint:<10} {procedure}")
    
    def display_stats(self, data):
        """显示统计信息"""
        if not data:
            return
        
        stats = data.get('stats', {})
        print(f"\n📊 Statistics")
        print("=" * 30)
        print(f"Active Aircraft: {stats.get('active_aircraft', 0)}")
        print(f"Total Aircraft:  {stats.get('total_aircraft', 0)}")
        print(f"Departures:      {stats.get('departed', 0)}")
        print(f"Arrivals:        {stats.get('arrived', 0)}")
        print(f"Simulation:      {'Running' if data.get('is_running') else 'Stopped'}")
        print(f"Time:            {data.get('simulation_time', 'N/A')}")
    
    def run(self):
        """运行文本显示"""
        print("=== Stockholm ATC Simulator - Terminal Text Display ===")
        print(f"Connecting to: {self.api_url}")
        print("Press Ctrl+C to stop\n")
        
        try:
            while self.running:
                self.clear_screen()
                
                # 显示标题和时间
                print(f"🎮 Stockholm ATC Simulator - {datetime.now().strftime('%H:%M:%S')}")
                print("=" * 60)
                
                # 获取并显示数据
                data = self.fetch_data()
                if data:
                    self.display_stats(data)
                    self.display_aircraft_table(data)
                    self.display_ascii_map(data)
                else:
                    print("❌ Cannot connect to simulator")
                
                print(f"\n🌐 Web Interface: {self.api_url}")
                print("Press Ctrl+C to stop")
                
                time.sleep(2)
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            self.running = False

def main():
    """主函数"""
    if len(sys.argv) > 1:
        if sys.argv[1] == '--text':
            # 文本模式
            display = TerminalTextDisplay()
            display.run()
        elif sys.argv[1] == '--help':
            print("Stockholm ATC Simulator - Terminal Display")
            print("Usage:")
            print("  python terminal_display.py          # Graphical mode")
            print("  python terminal_display.py --text   # Text mode")
            print("  python terminal_display.py --help   # Show this help")
        else:
            print("Unknown option. Use --help for usage information.")
    else:
        # 图形模式
        try:
            display = TerminalDisplay()
            display.start_display()
        except ImportError:
            print("Matplotlib not available. Falling back to text mode...")
            display = TerminalTextDisplay()
            display.run()

if __name__ == "__main__":
    main()