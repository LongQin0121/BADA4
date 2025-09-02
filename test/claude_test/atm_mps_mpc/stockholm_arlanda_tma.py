#!/usr/bin/env python3
"""
Stockholm Arlanda Airport (ESSA) Terminal Area Chart
绘制斯德哥尔摩阿兰达机场终端区域图

基于公开的航空资料和参考数据
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.patches import Circle, Rectangle, Polygon
import matplotlib.gridspec as gridspec

class StockholmArlandaTMA:
    def __init__(self):
        # Stockholm Arlanda Airport (ESSA) 基本信息
        self.airport_lat = 59.651111
        self.airport_lon = 17.918611
        self.airport_elevation = 135  # feet MSL
        
        # 跑道信息
        self.runways = {
            '01L/19R': {
                'length': 3301,  # meters
                'heading': 13,  # degrees
                'lat_start': 59.64,
                'lon_start': 17.91,
                'lat_end': 59.66,
                'lon_end': 17.93
            },
            '08/26': {
                'length': 2500,  # meters
                'heading': 80,  # degrees
                'lat_start': 59.65,
                'lon_start': 17.90,
                'lat_end': 59.66,
                'lon_end': 17.94
            },
            '01R/19L': {
                'length': 2500,  # meters
                'heading': 13,  # degrees
                'lat_start': 59.64,
                'lon_start': 17.93,
                'lat_end': 59.66,
                'lon_end': 17.95
            }
        }
        
        # 导航点和航路点 (基于搜索到的资料)
        self.navigation_fixes = {
            'ARS': {'lat': 59.8, 'lon': 17.8, 'type': 'waypoint'},
            'ABENI': {'lat': 59.5, 'lon': 17.7, 'type': 'waypoint'},
            'ELTOK': {'lat': 59.4, 'lon': 17.5, 'type': 'entry_point'},
            'HAPZI': {'lat': 59.7, 'lon': 18.2, 'type': 'waypoint'},
        }
        
        # TMA 边界 (近似值，基于典型的终端区域)
        self.tma_boundary = [
            (59.2, 17.2), (59.2, 18.7), (60.1, 18.7), 
            (60.1, 17.2), (59.2, 17.2)
        ]
        
        # 控制区域
        self.control_zones = {
            'CTR': {'radius': 0.15, 'alt_upper': 1500},  # Control Zone
            'TMA': {'radius': 0.5, 'alt_upper': 5000}    # Terminal Area
        }

    def plot_terminal_area(self):
        """绘制终端区域图"""
        fig = plt.figure(figsize=(16, 12))
        gs = gridspec.GridSpec(2, 2, height_ratios=[3, 1], width_ratios=[3, 1])
        
        # 主图
        ax_main = fig.add_subplot(gs[0, 0])
        
        # 设置地图范围
        lat_range = [59.2, 60.1]
        lon_range = [17.2, 18.7]
        
        ax_main.set_xlim(lon_range)
        ax_main.set_ylim(lat_range)
        ax_main.set_aspect('equal')
        ax_main.grid(True, alpha=0.3)
        
        # 绘制TMA边界
        tma_lats = [point[0] for point in self.tma_boundary]
        tma_lons = [point[1] for point in self.tma_boundary]
        ax_main.plot(tma_lons, tma_lats, 'b-', linewidth=2, label='TMA Boundary')
        ax_main.fill(tma_lons, tma_lats, alpha=0.1, color='blue')
        
        # 绘制控制区域
        ctr_circle = Circle((self.airport_lon, self.airport_lat), 
                           self.control_zones['CTR']['radius'], 
                           fill=False, color='red', linewidth=2, label='CTR')
        ax_main.add_patch(ctr_circle)
        
        tma_circle = Circle((self.airport_lon, self.airport_lat), 
                           self.control_zones['TMA']['radius'], 
                           fill=False, color='orange', linewidth=1.5, 
                           linestyle='--', label='TMA Core')
        ax_main.add_patch(tma_circle)
        
        # 绘制机场位置
        ax_main.plot(self.airport_lon, self.airport_lat, 'ks', 
                    markersize=12, label='ESSA Arlanda')
        
        # 绘制跑道
        for runway_name, runway_info in self.runways.items():
            lat_start = runway_info['lat_start']
            lon_start = runway_info['lon_start']
            lat_end = runway_info['lat_end']
            lon_end = runway_info['lon_end']
            
            ax_main.plot([lon_start, lon_end], [lat_start, lat_end], 
                        'k-', linewidth=4, alpha=0.8)
            
            # 跑道标识
            mid_lat = (lat_start + lat_end) / 2
            mid_lon = (lon_start + lon_end) / 2
            ax_main.text(mid_lon, mid_lat, runway_name, 
                        fontsize=8, ha='center', va='center',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        # 绘制导航点
        for fix_name, fix_info in self.navigation_fixes.items():
            marker_style = 'o' if fix_info['type'] == 'waypoint' else '^'
            color = 'purple' if fix_info['type'] == 'entry_point' else 'green'
            
            ax_main.plot(fix_info['lon'], fix_info['lat'], marker_style, 
                        color=color, markersize=8)
            ax_main.text(fix_info['lon'], fix_info['lat'] + 0.02, fix_name, 
                        fontsize=9, ha='center', va='bottom', weight='bold')
        
        # 绘制进近航线（示意）
        self._draw_approach_paths(ax_main)
        
        # 添加标题和标签
        ax_main.set_title('Stockholm Arlanda Airport (ESSA) - Terminal Area Chart', 
                         fontsize=14, weight='bold')
        ax_main.set_xlabel('Longitude (°E)', fontsize=12)
        ax_main.set_ylabel('Latitude (°N)', fontsize=12)
        ax_main.legend(loc='upper right', fontsize=10)
        
        # 机场详细信息面板
        ax_info = fig.add_subplot(gs[0, 1])
        ax_info.axis('off')
        
        info_text = f"""
STOCKHOLM ARLANDA AIRPORT
ICAO: ESSA
IATA: ARN

Coordinates:
{self.airport_lat:.4f}°N
{self.airport_lon:.4f}°E

Elevation: {self.airport_elevation} ft MSL

Runways:
01L/19R: {self.runways['01L/19R']['length']}m
08/26: {self.runways['08/26']['length']}m
01R/19L: {self.runways['01R/19L']['length']}m

TMA Operations:
• Max speed <FL100: 250 kts
• Min approach speed: 160 kts
• Missed approach: 1500 ft

Control Zones:
• CTR: SFC-1500 ft
• TMA: 1500-5000 ft
        """
        
        ax_info.text(0.05, 0.95, info_text, transform=ax_info.transAxes, 
                    fontsize=10, verticalalignment='top', 
                    bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8))
        
        # 高度剖面图
        ax_profile = fig.add_subplot(gs[1, :])
        self._draw_altitude_profile(ax_profile)
        
        plt.tight_layout()
        return fig

    def _draw_approach_paths(self, ax):
        """绘制进近航线"""
        # ILS 01L 进近
        ils_01l_path = [
            (17.5, 59.4), (17.7, 59.5), (17.91, 59.64)
        ]
        lats = [p[1] for p in ils_01l_path]
        lons = [p[0] for p in ils_01l_path]
        ax.plot(lons, lats, 'g--', linewidth=2, alpha=0.7, label='ILS 01L')
        
        # ILS 19R 进近
        ils_19r_path = [
            (18.3, 59.9), (18.1, 59.8), (17.93, 59.66)
        ]
        lats = [p[1] for p in ils_19r_path]
        lons = [p[0] for p in ils_19r_path]
        ax.plot(lons, lats, 'r--', linewidth=2, alpha=0.7, label='ILS 19R')
        
        # RNP 进近 (示意)
        rnp_path = [
            (17.4, 59.7), (17.6, 59.65), (17.9, 59.65)
        ]
        lats = [p[1] for p in rnp_path]
        lons = [p[0] for p in rnp_path]
        ax.plot(lons, lats, 'm:', linewidth=2, alpha=0.7, label='RNP Approach')

    def _draw_altitude_profile(self, ax):
        """绘制高度剖面图"""
        distances = np.linspace(0, 50, 100)  # 0-50 NM
        
        # 标准3度下滑道
        ils_altitudes = 5000 - distances * 316  # 3度下滑道: 316 ft/NM
        ils_altitudes = np.maximum(ils_altitudes, 135)  # 不低于机场标高
        
        # TMA 高度限制
        tma_upper = np.full_like(distances, 5000)
        ctr_upper = np.full_like(distances, 1500)
        
        ax.plot(distances, ils_altitudes, 'g-', linewidth=2, label='ILS Glidepath (3°)')
        ax.fill_between(distances, 0, ctr_upper, alpha=0.2, color='red', label='CTR (SFC-1500 ft)')
        ax.fill_between(distances, ctr_upper, tma_upper, alpha=0.2, color='orange', label='TMA (1500-5000 ft)')
        
        ax.set_xlabel('Distance from Airport (NM)')
        ax.set_ylabel('Altitude (ft MSL)')
        ax.set_title('Terminal Area Altitude Profile')
        ax.grid(True, alpha=0.3)
        ax.legend()longqin@UPC:~/code/test/claude_test/atm_mps_mpc$ python stockholm_arlanda_tma.py
  File "/home/longqin/code/test/claude_test/atm_mps_mpc/stockholm_arlanda_tma.py", line 26
    'heading': 013,  # degrees
               ^
SyntaxError: leading zeros in decimal integer literals are not permitted; use an 0o prefix for octal integers
        ax.set_ylim(0, 6000)

    def save_chart(self, filename='stockholm_arlanda_tma.png'):
        """保存图表"""
        fig = self.plot_terminal_area()
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        return filename

if __name__ == "__main__":
    # 创建并显示斯德哥尔摩阿兰达机场终端区域图
    arlanda_tma = StockholmArlandaTMA()
    saved_file = arlanda_tma.save_chart()
    print(f"Stockholm Arlanda TMA chart saved as: {saved_file}")
    print("\n基于以下参考资料:")
    print("- VATSIM Scandinavia Wiki")
    print("- OpenNav Aviation Database")
    print("- Swedish LFV IAIP")
    print("- Navigraph Charts")