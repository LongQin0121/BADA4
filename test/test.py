# import torch

# print("PyTorch version:", torch.__version__)
# print("CUDA available:", torch.cuda.is_available())
# print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU detected")

# print("=== GPU详细信息 ===")
# print(f"CUDA设备数量: {torch.cuda.device_count()}")
# print(f"当前设备: {torch.cuda.current_device()}")
# print(f"设备名称: {torch.cuda.get_device_name(0)}")

# props = torch.cuda.get_device_properties(0)
# print(f"总内存: {props.total_memory/1024**3:.1f}GB")
# print(f"多处理器数量: {props.multi_processor_count}")
# print(f"CUDA能力版本: {props.major}.{props.minor}")
import requests
import json
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon
import xml.etree.ElementTree as ET
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

class BarcelonaAirportVisualizer:
    def __init__(self):
        # 巴塞罗那机场 (LEBL) 基本信息
        self.airport_code = "LEBL"
        self.airport_name = "Barcelona El Prat"
        self.airport_coord = (41.2971, 2.0785)  # 纬度, 经度
        
        # 巴塞罗那机场跑道信息
        self.runways = {
            "06R/24L": {
                "start": (41.2945, 2.0695),
                "end": (41.2997, 2.0875),
                "heading": 64  # 修正后的航向
            },
            "06L/24R": {
                "start": (41.2890, 2.0740),
                "end": (41.2942, 2.0920),
                "heading": 64
            },
            "02/20": {
                "start": (41.2915, 2.0725),
                "end": (41.3025, 2.0835),
                "heading": 20
            }
        }
        
        # 标准进离场程序 (SID/STAR) 模拟数据
        self.sids = {
            "LORES1A": [
                (41.2971, 2.0785),  # 机场
                (41.3200, 2.1500),  # 转向点1
                (41.3800, 2.2500),  # 转向点2
                (41.4500, 2.3500)   # 离场点
            ],
            "GIRONA1B": [
                (41.2971, 2.0785),
                (41.3500, 2.1200),
                (41.4200, 2.1800),
                (41.5000, 2.2500)
            ],
            "COSTA1C": [
                (41.2971, 2.0785),
                (41.2800, 2.1500),
                (41.2500, 2.2200),
                (41.2200, 2.3000)
            ]
        }
        
        self.stars = {
            "LORES1A": [
                (41.4500, 2.3500),
                (41.3800, 2.2500),
                (41.3200, 2.1500),
                (41.2971, 2.0785)
            ],
            "SITGES1B": [
                (41.1500, 2.4000),
                (41.2000, 2.3000),
                (41.2500, 2.2000),
                (41.2971, 2.0785)
            ],
            "MARESME1C": [
                (41.5500, 2.0000),
                (41.4500, 2.0300),
                (41.3500, 2.0500),
                (41.2971, 2.0785)
            ]
        }
        
        # 扇区边界模拟数据 (基于西班牙空域结构)
        self.sectors = {
            "BCN_APP": [  # 巴塞罗那进近扇区
                (41.1500, 1.8000),
                (41.1500, 2.4000),
                (41.4500, 2.4000),
                (41.4500, 1.8000)
            ],
            "BCN_TWR": [  # 巴塞罗那塔台扇区
                (41.2700, 2.0400),
                (41.2700, 2.1200),
                (41.3200, 2.1200),
                (41.3200, 2.0400)
            ],
            "LECM_CTR": [  # 巴塞罗那管制中心扇区
                (41.0000, 1.5000),
                (41.0000, 2.8000),
                (41.7000, 2.8000),
                (41.7000, 1.5000)
            ]
        }
        
        # 导航设施
        self.navaids = {
            "BCN_VOR": (41.2971, 2.0785, "VOR"),
            "SABADELL_NDB": (41.5209, 2.1050, "NDB"),
            "GIRONA_VOR": (41.9011, 2.7606, "VOR"),
            "REUS_VOR": (41.1470, 1.1672, "VOR")
        }

    def fetch_openaip_data(self, country_code="ES"):
        """
        尝试从OpenAIP获取实际数据
        注意：这需要API密钥和正确的端点
        """
        try:
            # OpenAIP API示例 (需要注册获取API密钥)
            base_url = "https://api.openaip.net/v1"
            # 这里使用模拟数据，实际使用时需要替换为真实的API调用
            print("注意：使用模拟数据。实际使用时需要从OpenAIP获取真实数据。")
            return None
        except Exception as e:
            print(f"无法获取OpenAIP数据: {e}")
            return None

    def plot_airport_chart(self, figsize=(15, 12)):
        """绘制机场图表"""
        fig, ax = plt.subplots(figsize=figsize)
        
        # 设置地图范围
        lat_min, lat_max = 41.0, 41.6
        lon_min, lon_max = 1.5, 2.8
        ax.set_xlim(lon_min, lon_max)
        ax.set_ylim(lat_min, lat_max)
        
        # 绘制扇区边界
        self._plot_sectors(ax)
        
        # 绘制机场和跑道
        self._plot_airport_runways(ax)
        
        # 绘制SID航线
        self._plot_sids(ax)
        
        # 绘制STAR航线
        self._plot_stars(ax)
        
        # 绘制导航设施
        self._plot_navaids(ax)
        
        # 添加标题和标签
        ax.set_title(f'{self.airport_name} ({self.airport_code}) - 进离场航线与扇区边界', 
                    fontsize=16, fontweight='bold')
        ax.set_xlabel('经度 (°E)', fontsize=12)
        ax.set_ylabel('纬度 (°N)', fontsize=12)
        
        # 添加网格
        ax.grid(True, alpha=0.3)
        
        # 添加图例
        self._add_legend(ax)
        
        # 设置长宽比
        ax.set_aspect('equal', adjustable='box')
        
        plt.tight_layout()
        return fig, ax

    def _plot_sectors(self, ax):
        """绘制扇区边界"""
        colors = {'BCN_APP': 'blue', 'BCN_TWR': 'red', 'LECM_CTR': 'green'}
        alphas = {'BCN_APP': 0.2, 'BCN_TWR': 0.3, 'LECM_CTR': 0.1}
        
        for sector_name, coords in self.sectors.items():
            polygon = Polygon(coords, alpha=alphas[sector_name], 
                            facecolor=colors[sector_name], 
                            edgecolor=colors[sector_name], linewidth=2)
            ax.add_patch(polygon)
            
            # 添加扇区标签
            center_lat = sum(coord[0] for coord in coords) / len(coords)
            center_lon = sum(coord[1] for coord in coords) / len(coords)
            ax.text(center_lon, center_lat, sector_name, 
                   fontsize=10, fontweight='bold', 
                   ha='center', va='center',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

    def _plot_airport_runways(self, ax):
        """绘制机场和跑道"""
        # 机场位置
        ax.plot(self.airport_coord[1], self.airport_coord[0], 
               'ko', markersize=15, label='机场')
        ax.text(self.airport_coord[1] + 0.02, self.airport_coord[0] + 0.02,
               f'{self.airport_name}\n({self.airport_code})',
               fontsize=12, fontweight='bold',
               bbox=dict(boxstyle="round,pad=0.5", facecolor='yellow', alpha=0.8))
        
        # 跑道
        for runway_name, runway_data in self.runways.items():
            start = runway_data['start']
            end = runway_data['end']
            ax.plot([start[1], end[1]], [start[0], end[0]], 
                   'k-', linewidth=6, alpha=0.8, label='跑道' if runway_name == list(self.runways.keys())[0] else "")
            
            # 跑道标签
            mid_lat = (start[0] + end[0]) / 2
            mid_lon = (start[1] + end[1]) / 2
            ax.text(mid_lon, mid_lat, runway_name, 
                   fontsize=8, ha='center', va='center',
                   bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.9))

    def _plot_sids(self, ax):
        """绘制标准离场程序 (SID)"""
        colors = ['red', 'orange', 'purple']
        for i, (sid_name, waypoints) in enumerate(self.sids.items()):
            lats = [wp[0] for wp in waypoints]
            lons = [wp[1] for wp in waypoints]
            
            ax.plot(lons, lats, color=colors[i % len(colors)], 
                   linewidth=3, linestyle='-', alpha=0.8,
                   label=f'SID: {sid_name}')
            
            # 添加箭头显示方向
            for j in range(len(waypoints) - 1):
                ax.annotate('', xy=(lons[j+1], lats[j+1]), 
                           xytext=(lons[j], lats[j]),
                           arrowprops=dict(arrowstyle='->', 
                                         color=colors[i % len(colors)], 
                                         lw=2))
            
            # 标记最后一个航点
            ax.plot(lons[-1], lats[-1], 'o', color=colors[i % len(colors)], 
                   markersize=8)
            ax.text(lons[-1] + 0.01, lats[-1] + 0.01, sid_name,
                   fontsize=9, color=colors[i % len(colors)], fontweight='bold')

    def _plot_stars(self, ax):
        """绘制标准到达程序 (STAR)"""
        colors = ['blue', 'cyan', 'navy']
        for i, (star_name, waypoints) in enumerate(self.stars.items()):
            lats = [wp[0] for wp in waypoints]
            lons = [wp[1] for wp in waypoints]
            
            ax.plot(lons, lats, color=colors[i % len(colors)], 
                   linewidth=2, linestyle='--', alpha=0.8,
                   label=f'STAR: {star_name}')
            
            # 添加箭头显示方向
            for j in range(len(waypoints) - 1):
                ax.annotate('', xy=(lons[j+1], lats[j+1]), 
                           xytext=(lons[j], lats[j]),
                           arrowprops=dict(arrowstyle='->', 
                                         color=colors[i % len(colors)], 
                                         lw=1.5))
            
            # 标记第一个航点
            ax.plot(lons[0], lats[0], 's', color=colors[i % len(colors)], 
                   markersize=8)

    def _plot_navaids(self, ax):
        """绘制导航设施"""
        navaid_markers = {'VOR': '^', 'NDB': 'D', 'DME': 'h'}
        
        for navaid_name, (lat, lon, navaid_type) in self.navaids.items():
            marker = navaid_markers.get(navaid_type, 'o')
            ax.plot(lon, lat, marker, markersize=10, 
                   color='green', markeredgecolor='black', 
                   markeredgewidth=1, alpha=0.8,
                   label=navaid_type if navaid_name == list(self.navaids.keys())[0] else "")
            
            ax.text(lon + 0.02, lat + 0.02, 
                   f'{navaid_name}\n({navaid_type})',
                   fontsize=8, ha='left', va='bottom',
                   bbox=dict(boxstyle="round,pad=0.2", facecolor='lightgreen', alpha=0.7))

    def _add_legend(self, ax):
        """添加图例"""
        # 获取所有的图例元素
        handles, labels = ax.get_legend_handles_labels()
        
        # 创建自定义图例元素
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D
        
        custom_elements = [
            Patch(facecolor='blue', alpha=0.2, label='进近扇区'),
            Patch(facecolor='red', alpha=0.3, label='塔台扇区'),
            Patch(facecolor='green', alpha=0.1, label='管制扇区'),
            Line2D([0], [0], color='red', linewidth=3, label='离场航线 (SID)'),
            Line2D([0], [0], color='blue', linewidth=2, linestyle='--', label='到达航线 (STAR)'),
            Line2D([0], [0], color='black', linewidth=6, label='跑道'),
            Line2D([0], [0], marker='^', color='green', linewidth=0, markersize=8, label='导航设施')
        ]
        
        ax.legend(handles=custom_elements, loc='upper left', bbox_to_anchor=(0, 1))

    def generate_report(self):
        """生成机场信息报告"""
        report = f"""
=== {self.airport_name} ({self.airport_code}) 机场信息报告 ===

机场位置: {self.airport_coord[0]:.4f}°N, {self.airport_coord[1]:.4f}°E

跑道信息:
"""
        for runway, data in self.runways.items():
            report += f"  - {runway}: 航向 {data['heading']}°\n"
        
        report += f"\n标准离场程序 (SID): {len(self.sids)} 条"
        for sid in self.sids.keys():
            report += f"\n  - {sid}"
        
        report += f"\n\n标准到达程序 (STAR): {len(self.stars)} 条"
        for star in self.stars.keys():
            report += f"\n  - {star}"
        
        report += f"\n\n扇区边界: {len(self.sectors)} 个"
        for sector in self.sectors.keys():
            report += f"\n  - {sector}"
        
        report += f"\n\n导航设施: {len(self.navaids)} 个"
        for navaid, (lat, lon, nav_type) in self.navaids.items():
            report += f"\n  - {navaid} ({nav_type}): {lat:.4f}°N, {lon:.4f}°E"
        
        report += "\n\n注意：本数据为演示用途，实际飞行请使用官方最新航行资料。"
        
        return report

def main():
    """主函数"""
    print("巴塞罗那机场 (LEBL) 进离场航线与扇区边界可视化")
    print("=" * 50)
    
    # 创建可视化对象
    visualizer = BarcelonaAirportVisualizer()
    
    # 尝试获取OpenAIP数据
    visualizer.fetch_openaip_data()
    
    # 生成图表
    fig, ax = visualizer.plot_airport_chart()
    
    # 显示图表
    plt.show()
    
    # 生成报告
    report = visualizer.generate_report()
    print(report)

if __name__ == "__main__":
    main()