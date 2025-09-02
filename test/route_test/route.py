import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Circle
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class NavAidsVisualizer:
    def __init__(self, csv_file_path):
        """初始化导航设施可视化器"""
        self.csv_file_path = csv_file_path
        self.df = None
        self.load_data()
        
    def load_data(self):
        """加载CSV数据"""
        try:
            print(f"正在读取数据文件: {self.csv_file_path}")
            self.df = pd.read_csv(self.csv_file_path)
            print(f"成功加载数据: {len(self.df)} 行, {len(self.df.columns)} 列")
            print("\n数据列名:")
            for i, col in enumerate(self.df.columns):
                print(f"  {i+1:2d}. {col}")
            
            # 显示数据基本信息
            print(f"\n数据概览:")
            print(f"  总记录数: {len(self.df):,}")
            print(f"  国家数量: {self.df['iso_country'].nunique()}")
            print(f"  导航设施类型: {sorted(self.df['type'].unique())}")
            
        except Exception as e:
            print(f"读取数据文件时出错: {e}")
            return None
    
    def create_comprehensive_dashboard(self):
        """创建综合仪表板"""
        if self.df is None:
            print("数据未加载，无法创建可视化")
            return
            
        # 创建大图表
        fig = plt.figure(figsize=(20, 16))
        
        # 1. 世界地图分布 (主图)
        ax1 = plt.subplot(2, 3, (1, 2))
        self.plot_world_distribution(ax1)
        
        # 2. 类型分布饼图
        ax2 = plt.subplot(2, 3, 3)
        self.plot_type_distribution(ax2)
        
        # 3. 国家分布条形图
        ax3 = plt.subplot(2, 3, 4)
        self.plot_country_distribution(ax3)
        
        # 4. 频率分布直方图
        ax4 = plt.subplot(2, 3, 5)
        self.plot_frequency_distribution(ax4)
        
        # 5. 海拔分布
        ax5 = plt.subplot(2, 3, 6)
        self.plot_elevation_distribution(ax5)
        
        plt.tight_layout()
        plt.suptitle('航空导航设施全球分布分析仪表板', fontsize=20, fontweight='bold', y=0.98)
        return fig
    
    def plot_world_distribution(self, ax):
        """绘制世界地图分布"""
        # 过滤有效的经纬度数据
        valid_coords = self.df.dropna(subset=['latitude_deg', 'longitude_deg'])
        
        # 不同导航设施类型使用不同颜色和标记
        type_styles = {
            'VOR': {'color': 'red', 'marker': '^', 'size': 30, 'alpha': 0.7},
            'NDB': {'color': 'blue', 'marker': 'o', 'size': 20, 'alpha': 0.6},
            'DME': {'color': 'green', 'marker': 's', 'size': 25, 'alpha': 0.6},
            'VOR-DME': {'color': 'purple', 'marker': 'D', 'size': 35, 'alpha': 0.7},
            'VORTAC': {'color': 'orange', 'marker': '*', 'size': 40, 'alpha': 0.8},
            'TACAN': {'color': 'brown', 'marker': 'h', 'size': 30, 'alpha': 0.6}
        }
        
        # 绘制不同类型的导航设施
        for nav_type in valid_coords['type'].unique():
            if pd.isna(nav_type):
                continue
                
            type_data = valid_coords[valid_coords['type'] == nav_type]
            style = type_styles.get(nav_type, {'color': 'gray', 'marker': '.', 'size': 15, 'alpha': 0.5})
            
            ax.scatter(type_data['longitude_deg'], type_data['latitude_deg'], 
                      c=style['color'], marker=style['marker'], s=style['size'], 
                      alpha=style['alpha'], label=f"{nav_type} ({len(type_data)})")
        
        # 添加网格和标签
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('经度 (°)', fontsize=12)
        ax.set_ylabel('纬度 (°)', fontsize=12)
        ax.set_title('全球导航设施分布图', fontsize=14, fontweight='bold')
        
        # 设置合理的坐标范围
        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)
        
        # 添加图例
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    
    def plot_type_distribution(self, ax):
        """绘制导航设施类型分布饼图"""
        type_counts = self.df['type'].value_counts()
        
        # 只显示前8个最常见的类型，其他归为"其他"
        if len(type_counts) > 8:
            top_types = type_counts.head(8)
            other_count = type_counts.tail(-8).sum()
            if other_count > 0:
                top_types['其他'] = other_count
        else:
            top_types = type_counts
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(top_types)))
        
        wedges, texts, autotexts = ax.pie(top_types.values, labels=top_types.index, 
                                         autopct='%1.1f%%', colors=colors, startangle=90)
        
        ax.set_title('导航设施类型分布', fontsize=14, fontweight='bold')
        
        # 美化文本
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
    
    def plot_country_distribution(self, ax):
        """绘制国家分布条形图"""
        country_counts = self.df['iso_country'].value_counts().head(15)
        
        bars = ax.barh(range(len(country_counts)), country_counts.values, 
                      color=plt.cm.viridis(np.linspace(0, 1, len(country_counts))))
        
        ax.set_yticks(range(len(country_counts)))
        ax.set_yticklabels(country_counts.index)
        ax.set_xlabel('导航设施数量', fontsize=12)
        ax.set_title('各国导航设施数量 (前15名)', fontsize=14, fontweight='bold')
        
        # 在条形图上添加数值标签
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + max(country_counts.values) * 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{int(width)}', ha='left', va='center', fontweight='bold')
        
        ax.grid(axis='x', alpha=0.3)
    
    def plot_frequency_distribution(self, ax):
        """绘制频率分布直方图"""
        # 过滤有效的频率数据
        valid_freq = self.df['frequency_khz'].dropna()
        valid_freq = valid_freq[valid_freq > 0]  # 排除0值
        
        if len(valid_freq) > 0:
            ax.hist(valid_freq, bins=50, alpha=0.7, color='skyblue', edgecolor='navy')
            ax.set_xlabel('频率 (kHz)', fontsize=12)
            ax.set_ylabel('数量', fontsize=12)
            ax.set_title('导航设施频率分布', fontsize=14, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
            
            # 添加统计信息
            ax.text(0.02, 0.98, f'总数: {len(valid_freq):,}\n'
                               f'平均: {valid_freq.mean():.0f} kHz\n'
                               f'中位数: {valid_freq.median():.0f} kHz',
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        else:
            ax.text(0.5, 0.5, '无有效频率数据', ha='center', va='center', transform=ax.transAxes)
    
    def plot_elevation_distribution(self, ax):
        """绘制海拔分布"""
        valid_elev = self.df['elevation_ft'].dropna()
        
        if len(valid_elev) > 0:
            ax.hist(valid_elev, bins=50, alpha=0.7, color='lightcoral', edgecolor='darkred')
            ax.set_xlabel('海拔 (英尺)', fontsize=12)
            ax.set_ylabel('数量', fontsize=12)
            ax.set_title('导航设施海拔分布', fontsize=14, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
            
            # 添加统计信息
            ax.text(0.02, 0.98, f'总数: {len(valid_elev):,}\n'
                               f'平均: {valid_elev.mean():.0f} ft\n'
                               f'中位数: {valid_elev.median():.0f} ft\n'
                               f'最高: {valid_elev.max():.0f} ft',
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        else:
            ax.text(0.5, 0.5, '无有效海拔数据', ha='center', va='center', transform=ax.transAxes)
    
    def create_regional_focus(self, region_bounds=None, region_name="选定区域"):
        """创建区域聚焦图表"""
        if self.df is None:
            return
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # 如果没有指定区域，使用欧洲作为默认
        if region_bounds is None:
            region_bounds = {'lat_min': 35, 'lat_max': 70, 'lon_min': -10, 'lon_max': 40}
            region_name = "欧洲"
        
        # 过滤区域内的数据
        regional_data = self.df[
            (self.df['latitude_deg'] >= region_bounds['lat_min']) &
            (self.df['latitude_deg'] <= region_bounds['lat_max']) &
            (self.df['longitude_deg'] >= region_bounds['lon_min']) &
            (self.df['longitude_deg'] <= region_bounds['lon_max'])
        ].dropna(subset=['latitude_deg', 'longitude_deg'])
        
        # 左图：区域详细分布
        type_styles = {
            'VOR': {'color': 'red', 'marker': '^', 'size': 50},
            'NDB': {'color': 'blue', 'marker': 'o', 'size': 30},
            'DME': {'color': 'green', 'marker': 's', 'size': 40},
            'VOR-DME': {'color': 'purple', 'marker': 'D', 'size': 60},
            'VORTAC': {'color': 'orange', 'marker': '*', 'size': 80}
        }
        
        for nav_type in regional_data['type'].unique():
            if pd.isna(nav_type):
                continue
            type_data = regional_data[regional_data['type'] == nav_type]
            style = type_styles.get(nav_type, {'color': 'gray', 'marker': '.', 'size': 20})
            
            ax1.scatter(type_data['longitude_deg'], type_data['latitude_deg'],
                       c=style['color'], marker=style['marker'], s=style['size'],
                       alpha=0.7, label=f"{nav_type} ({len(type_data)})")
        
        ax1.set_xlim(region_bounds['lon_min'], region_bounds['lon_max'])
        ax1.set_ylim(region_bounds['lat_min'], region_bounds['lat_max'])
        ax1.grid(True, alpha=0.3)
        ax1.set_xlabel('经度 (°)', fontsize=12)
        ax1.set_ylabel('纬度 (°)', fontsize=12)
        ax1.set_title(f'{region_name}地区导航设施分布', fontsize=14, fontweight='bold')
        ax1.legend()
        
        # 右图：该区域的统计分析
        if len(regional_data) > 0:
            country_stats = regional_data['iso_country'].value_counts().head(10)
            bars = ax2.bar(range(len(country_stats)), country_stats.values,
                          color=plt.cm.tab10(np.linspace(0, 1, len(country_stats))))
            
            ax2.set_xticks(range(len(country_stats)))
            ax2.set_xticklabels(country_stats.index, rotation=45, ha='right')
            ax2.set_ylabel('导航设施数量', fontsize=12)
            ax2.set_title(f'{region_name}各国导航设施统计', fontsize=14, fontweight='bold')
            
            # 添加数值标签
            for i, bar in enumerate(bars):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + max(country_stats.values) * 0.01,
                        f'{int(height)}', ha='center', va='bottom', fontweight='bold')
        else:
            ax2.text(0.5, 0.5, f'{region_name}区域无数据', ha='center', va='center', transform=ax2.transAxes)
        
        plt.tight_layout()
        return fig
    
    def print_data_summary(self):
        """打印数据摘要"""
        if self.df is None:
            return
            
        print("\n" + "="*60)
        print("导航设施数据详细摘要")
        print("="*60)
        
        print(f"\n📊 基本统计:")
        print(f"   总记录数: {len(self.df):,}")
        print(f"   覆盖国家: {self.df['iso_country'].nunique()} 个")
        print(f"   导航设施类型: {self.df['type'].nunique()} 种")
        
        print(f"\n🌍 地理分布:")
        lat_range = self.df['latitude_deg'].max() - self.df['latitude_deg'].min()
        lon_range = self.df['longitude_deg'].max() - self.df['longitude_deg'].min()
        print(f"   纬度范围: {self.df['latitude_deg'].min():.2f}° 到 {self.df['latitude_deg'].max():.2f}° (跨度 {lat_range:.2f}°)")
        print(f"   经度范围: {self.df['longitude_deg'].min():.2f}° 到 {self.df['longitude_deg'].max():.2f}° (跨度 {lon_range:.2f}°)")
        
        print(f"\n📡 导航设施类型分布:")
        type_counts = self.df['type'].value_counts()
        for nav_type, count in type_counts.head(10).items():
            percentage = (count / len(self.df)) * 100
            print(f"   {nav_type:<12}: {count:>6,} ({percentage:>5.1f}%)")
        
        print(f"\n🏳️ 主要国家分布:")
        country_counts = self.df['iso_country'].value_counts()
        for country, count in country_counts.head(10).items():
            percentage = (count / len(self.df)) * 100
            print(f"   {country:<3}: {count:>6,} ({percentage:>5.1f}%)")
        
        # 频率统计
        valid_freq = self.df['frequency_khz'].dropna()
        if len(valid_freq) > 0:
            print(f"\n📻 频率统计:")
            print(f"   有效频率记录: {len(valid_freq):,}")
            print(f"   频率范围: {valid_freq.min():.0f} - {valid_freq.max():.0f} kHz")
            print(f"   平均频率: {valid_freq.mean():.0f} kHz")
        
        # 海拔统计
        valid_elev = self.df['elevation_ft'].dropna()
        if len(valid_elev) > 0:
            print(f"\n⛰️ 海拔统计:")
            print(f"   有效海拔记录: {len(valid_elev):,}")
            print(f"   海拔范围: {valid_elev.min():.0f} - {valid_elev.max():.0f} 英尺")
            print(f"   平均海拔: {valid_elev.mean():.0f} 英尺")

def main():
    """主函数"""
    csv_file_path = "/home/longqin/Downloads/navaids.csv"  # 您的CSV文件路径
    
    print("航空导航设施数据可视化分析工具")
    print("="*50)
    
    # 创建可视化器
    visualizer = NavAidsVisualizer(csv_file_path)
    
    if visualizer.df is not None:
        # 打印数据摘要
        visualizer.print_data_summary()
        
        # 创建综合仪表板
        print("\n正在生成综合分析仪表板...")
        fig1 = visualizer.create_comprehensive_dashboard()
        
        # 创建欧洲区域聚焦图
        print("正在生成欧洲区域详细分析...")
        fig2 = visualizer.create_regional_focus()
        
        # 创建北美区域聚焦图
        print("正在生成北美区域详细分析...")
        na_bounds = {'lat_min': 25, 'lat_max': 70, 'lon_min': -170, 'lon_max': -50}
        fig3 = visualizer.create_regional_focus(na_bounds, "北美")
        
        plt.show()
        
        print("\n✅ 可视化完成！")
        print("💡 提示: 可以根据需要调整区域边界来查看特定地区的详细信息")
    else:
        print("❌ 数据加载失败，请检查文件路径")

if __name__ == "__main__":
    main()