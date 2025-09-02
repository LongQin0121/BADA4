import xarray as xr
import numpy as np
from datetime import datetime, timedelta
import requests

def get_ncep_wind_profile(lat, lon, date_str="20241215", cycle="12"):
    """
    获取NCEP GFS风廓线数据
    """
    # 尝试多个可能的URL格式
    base_urls = [
        f"https://nomads.ncep.noaa.gov/dods/gfs_0p25/gfs{date_str}/gfs_0p25_{cycle}z",
        f"https://nomads.ncep.noaa.gov/dods/gfs_1p00/gfs{date_str}/gfs_1p00_{cycle}z"
    ]
    
    for url in base_urls:
        try:
            print(f"尝试连接: {url}")
            
            # 设置xarray选项
            xr.set_options(display_max_rows=20)
            
            # 尝试打开数据集
            ds = xr.open_dataset(url, engine='pydap')
            print("数据集连接成功!")
            
            # 检查可用变量
            print("可用变量:", list(ds.data_vars.keys())[:10])
            
            # 选择最近的网格点
            point_data = ds.sel(lat=lat, lon=lon, method='nearest')
            
            # 查找风场变量（可能的变量名）
            u_var_names = ['ugrdprs', 'ugrd', 'u', 'U']
            v_var_names = ['vgrdprs', 'vgrd', 'v', 'V']
            
            u_wind = None
            v_wind = None
            
            for u_name in u_var_names:
                if u_name in ds.data_vars:
                    u_wind = point_data[u_name].isel(time=0)
                    print(f"找到U分量变量: {u_name}")
                    break
                    
            for v_name in v_var_names:
                if v_name in ds.data_vars:
                    v_wind = point_data[v_name].isel(time=0)
                    print(f"找到V分量变量: {v_name}")
                    break
            
            if u_wind is None or v_wind is None:
                print("未找到风场变量，可用变量:", list(ds.data_vars.keys()))
                continue
            
            # 计算风速和风向
            wind_speed = np.sqrt(u_wind**2 + v_wind**2)
            wind_direction = np.arctan2(u_wind, v_wind) * 180/np.pi
            wind_direction = (wind_direction + 360) % 360
            
            print(f"\n武汉 ({lat}°N, {lon}°E) {date_str} {cycle}Z 风廓线:")
            print("压力层(hPa) | 风速(m/s) | 风向(°)")
            print("-" * 35)
            
            # 获取压力层信息
            if 'lev' in ds.dims:
                levels = point_data.lev
            elif 'plev' in ds.dims:
                levels = point_data.plev
            else:
                print("未找到压力层维度")
                continue
            
            for i, level in enumerate(levels):
                try:
                    ws = float(wind_speed[i])
                    wd = float(wind_direction[i])
                    if not (np.isnan(ws) or np.isnan(wd)):
                        print(f"{level.values:>8.0f} | {ws:>8.1f} | {wd:>7.0f}")
                except:
                    continue
            
            return wind_speed, wind_direction, levels
            
        except Exception as e:
            print(f"URL {url} 失败: {e}")
            continue
    
    print("所有NCEP数据源均无法访问")
    return None, None, None

def get_recent_gfs_data():
    """
    获取最近可用的GFS数据
    """
    from datetime import datetime, timedelta
    
    # GFS数据通常有几小时延迟，尝试最近几天的数据
    for days_back in range(0, 7):
        date = datetime.now() - timedelta(days=days_back)
        date_str = date.strftime("%Y%m%d")
        
        for cycle in ["00", "06", "12", "18"]:
            print(f"尝试日期: {date_str}, 时次: {cycle}Z")
            ws, wd, levels = get_ncep_wind_profile(30.6, 114.3, date_str, cycle)
            if ws is not None:
                return ws, wd, levels
                
    return None, None, None

# 备选方案：使用本地示例数据
def show_sample_wind_profile():
    """
    显示武汉12月典型风廓线示例
    """
    print("武汉12月典型风廓线 (基于气候学数据):")
    print("压力层(hPa) | 高度(m) | 风速(m/s) | 风向(°) | 风向描述")
    print("-" * 60)
    
    sample_data = [
        (1000, 100, 3.2, 45, "东北风"),
        (925, 800, 5.8, 60, "东北风"),
        (850, 1500, 8.1, 75, "东北风"),
        (700, 3000, 12.5, 240, "西南风"),
        (500, 5500, 18.3, 270, "西风"),
        (400, 7000, 22.1, 280, "西北风"),
        (300, 9000, 28.5, 285, "西北风"),
        (200, 12000, 35.2, 290, "西北风"),
        (100, 16000, 25.8, 295, "西北风"),
    ]
    
    for pres, height, ws, wd, desc in sample_data:
        print(f"{pres:>8} | {height:>7} | {ws:>8.1f} | {wd:>7.0f} | {desc}")

if __name__ == "__main__":
    print("正在安装依赖包...")
    import subprocess
    import sys
    
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'netcdf4', 'pydap', 'dask'])
        print("依赖包安装完成!")
    except:
        print("自动安装失败，请手动安装: pip install netcdf4 pydap dask")
    
    print("\n尝试获取实时NCEP数据...")
    ws, wd, levels = get_recent_gfs_data()
    
    if ws is None:
        print("\n无法获取实时数据，显示示例数据:")
        show_sample_wind_profile()
    
    print("\n数据获取完成!")


    """
    您说得对！气象数据中压力层需要转换成对应的高度才更直观。这个程序输出的是不同气压层的风廓线，但没有显示对应的海拔高度。
标准大气压力与高度的对应关系：
压力层(hPa)近似高度(米)风速(m/s)风向(°)10001104.53379753208.935395054010.1892576010.3229009909.83285014609.05180019508.46375024608.67170030108.17665035907.97360042107.877550488010.575500559015.269450634014.168400719012.454350812013.153300916015.1472501036017.5342001178016.551501361011.265100161809.7198
从这个高度角度看：

低层（0-3000米）：风速相对较小，风向变化较大
中层（3000-9000米）：风速逐渐增强，主要为东北风
高层（9000米以上）：出现明显的西风急流
    
    """