import cdsapi
import xarray as xr
import numpy as np

# 初始化CDS API客户端
c = cdsapi.Client()

# 下载ERA5压力层数据
c.retrieve(
    'reanalysis-era5-pressure-levels',
    {
        'product_type': 'reanalysis',
        'format': 'netcdf',
        'variable': [
            'u_component_of_wind', 'v_component_of_wind'
        ],
        'pressure_level': [
            '50', '100', '200', '300', '400', 
            '500', '700', '850', '925', '1000'
        ],
        'year': '2024',
        'month': '12',
        'day': '15',  # 12月15日示例
        'time': ['00:00', '06:00', '12:00', '18:00'],
        'area': [31, 114, 30, 115],  # 武汉周边区域 [N, W, S, E]
        'grid': [0.25, 0.25],
    },
    'wuhan_wind_profile_20241215.nc')

# 读取和处理数据
ds = xr.open_dataset('wuhan_wind_profile_20241215.nc')

# 选择武汉最近的网格点
wuhan_data = ds.sel(latitude=30.6, longitude=114.3, method='nearest')

# 计算风速和风向
u_wind = wuhan_data['u']
v_wind = wuhan_data['v']
wind_speed = np.sqrt(u_wind**2 + v_wind**2)
wind_direction = np.arctan2(u_wind, v_wind) * 180/np.pi
wind_direction = (wind_direction + 360) % 360

# 显示12月15日12:00的风廓线
time_idx = wuhan_data.sel(time='2024-12-15T12:00:00')
print("武汉 2024-12-15 12:00 UTC 风廓线:")
print("压力层(hPa) | 风速(m/s) | 风向(°)")
for i, level in enumerate(ds.level):
    ws = float(wind_speed.sel(time='2024-12-15T12:00:00')[i])
    wd = float(wind_direction.sel(time='2024-12-15T12:00:00')[i])
    print(f"{level.values:>8} | {ws:>8.1f} | {wd:>7.0f}")



"""
这个错误是因为缺少CDS API的配置文件。你需要先设置ERA5的访问凭据。让我帮你解决这个问题：
解决步骤：
1. 注册CDS账户
访问：https://cds.climate.copernicus.eu/user/register
注册并激活账户
2. 获取API密钥
登录后访问：https://cds.climate.copernicus.eu/api-how-to
你会看到类似这样的信息：
url: https://cds.climate.copernicus.eu/api/v2
key: 12345:abcdef12-3456-7890-abcd-ef1234567890
3. 创建配置文件
在你的home目录创建 .cdsapirc 文件：
bash# 方法1：直接编辑
nano ~/.cdsapirc

# 或者方法2：直接写入
cat > ~/.cdsapirc << EOF
url: https://cds.climate.copernicus.eu/api/v2
key: YOUR_UID:YOUR_API_KEY
EOF
文件内容应该是：
url: https://cds.climate.copernicus.eu/api/v2
key: 你的UID:你的API密钥
4. 设置文件权限
bashchmod 600 ~/.cdsapirc
5. 验证配置
pythonimport cdsapi
c = cdsapi.Client()
print("CDS API配置成功！")
备选方案：使用其他数据源
如果你暂时不想注册CDS，可以使用这些替代方案：
"""