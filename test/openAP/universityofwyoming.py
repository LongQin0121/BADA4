import requests
import pandas as pd
from io import StringIO

def get_sounding_data(station_id="57494", year="2024", month="12", day="15", hour="12"):
    """
    获取探空数据
    station_id: 57494 是武汉站的WMO编号
    """
    url = "http://weather.uwyo.edu/cgi-bin/sounding"
    params = {
        'region': 'seasia',
        'TYPE': 'TEXT%3ALIST',
        'YEAR': year,
        'MONTH': month,
        'FROM': f"{day}{hour}",
        'TO': f"{day}{hour}",
        'STNM': station_id
    }
    
    try:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            print(f"武汉探空数据 {year}-{month}-{day} {hour}Z:")
            print("从University of Wyoming获取的实测数据")
            print(response.text[:1000] + "...")  # 显示前1000字符
            return response.text
        else:
            print(f"数据获取失败，状态码: {response.status_code}")
            return None
    except Exception as e:
        print(f"探空数据获取失败: {e}")
        return None

# 使用示例
sounding_data = get_sounding_data()