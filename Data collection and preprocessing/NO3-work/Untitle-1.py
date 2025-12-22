import requests
import pandas as pd
from datetime import datetime
import time

def get_flight_data(airport_code="PEK", date=None):
    """\
    获取指定机场和日期的航班信息\
    :param airport_code: 机场ICAO代码 (默认: 北京首都PEK)\
    :param date: 日期 (格式: YYYY-MM-DD, 默认当天)\
    :return: 包含航班数据的DataFrame\
    """
    if not date:
        date = datetime.now().strftime("%Y-%m-%d")
    
    api_url = f"https://api.flightera.net/api9/airport?airport={airport_code}&date={date}&lang=zh"
    
    # 修改请求头关闭持久连接（解决连接池问题）
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
        "Connection": "close"  # 新增此行
    }

    
    try:
        response = requests.get(api_url, headers=headers, timeout=15)  # 超时延长至15秒
    except requests.exceptions.ConnectionError:
        print("域名解析失败，正在重试...")
        time.sleep(5)  # 等待5秒后重试
        response = requests.get(api_url, headers=headers, timeout=15)
        response.raise_for_status()
        data = response.json()
        
        # 合并到达和出发航班
        flights = data.get("arrivals", []) + data.get("departures", [])
        if not flights:
            print("未找到航班数据")
            return None
        
        # 创建DataFrame
        df = pd.DataFrame([{
            "航班号": f.get("flight"),
            "航空公司": f.get("airline"),
            "出发机场": f.get("origin") if "origin" in f else airport_code,
            "到达机场": f.get("destination") if "destination" in f else airport_code,
            "计划时间": f.get("scheduled_time"),
            "实际/预计时间": f.get("actual_time"),
            "状态": f.get("status"),
            "机型": f.get("aircraft_type"),
            "数据来源": "Flightera"
        } for f in flights])
        
        return df.drop_duplicates()
    
    except Exception as e:
        print(f"爬取失败: {e}")
        return None

# 使用示例
if __name__ == "__main__":
    # 获取北京首都机场今日航班
    df = get_flight_data("PEK")  
    
    if df is not None:
        print(f"共获取 {len(df)} 条航班记录")
        print(df.head(5))  # 打印前5条
        
        # 保存到CSV
        df.to_csv(f"flights_PEK_{datetime.now().strftime('%Y%m%d')}.csv", index=False)
        print("数据已保存到CSV文件")
