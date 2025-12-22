import requests
from lxml import etree
import pandas as pd
from typing import List, Dict
import re
import time

urls = [f"https://www.cnhnb.com/p/jiuhuang-0-0-0-0-{page}/" for page in range(1, 27)]

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    "Referer": "https://www.cnhnb.com/"
}

def fetch_html(url: str) -> str:
    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.encoding = response.apparent_encoding  # 自动识别编码
        if response.status_code == 200:
            return response.text
        else:
            print(f"访问失败，状态码：{response.status_code}")
            return ""
    except Exception as e:
        print(f"爬取网页{url}时出错：{str(e)}")
        return ""

def parse_jiuhuang_data(html_content: str) -> List[Dict]:
    tree = etree.HTML(html_content)
    items = tree.xpath('//div[@class="supply-item"]')
    data_list = []

    for item in items:
        # 商品标题（保留原始格式）
        title = item.xpath('.//h2/text()')
        title = ''.join([t.strip() for t in title if t.strip()]) if title else "查无信息"
        
        # 价格（保留原始格式）
        price = item.xpath('.//span[@class="sp1"]/text()')
        price = price[0].strip() if price else "查无信息"
        
        # 单位（保留原始格式）
        unit = item.xpath('.//div[@class="shops-price"]/text()')
        unit = ''.join([u.strip() for u in unit if u.strip() and u.strip() != price]) if unit else "查无信息"
        
        # 产地（保留原始格式）
        origin = item.xpath('.//div[@class="r-shop-btm"]/text()')
        origin = origin[0].strip() if origin else "查无信息"
        
        # 供应商（保留原始格式）
        supplier = item.xpath('.//div[@class="l-shop-btm"]/a/text()')
        supplier = supplier[0].strip() if supplier else "查无信息"
        
        # 茬数（原始提取结果）
        spec_str = title + ''.join(item.xpath('.//div[@class="title"]/h2/text()'))
        stubble = "查无信息"
        if "头茬" in spec_str:
            stubble = "头茬"
        elif "二茬" in spec_str:
            stubble = "二茬"
        elif "三茬" in spec_str:
            stubble = "三茬"
        
        # 高度（原始提取结果）
        height = "查无信息"
        height_keywords = ["20~30cm", "30~40公分", "40~50cm", "50~60cm", "60~70cm"]
        for h in height_keywords:
            if h in spec_str:
                height = h
                break
        
        # 成交金额（保留原始格式）
        turnover = item.xpath('.//div[@class="turnover"]/text()')
        turnover = turnover[0].strip() if turnover else "查无信息"
        
        # 标签（保留原始格式）
        tags = item.xpath('.//div[@class="cw-tag"]/text()')
        tags = [tag.strip() for tag in tags if tag.strip()]
        tags = ",".join(tags) if tags else "查无信息"
        
        # 组装原始数据（不做任何修改）
        data = {
            "商品标题": title,
            "价格": price,
            "单位": unit,
            "产地": origin,
            "供应商": supplier,
            "茬数": stubble,
            "高度": height,
            "成交金额": turnover,
            "标签": tags
        }
        data_list.append(data)
    
    return data_list

def crawl_online_jiuhuang() -> List[Dict]:
    all_data = []
    
    for url in urls:
        print(f"正在爬取：{url}")
        html = fetch_html(url)
        if not html:
            time.sleep(2)
            continue
        page_data = parse_jiuhuang_data(html)
        all_data.extend(page_data)
        print(f"该页爬取到{len(page_data)}条原始数据")
        time.sleep(2)  # 爬取间隔，避免请求过快
    
    return all_data

def clean_jiuhuang_data(raw_data: List[Dict]) -> List[Dict]:
    """清洗韭黄批发数据：去重 + 补全空值"""
    cleaned_data = []
    seen = set()  # 去重集合
    
    for data in raw_data:
        # 1. 去重处理：基于核心字段判断，避免重复记录
        unique_key = (data["商品标题"], data["价格"], data["单位"], data["产地"], data["供应商"])
        if unique_key in seen:
            continue
        seen.add(unique_key)
        
        # 2. 补全空值
        clean_data = data.copy()
        # 价格补全
        if clean_data["价格"] == "查无信息":
            clean_data["价格_清洗后"] = "暂无数据"
        else:
            # 提取数字，统一格式
            price_nums = re.findall(r'\d+\.?\d*', clean_data["价格"])
            if len(price_nums) == 1:
                clean_data["价格_清洗后"] = f"{float(price_nums[0]):.2f}"
            elif len(price_nums) >= 2:
                price_avg = (float(price_nums[0]) + float(price_nums[1])) / 2
                clean_data["价格_清洗后"] = f"{price_avg:.2f}"
            else:
                clean_data["价格_清洗后"] = "暂无数据"
        
        # 单位补全（默认元/斤）
        clean_data["单位_清洗后"] = clean_data["单位"] if clean_data["单位"] != "查无信息" else "元/斤"
        
        # 产地补全（默认未知）
        clean_data["产地_清洗后"] = clean_data["产地"] if clean_data["产地"] != "查无信息" else "未知"
        
        # 供应商补全（默认未知供应商）
        clean_data["供应商_清洗后"] = clean_data["供应商"] if clean_data["供应商"] != "查无信息" else "未知供应商"
        
        # 茬数补全（默认未知茬数）
        clean_data["茬数_清洗后"] = clean_data["茬数"] if clean_data["茬数"] != "查无信息" else "未知茬数"
        
        # 高度补全（默认未知高度）
        clean_data["高度_清洗后"] = clean_data["高度"] if clean_data["高度"] != "查无信息" else "未知高度"
        
        # 成交金额补全（默认0）
        if clean_data["成交金额"] == "查无信息":
            clean_data["成交金额_清洗后"] = "0"
        else:
            turnover_nums = re.findall(r'\d+\.?\d*', clean_data["成交金额"])
            clean_data["成交金额_清洗后"] = turnover_nums[0] if turnover_nums else "0"
        
        # 标签补全（默认无标签）
        clean_data["标签_清洗后"] = clean_data["标签"] if clean_data["标签"] != "查无信息" else "无标签"
        
        cleaned_data.append(clean_data)
    
    return cleaned_data

print("开始爬取26个分页的韭黄批发原始数据...")
raw_result = crawl_online_jiuhuang()

# 保存原始数据
raw_df = pd.DataFrame(raw_result)
raw_df.to_csv("原始韭黄批发数据_26页.csv", index=False, encoding="utf-8-sig")
print(f"\n原始数据保存完成！共获取{len(raw_result)}条原始数据")
print("原始数据保存路径：原始韭黄批发数据_26页.csv")

# 数据清洗（去重+补全空值）
print("\n开始清洗数据...")
cleaned_result = clean_jiuhuang_data(raw_result)

# 保存清洗后数据
cleaned_df = pd.DataFrame(cleaned_result)[[
    "商品标题", "价格_清洗后", "单位_清洗后", "产地_清洗后", 
    "供应商_清洗后", "茬数_清洗后", "高度_清洗后", "成交金额_清洗后", "标签_清洗后"
]]
cleaned_df.columns = [
    "商品标题", "价格（元）", "单位", "产地", "供应商", 
    "茬数", "高度", "成交金额", "标签"
]
cleaned_df.to_csv("清洗后韭黄批发数据_26页.csv", index=False, encoding="utf-8-sig")

print(f"\n数据清洗完成！清洗后保留{len(cleaned_result)}条有效数据")
print("清洗后数据保存路径：清洗后韭黄批发数据_26页.csv")