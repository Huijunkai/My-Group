# 导入依赖包
import requests as req
import re
import json
from lxml import etree

# 设置伪装信息
heads = {
    'user-agent' : 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36'
}

# 发送请求
res = req.get('https://www.baidu.com', headers=heads)

# 设置中文编码
res.encoding = 'utf-8'

# 打印信息
startIndex = res.text.index('<textarea id="hotsearch_data" style="display:none;">')
endIndex = res.text.index('</textarea></div></div>')
print(startIndex)
print(endIndex)

allNewsStr = res.text[startIndex:endIndex].replace('<textarea id="hotsearch_data" style="display:none;">', '')
print(allNewsStr)
# 把字符串转成字典格式
news = json.loads(allNewsStr)
print(news['hotsearch'])

# 开始解析数据 <span class="title-content-title">00后女钢筋工每天工地干10小时赚320</span>
# <span class="title-content-title">(.*)</span>
result = re.findall('<span class="title-content-title">(.*?)</span>', res.text)
print(result)

# <textarea id="hotsearch_data" style="display:none;">(.*)</textarea>
result2 = re.findall('<script type="application/json" id="placeholder-data" data-for="result-data">(.*)</script>', res.text)
print(result2)

# //*[@id="recent-hot"]/div/div[2]/div/div[2]/div[1]/div[2]/div[2]/a/div[2]/span/span[1]
