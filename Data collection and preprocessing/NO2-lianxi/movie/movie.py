# 导入依赖包
import requests as req
import json
import re
from bs4 import BeautifulSoup

# 发送请求

# 伪装
heads ={
    'user-agent' : 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36 Edg/141.0.0.0',
    'Cookie' : 'll="118302"; bid=GmUemmMKVUA; _pk_ref.100001.4cf6=%5B%22%22%2C%22%22%2C1757987431%2C%22https%3A%2F%2Fwww.baidu.com%2Flink%3Furl%3DN369YYCCjQz7Haokiak4MK6ozhTmmmN2zvL3axLYudLSI59upGbF5amcLwGdrjYD%26wd%3D%26eqid%3D84073e5b02607de20000000668c8c262%22%5D; _pk_id.100001.4cf6=4ebba9ce5ff2da0a.1757987431.; _pk_ses.100001.4cf6=1; ap_v=0,6.0; __utma=30149280.1858202652.1757987432.1757987432.1757987432.1; __utmb=30149280.0.10.1757987432; __utmc=30149280; __utmz=30149280.1757987432.1.1.utmcsr=baidu|utmccn=(organic)|utmcmd=organic; __utma=223695111.1249666086.1757987432.1757987432.1757987432.1; __utmb=223695111.0.10.1757987432; __utmc=223695111; __utmz=223695111.1757987432.1.1.utmcsr=baidu|utmccn=(organic)|utmcmd=organic; __yadk_uid=zUwiroJynzpJtsayAHZ1SNSr7ILPBCt2; _vwo_uuid_v2=D8D6011DB1CFB180005FA9A7EB6C1E29D|9f8059e35747060f1b2beb84cb0039e7'
}

# 请求
res = req.get('https://movie.douban.com/', headers = heads)

# 中文编码UTF-8
res.encoding = 'utf-8'

# 打印信息
print(res.text)

# 打印信息
startIndex = res.text.index('<ul class="ui-slide-content" data-slide-index="1" data-index-max="9">')
endIndex = res.text.index(' <li class="ui-slide-item">')
print(startIndex)
print(endIndex)

allNewsStr = res.text[startIndex:endIndex].replace('<ul class="ui-slide-content" data-slide-index="1" data-index-max="9">', '')
print(allNewsStr)