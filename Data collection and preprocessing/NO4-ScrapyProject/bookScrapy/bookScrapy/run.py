import os
from scrapy import cmdline

# 切换到项目根目录（根据实际路径修改）
os.chdir(r"F:\My-study\Data collection and preprocessing\NO4-ScrapyProject\bookScrapy")

# 执行爬虫命令
cmdline.execute('scrapy crawl book'.split())

# from scrapy import cmdline