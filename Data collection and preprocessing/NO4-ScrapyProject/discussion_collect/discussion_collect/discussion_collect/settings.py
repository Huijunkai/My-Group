# Scrapy settings for discussion_collect project
#
# For simplicity, this file contains only settings considered important or
# commonly used. You can find more settings consulting the documentation:
#
#     https://docs.scrapy.org/en/latest/topics/settings.html
#     https://docs.scrapy.org/en/latest/topics/downloader-middleware.html
#     https://docs.scrapy.org/en/latest/topics/spider-middleware.html

BOT_NAME = "discussion_collect"

SPIDER_MODULES = ["discussion_collect.spiders"]
NEWSPIDER_MODULE = "discussion_collect.spiders"


# Crawl responsibly by identifying yourself (and your website) on the user-agent
#USER_AGENT = "discussion_collect (+http://www.yourdomain.com)"

# Obey robots.txt rules
ROBOTSTXT_OBEY = True

# Configure maximum concurrent requests performed by Scrapy (default: 16)
#CONCURRENT_REQUESTS = 32

# Configure a delay for requests for the same website (default: 0)
# See https://docs.scrapy.org/en/latest/topics/settings.html#download-delay
# See also autothrottle settings and docs
#DOWNLOAD_DELAY = 3
# The download delay setting will honor only one of:
#CONCURRENT_REQUESTS_PER_DOMAIN = 16
#CONCURRENT_REQUESTS_PER_IP = 16

# Disable cookies (enabled by default)
#COOKIES_ENABLED = False

# Disable Telnet Console (enabled by default)
#TELNETCONSOLE_ENABLED = False

# Override the default request headers:
#DEFAULT_REQUEST_HEADERS = {
#    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
#    "Accept-Language": "en",
#}

# Enable or disable spider middlewares
# See https://docs.scrapy.org/en/latest/topics/spider-middleware.html
#SPIDER_MIDDLEWARES = {
#    "discussion_collect.middlewares.DiscussionCollectSpiderMiddleware": 543,
#}

# Enable or disable downloader middlewares
# See https://docs.scrapy.org/en/latest/topics/downloader-middleware.html
#DOWNLOADER_MIDDLEWARES = {
#    "discussion_collect.middlewares.DiscussionCollectDownloaderMiddleware": 543,
#}

# Enable or disable extensions
# See https://docs.scrapy.org/en/latest/topics/extensions.html
#EXTENSIONS = {
#    "scrapy.extensions.telnet.TelnetConsole": None,
#}

# Configure item pipelines
# See https://docs.scrapy.org/en/latest/topics/item-pipeline.html
#ITEM_PIPELINES = {
#    "discussion_collect.pipelines.DiscussionCollectPipeline": 300,
#}

# Enable and configure the AutoThrottle extension (disabled by default)
# See https://docs.scrapy.org/en/latest/topics/autothrottle.html
#AUTOTHROTTLE_ENABLED = True
# The initial download delay
#AUTOTHROTTLE_START_DELAY = 5
# The maximum download delay to be set in case of high latencies
#AUTOTHROTTLE_MAX_DELAY = 60
# The average number of requests Scrapy should be sending in parallel to
# each remote server
#AUTOTHROTTLE_TARGET_CONCURRENCY = 1.0
# Enable showing throttling stats for every response received:
#AUTOTHROTTLE_DEBUG = False

# Enable and configure HTTP caching (disabled by default)
# See https://docs.scrapy.org/en/latest/topics/downloader-middleware.html#httpcache-middleware-settings
#HTTPCACHE_ENABLED = True
#HTTPCACHE_EXPIRATION_SECS = 0
#HTTPCACHE_DIR = "httpcache"
#HTTPCACHE_IGNORE_HTTP_CODES = []
#HTTPCACHE_STORAGE = "scrapy.extensions.httpcache.FilesystemCacheStorage"

# Set settings whose default value is deprecated to a future-proof value
REQUEST_FINGERPRINTER_IMPLEMENTATION = "2.7"
TWISTED_REACTOR = "twisted.internet.asyncioreactor.AsyncioSelectorReactor"
FEED_EXPORT_ENCODING = "utf-8"


# 禁用SSL警告（可选，可放在爬虫文件中）
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# 自定义配置
# ---- 反爬优化配置（合并与增强） ----
# 是否遵循 robots.txt（爬取测试时可关闭，生产请遵守目标站点政策）
ROBOTSTXT_OBEY = False

# 并发与延迟
CONCURRENT_REQUESTS = 8
CONCURRENT_REQUESTS_PER_DOMAIN = 4
DOWNLOAD_DELAY = 1.0
RANDOMIZE_DOWNLOAD_DELAY = True
DOWNLOAD_TIMEOUT = 20

# Cookie 与日志
COOKIES_ENABLED = False
LOG_LEVEL = 'INFO'
TELNETCONSOLE_ENABLED = False

# 自动节流（AutoThrottle）
AUTOTHROTTLE_ENABLED = True
AUTOTHROTTLE_START_DELAY = 1
AUTOTHROTTLE_MAX_DELAY = 10
AUTOTHROTTLE_TARGET_CONCURRENCY = 1.0
AUTOTHROTTLE_DEBUG = False

# 重试设置（处理 429 等错误）
RETRY_ENABLED = True
RETRY_TIMES = 5
# 包含 429（Too Many Requests）以便重试
RETRY_HTTP_CODES = [408, 429, 500, 502, 503, 504]

# 请求头与编码
DEFAULT_REQUEST_HEADERS = {
	'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
	'Accept-Language': 'zh-CN,zh;q=0.9',
}

# 用户代理池（仅使用 Windows 系统伪装）
USER_AGENT_LIST = [
	# Chrome on Windows 11/10
	'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
	'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.5845.96 Safari/537.36',
	# Edge on Windows
	'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edg/120.0.0.0 Chrome/120.0.0.0 Safari/537.36',
	'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edg/116.0.1938.62 Chrome/116.0.5845.96 Safari/537.36',
	# Firefox on Windows
	'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:119.0) Gecko/20100101 Firefox/119.0',
	'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:115.0) Gecko/20100101 Firefox/115.0',
	# Opera on Windows
	'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) OPR/100.0.0.0 Chrome/120.0.0.0 Safari/537.36',
	# Older Windows (兼容) Chrome
	'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36',
]

# 代理池（示例）：可以在运行时从文件/服务加载或使用动态代理池
PROXY_LIST = [
	# 'http://username:password@proxy.example:8080',
	# 'http://proxy2.example:3128',
]

# 启用自定义下载中间件（随机 UA、代理）以及内置的重试与代理中间件
DOWNLOADER_MIDDLEWARES = {
	# 自定义随机 UA 中间件（实现见 middlewares.py）
	'discussion_collect.middlewares.RandomUserAgentMiddleware': 400,
	# 内置重试中间件（确保在 UA 之后运行）
	'scrapy.downloadermiddlewares.retry.RetryMiddleware': 500,
	# 内置 HTTP 代理中间件
	'scrapy.downloadermiddlewares.httpproxy.HttpProxyMiddleware': 750,
	# 自定义代理中间件（如果 PROXY_LIST 非空）
	'discussion_collect.middlewares.RandomProxyMiddleware': 740,
}

# 其他建议
# 如果需要绕过 JS 渲染强的页面，可以考虑整合 Selenium 或 Playwright
# ---- 结束配置 ----