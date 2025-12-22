""" 
此子项目用于爬取作者标题的详细页内容的文本内容和评论内容，并保存为json格式文件，
后续对JSON文件进行数据处理，过滤所有的特殊符号，只保留纯文本内容和数字以及常用标点符号，
清洗后保存为新的CSV文件，
文件命名格式为：discussion_Detailed_Page_cleaned.csv,
标头为：标题，清洗后的正文内容，清洗后的评论内容列表（多个评论以分隔符分隔），评论带上评论人的用户名。
 """
import csv
import json
import os
import re
import threading
from tqdm import tqdm
from scrapy import Spider, Request
from scrapy.http import HtmlResponse
from urllib.parse import urlparse

class DiscussionDetailedPageSpider(Spider):
    name = 'discussion_Detailed_Page_spiders'
    
    # 文件路径配置
    input_csv_path = os.path.join('data', 'csv', 'fedora_centos_topics_cleaned.csv')  # 清洗后的输入CSV
    output_csv_path = os.path.join('data', 'csv', 'discussion_Detailed_Page_cleaned.csv')  # 最终输出CSV
    url_field_name = '标题详情URL'  # 请根据实际字段名修改
    
    # 特殊字符过滤正则（保留中文、字母、数字、常用标点、空格、小数点）
    ALLOWED_CHARS_PATTERN = r'[^\u4e00-\u9fa5a-zA-Z0-9\s\.,，。！？；;:""''（）\(\)-_=+\[\]{}、·…—]'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.total_urls = 0  # 总待爬取URL数
        self.completed_urls = 0  # 已完成爬取的URL数
        self.pbar = None  # 进度条对象
        self.lock = threading.Lock()  # 线程锁（保证多线程下进度条更新安全）
        
        # 自定义Scrapy配置：降低日志级别、禁用默认进度条、控制并发
        self.custom_settings = {
            'LOG_LEVEL': 'INFO',  # 只输出INFO及以上日志，减少进度条干扰
            'DOWNLOAD_DELAY': 0.5,  # 下载延迟，避免被目标网站封禁
            'CONCURRENT_REQUESTS': 5,  # 并发请求数，根据目标网站调整
            'USER_AGENT': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }

    def start_requests(self):
        """从清洗后的CSV读取URL，统计总数并生成爬取请求"""
        # 检查输入文件是否存在
        if not os.path.exists(self.input_csv_path):
            self.logger.error(f"输入文件不存在: {self.input_csv_path}")
            return
        
        # 第一步：先读取所有有效URL，统计总数
        url_list = []
        try:
            with open(self.input_csv_path, 'r', encoding='utf-8-sig', newline='') as f:
                reader = csv.DictReader(f)
                # 检查URL字段是否存在
                if self.url_field_name not in reader.fieldnames:
                    self.logger.error(f"CSV中未找到「{self.url_field_name}」字段，请检查字段名！")
                    return
                
                for row in reader:
                    url = row.get(self.url_field_name, '').strip()
                    title = row.get('标题', '未获取到标题').strip()
                    
                    if not url or url == '未获取到信息':
                        self.logger.warning(f"跳过空URL，标题：{title}")
                        continue
                    url_list.append((url, title))
            
            # 初始化进度条
            self.total_urls = len(url_list)
            if self.total_urls > 0:
                self.pbar = tqdm(
                    total=self.total_urls,
                    desc="爬取进度",
                    unit="URL",
                    ncols=80,  # 进度条宽度
                    colour="green",  # 进度条颜色
                    leave=True  # 爬虫结束后保留进度条
                )
                self.logger.info(f"共检测到 {self.total_urls} 个有效爬取URL")
            else:
                self.logger.warning("没有可爬取的有效URL")
                return
        
        except Exception as e:
            self.logger.error(f"读取输入CSV失败: {str(e)}")
            return
        
        # 第二步：生成爬取请求
        for url, title in url_list:
            yield Request(
                url=url,
                callback=self.parse_page,
                meta={'title': title},
                errback=self.handle_request_error
            )
    
    def handle_request_error(self, failure):
        """处理请求错误，并更新进度条"""
        request = failure.request
        self.logger.error(f"请求失败: {request.url}, 错误: {str(failure.value)}")
        
        # 记录错误数据到输出CSV
        self.save_to_csv(
            title=request.meta.get('title', '未获取到标题'),
            cleaned_content='请求失败',
            cleaned_comments='请求失败'
        )
    
    def parse_page(self, response):
        """解析网页内容，提取正文和评论"""
        title = response.meta.get('title', '未获取到标题')
        self.logger.info(f"开始解析: {title} - {response.url}")
        
        try:
            # 1. 提取预加载的JSON数据（核心数据来源）
            preloaded_data = response.css('#data-preloaded::attr(data-preloaded)').get()
            if not preloaded_data:
                self.logger.warning(f"未找到预加载数据: {response.url}")
                self.save_to_csv(title, '未获取到正文内容', '未获取到评论内容')
                return
            
            # 2. 解析JSON数据
            preloaded_dict = json.loads(preloaded_data)
            topic_data = None
            
            # 遍历找到topic数据
            for key, value in preloaded_dict.items():
                if key.startswith('topic_') and isinstance(value, str):
                    try:
                        topic_data = json.loads(value)
                        break
                    except json.JSONDecodeError:
                        continue
            
            if not topic_data:
                self.logger.warning(f"未解析到topic数据: {response.url}")
                self.save_to_csv(title, '未获取到正文内容', '未获取到评论内容')
                return
            
            # 3. 提取正文内容（第一个post的cooked字段）
            posts = topic_data.get('post_stream', {}).get('posts', [])
            content = ''
            if posts:
                raw_content = posts[0].get('cooked', '')
                content = self.clean_text(raw_content)
            
            # 4. 提取评论内容（排除第一个post和system消息）
            comments = []
            for post in posts[1:]:  # 跳过第一个正文post
                # 过滤system用户的消息
                if post.get('username') == 'system':
                    continue
                
                # 提取评论用户名和内容
                username = post.get('username', '未知用户')
                raw_comment = post.get('cooked', '')
                cleaned_comment = self.clean_text(raw_comment)
                
                if cleaned_comment and cleaned_comment != '未获取到信息':
                    comments.append(f"{username}: {cleaned_comment}")
            
            # 5. 处理评论为空的情况
            comments_str = '|||'.join(comments) if comments else '未获取到评论内容'
            
            # 6. 保存到CSV并更新进度条
            self.save_to_csv(title, content, comments_str)
            self.logger.info(f"解析完成: {title} - 评论数: {len(comments)}")
            
        except Exception as e:
            self.logger.error(f"解析页面失败 {response.url}: {str(e)}")
            self.save_to_csv(title, '解析失败', '解析失败')
    
    def clean_text(self, text):
        """
        清洗文本：
        1. 移除HTML标签
        2. 过滤特殊字符
        3. 处理空值
        """
        if not text or text.strip() == '':
            return '未获取到信息'
        
        # 1. 移除HTML标签
        html_pattern = r'<[^>]+>'
        text_without_html = re.sub(html_pattern, '', text)
        
        # 2. 移除转义字符
        text_unescaped = (
            text_without_html
            .replace('\\u003cp\\u003e', '')
            .replace('\\u003c/p\\u003e', '\n')
            .replace('\\u003cbr\\u003e', '\n')
            .replace('\\u003ccode\\u003e', '`')
            .replace('\\u003c/code\\u003e', '`')
            .replace('\\u003cpre\\u003e', '\n')
            .replace('\\u003c/pre\\u003e', '\n')
            .replace('\\n', '\n')
            .replace('\\', '')
        )
        
        # 3. 过滤特殊字符
        cleaned_text = re.sub(self.ALLOWED_CHARS_PATTERN, '', text_unescaped)
        
        # 4. 清理多余空格和换行
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
        
        # 5. 处理空值
        return cleaned_text if cleaned_text else '未获取到信息'
    
    def save_to_csv(self, title, cleaned_content, cleaned_comments):
        """保存数据到CSV文件（支持增量写入），并线程安全更新进度条"""
        # 检查输出目录是否存在
        output_dir = os.path.dirname(self.output_csv_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 检查文件是否已存在，不存在则写入表头
        file_exists = os.path.exists(self.output_csv_path)
        
        with open(self.output_csv_path, 'a', encoding='utf-8-sig', newline='') as f:
            writer = csv.writer(f)
            
            # 写入表头（仅第一次）
            if not file_exists:
                writer.writerow(['标题', '清洗后的正文内容', '清洗后的评论内容列表'])
            
            # 写入数据行
            writer.writerow([
                title if title else '未获取到标题',
                cleaned_content if cleaned_content else '未获取到正文内容',
                cleaned_comments if cleaned_comments else '未获取到评论内容'
            ])
        
        # 线程安全更新进度条
        with self.lock:
            self.completed_urls += 1
            if self.pbar is not None:
                self.pbar.update(1)
                # 实时更新进度条描述（显示已完成/总数）
                self.pbar.set_description(f"爬取进度 {self.completed_urls}/{self.total_urls}")
    
    def closed(self, reason):
        """爬虫结束时关闭进度条并输出统计信息"""
        # 关闭进度条
        if self.pbar is not None:
            self.pbar.close()
        
        self.logger.info(f"\n爬虫结束，原因: {reason}")
        self.logger.info(f"最终数据文件: {os.path.abspath(self.output_csv_path)}")
        
        # 统计输出文件的行数
        if os.path.exists(self.output_csv_path):
            with open(self.output_csv_path, 'r', encoding='utf-8-sig') as f:
                row_count = sum(1 for _ in f) - 1  # 减去表头行
                self.logger.info(f"共生成 {row_count} 条有效记录")
        else:
            self.logger.info("未生成任何有效记录")