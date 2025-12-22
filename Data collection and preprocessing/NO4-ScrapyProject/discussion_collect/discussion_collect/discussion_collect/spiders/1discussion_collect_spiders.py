""" 
此爬虫项目用于爬取Fedora/CentOS官方讨论区的热门话题数据，
包括标题、标签、作者、评论数、阅读量、发布时间及作者头像等信息。
爬取目标为2000条数据，数据存储为CSV文件，头像图片单独存储在指定文件夹中。
采用Scrapy框架结合多线程方式实现高效爬取与下载，确保数据完整性与下载成功率。
"""
import csv
import os
import time
import requests
import threading
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED
from tqdm import tqdm
from scrapy import Spider, Request

class DiscussionSpider(Spider):
    name = 'discussion_collect_spiders'
    allowed_domains = ['discussion.fedoraproject.org']
    start_urls = []
    all_data = []
    target_count = 2000
    current_page = 1
    
    # 分文件夹存储：CSV和头像分开
    csv_dir = 'data/csv'          # CSV文件文件夹
    avatar_dir = 'data/avatars'   # 头像文件文件夹
    
    # 线程池配置
    thread_pool = None
    max_threads = 12
    download_futures = []
    
    # 头像下载统计（线程安全）
    total_avatar_tasks = 0
    successful_avatar_downloads = 0
    lock = threading.Lock()    # 线程锁保证计数安全
    
    # tqdm进度条对象
    data_pbar = None  # 数据爬取进度条
    avatar_pbar = None  # 头像下载进度条

    # Scrapy配置：过滤所有非必要日志，仅保留警告/错误
    custom_settings = {
        'CONCURRENT_REQUESTS': 16,
        'CONCURRENT_REQUESTS_PER_DOMAIN': 8,
        'DOWNLOAD_DELAY': 0.2,
        'USER_AGENT': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'LOG_LEVEL': 'WARNING',  # 仅输出WARNING及以上日志（过滤Scrapy默认INFO日志）
        'LOG_STATS_INTERVAL': 0,  # 彻底关闭Scrapy进度统计日志
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 初始化tqdm进度条（在start_requests中实际创建）
        self.data_pbar = None
        self.avatar_pbar = None

    def start_requests(self):
        # 创建分离子文件夹（CSV和头像）
        for dir_path in [self.csv_dir, self.avatar_dir]:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
                self.logger.warning(f'创建文件夹成功：{dir_path}')  # 用WARNING级别确保显示
        
        # 初始化线程池
        self.thread_pool = ThreadPoolExecutor(max_workers=self.max_threads)
        self.logger.warning(f'初始化线程池，最大线程数：{self.max_threads}')  # 用WARNING级别确保显示

        # 初始化tqdm进度条（和上一个爬虫样式一致）
        self.data_pbar = tqdm(
            total=self.target_count,
            desc="数据爬取进度",
            unit="条",
            ncols=80,  # 进度条宽度
            colour="green",  # 绿色进度条
            leave=True  # 爬虫结束后保留进度条
        )
        self.avatar_pbar = tqdm(
            total=0,  # 初始总数为0，后续动态更新
            desc="头像下载进度",
            unit="张",
            ncols=80,
            colour="blue",  # 蓝色区分头像进度
            leave=True
        )
        self.url_template = "https://discussion.fedoraproject.org/top?page={}&per_page=50&period=all"
        first_page_url = self.url_template.format(self.current_page)
        yield Request(
            url=first_page_url,
            callback=self.parse_topic_list,
            errback=self.handle_error,
            dont_filter=True
        )

    def clean_number(self, text):
        if not text:
            return ''
        return text.strip()
    
    def update_progress(self):
        """线程安全更新进度条显示"""
        with self.lock:
            # 更新数据进度条（当前爬取条数）
            self.data_pbar.n = len(self.all_data)
            self.data_pbar.refresh()  # 强制刷新显示
            
            # 更新头像进度条总数（动态调整）
            self.avatar_pbar.total = self.total_avatar_tasks
            self.avatar_pbar.refresh()

    def download_avatar(self, avatar_url, author_name, topic_index):
        """头像下载核心方法（线程池执行）"""
        try:
            if not avatar_url:
                self.update_progress()
                return None
            
            # 处理作者名称，避免文件名非法字符
            safe_author_name = "".join([c for c in author_name if c.isalnum() or c in (' ', '-', '_')]).strip()
            if not safe_author_name:
                safe_author_name = f"unknown_{topic_index}"
            
            # 生成文件名
            file_ext = avatar_url.split('.')[-1]
            if len(file_ext) > 4:  # 防止url参数被当作扩展名
                file_ext = 'png'
            filename = f"{safe_author_name}_{topic_index}.{file_ext}"
            filepath = os.path.join(self.avatar_dir, filename)
            
            # 请求头像图片
            headers = {
                'User-Agent': self.custom_settings['USER_AGENT']
            }
            response = requests.get(
                avatar_url, 
                headers=headers, 
                timeout=10,
                allow_redirects=True,
                stream=True
            )
            response.raise_for_status()
            
            # 分块保存图片
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # 线程安全更新成功数和进度条
            with self.lock:
                self.successful_avatar_downloads += 1
                self.avatar_pbar.update(1)  # 头像进度条+1
            self.update_progress()
            return filepath
        except Exception as e:
            # 仅输出关键错误（WARNING级别，避免干扰）
            self.logger.warning(f'头像下载失败 | 作者：{author_name} | 错误：{str(e)[:100]}')
            self.update_progress()
            return None

    def parse_topic_list(self, response):
        if len(self.all_data) >= self.target_count:
            return

        try:
            topic_rows = response.xpath('//tr[contains(@class, "topic-list-item") or @data-topic-id]')
            if not topic_rows:
                topic_rows = response.xpath('//table//tbody/tr')

            for row_idx, row in enumerate(topic_rows):
                if len(self.all_data) >= self.target_count:
                    break

                # 标题提取
                title_text = row.xpath('.//td//a[contains(@href, "/t/") and not(contains(@class, "badge-category"))]/text()').extract_first()
                title_href = row.xpath('.//td//a[contains(@href, "/t/")]/@href').extract_first()
                if not title_text and title_href and '/t/' in title_href:
                    title_text = title_href.split('/t/')[-1].split('/')[0].replace('-', ' ')
                title_value = title_text.strip() if title_text else ''

                # 拼接完整的详情页URL（仅提取，不请求）
                detail_url = ''
                if title_href:
                    if title_href.startswith('http'):
                        detail_url = title_href
                    else:
                        detail_url = response.urljoin(title_href)

                # 标签提取
                tags = row.xpath('.//a[contains(@class, "discourse-tag") or contains(@class, "tag")]/text()').extract()
                tags = [tag.strip() for tag in tags if tag.strip()]
                tag_value = ', '.join(tags) if tags else ''

                # 作者提取
                author_title = row.xpath('.//img[contains(@class, "avatar")]/@title').extract_first()
                author_value = ''
                if author_title:
                    author_value = author_title.split(' - ')[0].strip() if ' - ' in author_title else author_title.strip()
                if not author_value:
                    author_text = row.xpath('.//a[contains(@href, "/u/")]/text()').extract_first()
                    author_value = author_text.strip() if author_text else ''

                # 头像URL提取
                avatar_url = row.xpath('.//img[contains(@class, "avatar")]/@src').extract_first()
                if avatar_url and not avatar_url.startswith('http'):
                    avatar_url = response.urljoin(avatar_url)
                
                # 评论数提取
                comment_count = row.xpath('.//td[contains(@class, "replies") or position()=3]//span[contains(@class, "number")]/text()').extract_first()
                if not comment_count:
                    comment_text = row.xpath('.//td[contains(@class, "replies") or position()=3]//text()').extract()
                    for text in comment_text:
                        if text.strip():
                            comment_count = text
                            break
                comment_count = self.clean_number(comment_count)

                # 阅读量提取
                view_count = row.xpath('.//td[contains(@class, "views") or position()=4]//span[contains(@class, "number")]/text()').extract_first()
                if not view_count:
                    view_text = row.xpath('.//td[contains(@class, "views") or position()=4]//text()').extract()
                    for text in view_text:
                        if text.strip():
                            view_count = text
                            break
                view_count = self.clean_number(view_count)

                # 发布时间提取
                time_stamp = row.xpath('.//span[contains(@class, "relative-date")]/@data-time').extract_first()
                publish_time = ''
                if time_stamp and time_stamp.isdigit():
                    publish_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(int(time_stamp)))
                if not publish_time:
                    time_text = row.xpath('.//span[contains(@class, "relative-date")]/text()').extract_first()
                    publish_time = time_text.strip() if time_text else ''
                if not publish_time:
                    time_text = row.xpath('.//td[contains(@class, "created") or position()=5]//text()').extract()
                    time_text = [t.strip() for t in time_text if t.strip() and not t.strip().isdigit()]
                    publish_time = time_text[0] if time_text else ''

                # 组装数据（标题详细内容标注为"未解析详情页"）
                topic_info = {
                    "标题": title_value,
                    "标题详情URL": detail_url,
                    "标签": tag_value,
                    "作者": author_value,
                    "总评论数": comment_count,
                    "阅读量": view_count,
                    "发布时间": publish_time,
                    "作者头像URL": avatar_url if avatar_url else ""
                }

                # 直接加入数据列表（不再请求详情页）
                self.all_data.append(topic_info)
                
                # 更新头像任务数
                with self.lock:
                    self.total_avatar_tasks += 1
                
                # 提交头像下载任务
                current_topic_index = len(self.all_data)
                future = self.thread_pool.submit(
                    self.download_avatar,
                    avatar_url,
                    author_value,
                    current_topic_index
                )
                self.download_futures.append(future)
                
                # 更新进度条
                self.update_progress()

            # 请求下一页
            if len(self.all_data) < self.target_count:
                self.current_page += 1
                next_page_url = self.url_template.format(self.current_page)
                yield Request(
                    url=next_page_url,
                    callback=self.parse_topic_list,
                    errback=self.handle_error,
                    dont_filter=True
                )

        except Exception as e:
            self.logger.error(f'列表页解析异常：{e} | 页码：{self.current_page}', exc_info=True)

    def handle_error(self, failure):
        request = failure.request
        self.logger.error(f'列表页请求失败：{request.url} | 错误类型：{str(failure.type)[:30]}', exc_info=False)
        if len(self.all_data) < self.target_count:
            self.current_page += 1
            next_page_url = self.url_template.format(self.current_page)
            yield Request(
                url=next_page_url,
                callback=self.parse_topic_list,
                errback=self.handle_error,
                dont_filter=True
            )

    def closed(self, reason):
        # 等待所有头像下载任务完成
        if self.download_futures:
            wait(self.download_futures, return_when=ALL_COMPLETED)
        
        # 关闭线程池
        if self.thread_pool:
            self.thread_pool.shutdown(wait=True)

        # 关闭并刷新进度条（确保最终状态显示）
        if self.data_pbar:
            self.data_pbar.n = len(self.all_data[:self.target_count])
            self.data_pbar.refresh()
            self.data_pbar.close()
        if self.avatar_pbar:
            self.avatar_pbar.refresh()
            self.avatar_pbar.close()

        # 计算最终统计数据
        data_total = len(self.all_data[:self.target_count])
        avatar_total = self.total_avatar_tasks
        avatar_success = self.successful_avatar_downloads
        avatar_fail = avatar_total - avatar_success
        avatar_success_rate = (avatar_success / avatar_total * 100) if avatar_total > 0 else 0

        # 最终统一输出统计信息（WARNING级别确保显示）
        self.logger.warning('\n' + '='*70)
        self.logger.warning(f'最终数据统计：')
        self.logger.warning(f'目标数据量：{self.target_count} | 实际爬取数据量：{data_total}')
        self.logger.warning(f'最终头像统计：')
        self.logger.warning(f'总下载任务数：{avatar_total} | 成功下载数：{avatar_success} | 失败下载数：{avatar_fail}')
        self.logger.warning(f'头像下载成功率：{avatar_success_rate:.1f}%')
        self.logger.warning('='*70)

        # 处理最终数据并保存
        final_data = self.all_data[:self.target_count]
        for idx, item in enumerate(final_data, start=1):
            item["编号"] = idx

        # 扩展CSV表头，保留标题详情URL（但内容标注为未解析）
        csv_headers = [
            "编号", "标题", "标题详情URL","标签", "作者", 
            "总评论数", "阅读量", "发布时间",
            "作者头像URL"
        ]

        try:
            save_path = os.path.join(self.csv_dir, 'fedora_centos_topics.csv')
            with open(save_path, mode='w', encoding='utf-8-sig', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=csv_headers)
                writer.writeheader()
                writer.writerows(final_data)
            
            self.logger.warning(f'文件保存路径：')
            self.logger.warning(f'CSV文件：{os.path.abspath(save_path)}')
            self.logger.warning(f'头像文件夹：{os.path.abspath(self.avatar_dir)}')
        except Exception as e:
            self.logger.error(f'保存CSV失败：{str(e)[:50]}', exc_info=False)