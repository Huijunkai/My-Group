from scrapy import Spider
from bookScrapy.items import BookscrapyItem


class BookSpider(Spider):
    name = 'book'
    start_urls = ['https://books.toscrape.com/']
    for i in range(1, 51):
        start_urls.append(f'https://books.toscrape.com/catalogue/page-{i+1}.html')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.book_count = 1

    def parse(self, response):
        # 定位所有书籍节点
        books = response.xpath('//*[@id="default"]/div/div/div/div/section/div[2]/ol/li')

        for book in books:
            # 书名
            name = book.xpath('article/h3/a/@title').get() or book.xpath('article/h3/a/text()').get()
            # 星级
            star = book.xpath('article/p[contains(@class, "star-rating")]/@class').get()
            star = star.split()[-1] if star else '未知'
            # 价格 
            price = book.xpath('article/div[2]/p[@class="price_color"]/text()').get()
            price = price.strip() if price else '未知'
            # 封面图片网址
            pic_url = book.xpath('article/div[1]/a/img/@src').get()
            pic_url = response.urljoin(pic_url) if pic_url else '未知'
            # 书籍详情页网址
            detail_url = book.xpath('article/h3/a/@href').get()
            detail_url = response.urljoin(detail_url) if detail_url else '未知'

            print(f"===== 第 {self.book_count} 本书 =====")
            print(f"书名：{name}")
            print(f"星级：{star}")
            print(f"价格：{price}")
            print(f"封面图片：{pic_url}")
            print(f"详情页：{detail_url}")
            print("==" * 50)
            self.book_count += 1



            # 创建item对象
            item = BookscrapyItem()
            item['name'] = name
            item['star'] = star
            item['price'] = price
            item['pic_url'] = pic_url
            item['detail_url'] = detail_url

            # 把item对象yield到管道文件持久化
            yield item