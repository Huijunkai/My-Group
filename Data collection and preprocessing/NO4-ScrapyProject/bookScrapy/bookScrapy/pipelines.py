# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html


# useful for handling different item types with a single interface
from itemadapter import ItemAdapter
from scrapy.pipelines.images import ImagesPipeline
from scrapy import Request
from scrapy.exceptions import DropItem
import logging


class BookscrapyPipeline:
    def process_item(self, item, spider):
        # 把item对象持久化到文件中
        with open('books.txt', 'a', encoding='utf-8') as f:
            f.write(f"===== 第 {spider.book_count - 1} 本书 =====\n")
            f.write(f"书名: {item['name']}\n")
            f.write(f"星级: {item['star']}\n")
            f.write(f"价格: {item['price']}\n")
            # 如果图片已下载，优先写入本地保存路径
            local_paths = item.get('image_paths')
            if local_paths:
                f.write(f"封面图片(本地)：{local_paths[0]}\n")
            else:
                f.write(f"封面图片: {item.get('pic_url')}\n")
            f.write(f"详情页: {item['detail_url']}\n")

        # videName = str(random.Random.random()) + '.mp4'
        # savePath ='./videos/' + videName
        # with open(savePath, 'wb') as f:
        #     f.write(item['video_content'])

        return item

logger = logging.Logger('MyImagesPipeline')
class MyImagesPipeline(ImagesPipeline):
    def get_media_requests(self, item, info):
        return Request(url=item['pic_url'])

    def item_completed(self, results, item, info):
        # 判断图片是否下载成功
        if not results[0][0]:
            raise DropItem("图片下载失败")
        logger.debug("图片下载成功")
        return item
    
    def file_path(self, request, response=None, info=None, *, item):
        filename = request.url.split('/')[-1]
        return filename