import requests
import json
import threading
from queue import Queue
import time
class HeiMaTouTiao:
    def __init__(self):
        self.headers = {
           "User-Agent": "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36"
        }
        self.url_queue=Queue()
        self.content_queue=Queue()
    def get_url_list(self, start_page, end_page):
        url_temp = 'https://discussion.fedoraproject.org/c/neighbors/centos/71.json?page={}'
        url_list = [url_temp.format(i) for i in range(start_page, end_page + 1)]
        for url in url_list:
            print('正在请求：',url)
            self.url_queue.put(url)

    def get_data(self):
        content_li=[]
        while True:
            url = self.url_queue.get()
            comment=requests.get(url=url, headers=self.headers).text
            data=comment.loads(comment) #用错了loads方法
            data=data['data']['results']
            for index in range(len(data)):
                content=dict()
                content["标题"] = data[index]["title"]
                if data[index]['comment_status'] is True:
                    content['评论状态']='正常'
                else:
                    content['评论状态']='关闭'
                content['总评论数']=data[index]['total_comment_count']
                content['粉丝评论数']=data[index]['fans_comment_count']
                content_li.append(content)
            self.content_queue.put(content_li)
            self.url_queue.task_done()
    def save_data(self):
            while True:
                content_list = self.content_queue.get()
                with open('toutiao.json', mode='a+', encoding='utf-8') as f:
                    f.write(json.dumps(content_list,ensure_ascii=False,indent=2))
                self.content_queue.task_done()
    def save_data(self):
        while True:
            content_list = self.content_queue.get()
            with open('toutiao.json', mode='a+', encoding='utf-8') as f:
                f.write(json.dumps(content_list,ensure_ascii=False,indent=2))
        self.content_queue.task_done()  #冗余代码
    def run(self):
        start_page=int(input('请输入抓取的起始页：'))
        end_page=int(input('请输入抓取的结束页：'))
        t_list=[]
        if start_page<=0:
            print('抓取的起始页从1开始。')
        else:
            t_url=threading.Thread(target=self.get_url_list,args=(start_page,end_page))
            t_list.append(t_url) #用错了append方法
        t_save=threading.Thread(target=self.save_data)
        t_list.append(t_save)
        for t in t_list:
            t.setDaemon(True)
            t.start()
        for q in [self.url_queue,self.content_queue]:
            q.join()
if __name__ == '__main__':
    heimatoutiao = HeiMaTouTiao()
    heimatoutiao.run()