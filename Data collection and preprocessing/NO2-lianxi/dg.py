import requests
import json
import threading
from queue import Queue
import time
import random

class FedoraCentosSpider:
    def __init__(self):
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        self.url_queue = Queue()
        self.content_queue = Queue()
        self.all_data = []

    def get_url_list(self, start_page, end_page):
        url_temp = "https://discussion.fedoraproject.org/c/neighbors/centos/71.json?page={}"
        for i in range(start_page, end_page + 1):
            url = url_temp.format(i)
            print(f'添加请求URL：{url}')
            self.url_queue.put(url)

    def get_data(self):
        while True:
            try:
                url = self.url_queue.get()
                print(f'\n正在请求：{url}')
                
                # 新增：随机延迟（反爬优化，避免请求过快）
                time.sleep(random.uniform(0.5, 1.5))
                
                response = requests.get(
                    url=url, 
                    headers=self.headers, 
                    timeout=15,
                    verify=False  # 可选：忽略SSL证书验证（部分环境可能需要）
                )
                
                # 关键调试：打印状态码和响应前500字符（看服务器真实返回）
                print(f"状态码：{response.status_code}")
                print(f"响应内容（前500字符）：{response.text[:500]}")
                
                response.raise_for_status()  # 4xx/5xx状态码直接抛异常
                response.encoding = response.apparent_encoding
                
                # 解析JSON，并严格校验类型
                try:
                    data = json.loads(response.text)
                except json.JSONDecodeError as e:
                    print(f'JSON解析失败：{e}，响应内容：{response.text}')
                    self.url_queue.task_done()
                    continue
                
                # 必须是字典类型才继续（解决核心错误）
                if not isinstance(data, dict):
                    print(f'错误：返回的JSON不是字典，而是 {type(data)} 类型，内容：{data}')
                    self.url_queue.task_done()
                    continue
                
                # 提取数据（适配真实结构）
                topic_list = data.get('topic_list', {})
                if not isinstance(topic_list, dict):
                    print(f'错误：topic_list不是字典，类型：{type(topic_list)}')
                    self.url_queue.task_done()
                    continue
                
                results = topic_list.get('topics', [])
                if not isinstance(results, list):
                    print(f'错误：topics不是列表，类型：{type(results)}')
                    self.url_queue.task_done()
                    continue
                
                # 提取有效数据
                content_li = []
                for item in results:
                    if not isinstance(item, dict):
                        continue  # 跳过非字典的无效数据
                    content = {
                        "标题": item.get("title", "无标题"),
                        "作者": item.get("created_by", {}).get("username", "匿名用户"),
                        "发布时间": item.get("created_at", "未知时间"),
                        "评论状态": "关闭" if item.get('closed', False) else "正常",
                        "总评论数": item.get('posts_count', 0) - 1,  # 减去作者自身回复
                        "查看次数": item.get('views', 0),
                        "标签": [tag.get("name") for tag in item.get("tags", []) if isinstance(tag, dict)]
                    }
                    content_li.append(content)
                
                if content_li:
                    self.content_queue.put(content_li)
                    print(f'请求成功：{url}，获取 {len(content_li)} 条数据')
                else:
                    print(f'请求成功：{url}，无有效数据（可能页面为空或无权限）')
                
            except requests.exceptions.RequestException as e:
                print(f'请求异常：{e}')
            except Exception as e:
                print(f'未知错误：{e}')
            finally:
                self.url_queue.task_done()

    def save_data(self):
        while True:
            try:
                content_list = self.content_queue.get()
                self.all_data.extend(content_list)
                print(f'暂存成功：{len(content_list)} 条数据（累计：{len(self.all_data)} 条）')
            except Exception as e:
                print(f'暂存失败：{e}')
            finally:
                self.content_queue.task_done()

    def final_save(self):
        with open('fedora_centos_topics.json', mode='w', encoding='utf-8') as f:
            json.dump(self.all_data, f, ensure_ascii=False, indent=2)
        print(f'\n最终保存成功：共 {len(self.all_data)} 条数据，文件：fedora_centos_topics.json')

    def run(self):
        try:
            start_page = int(input('请输入抓取的起始页：'))
            end_page = int(input('请输入抓取的结束页：'))
            
            # 页码校验
            start_page = max(1, start_page)
            end_page = max(start_page, end_page)

            t_list = []

            # 1. URL生成线程
            t_url = threading.Thread(target=self.get_url_list, args=(start_page, end_page))
            t_list.append(t_url)

            # 2. 爬取线程（减少到2个，降低反爬风险）
            for _ in range(2):
                t_content = threading.Thread(target=self.get_data)
                t_list.append(t_content)

            # 3. 保存线程
            t_save = threading.Thread(target=self.save_data)
            t_list.append(t_save)

            # 启动线程
            for t in t_list:
                t.setDaemon(True)
                t.start()

            # 等待队列完成
            for q in [self.url_queue, self.content_queue]:
                q.join()

            # 最终保存
            self.final_save()
            print('=' * 50)
            print('所有爬取任务完成！')

        except ValueError:
            print('错误：请输入有效的数字页码！')
        except Exception as e:
            print(f'程序运行出错：{e}')

if __name__ == '__main__':
    # 禁用requests的SSL警告（可选，避免环境报错）
    requests.packages.urllib3.disable_warnings()
    spider = FedoraCentosSpider()
    spider.run()