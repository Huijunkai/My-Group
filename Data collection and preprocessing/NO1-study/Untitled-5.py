# # 模拟登录的课程学习

# import requests as req
# from lxml import etree

# headers = {
#     'Cookie' : 'll="118302"; bid=GmUemmMKVUA; _vwo_uuid_v2=D8D6011DB1CFB180005FA9A7EB6C1E29D|9f8059e35747060f1b2beb84cb0039e7; _pk_ref.100001.8cb4=%5B%22%22%2C%22%22%2C1761010299%2C%22https%3A%2F%2Fcn.bing.com%2F%22%5D; _pk_id.100001.8cb4=255c0356edb72cae.1761010299.; _pk_ses.100001.8cb4=1; __utma=30149280.1858202652.1757987432.1758588251.1761010300.8; __utmc=30149280; __utmz=30149280.1761010300.8.2.utmcsr=cn.bing.com|utmccn=(referral)|utmcmd=referral|utmcct=/; __utmt=1; dbcl2="291921244:KeJdUxF/rjI"; ck=j7Mx; ap_v=0,6.0; frodotk_db="a05ba9cdcd0d7821e112fd6967286863"; __yadk_uid=xf4es3hQ5TbUBEMZ7coEzdUYYEnMrpSw; push_noty_num=0; push_doumail_num=0; __utmv=30149280.29192; __utmb=30149280.18.10.1761010300',
#     'User-Agent' : 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36 Edg/142.0.0.0'
# }

# res = req.get('https://www.douban.com/people/291921244/?_i=1010441GmUemmM', headers=headers)

# # print(res.text)


# # 三种解析方式
# """
# 1. 正则表达式
# 2. XPath
# 3. bs4
# """
# # XPath解析
# html = etree.HTML(res.text)

# title = html.xpath('//*[@id="note"]/div/div[1]/div[1]/a/text()')
# print(title)


from selenium import webdriver
from selenium.webdriver.edge.options import Options
from selenium.webdriver.common.by import By


options = Options()
options.add_experimental_option('detach', True)  # 让浏览器不自动关闭
# options.add_argument(
#     "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36",
#     "Cookie=tt_webid=7563849896224081449; gfkadpd=24,6457; ttcid=0b2c9410a2b046ecb0bc698662673e9e41; s_v_web_id=verify_mh1b4bd9_Ij7ISTR4_HRZM_4Px0_Bzjm_gZzjBWQvUTm2; local_city_cache=%E5%8D%97%E5%AE%81; ttwid=1%7CkGbkxbO04TpT_t1LqTnn72rbep-U4FVxVQmn6-WsxVs%7C1761096052%7C09c93c0f0ca035cc205b9b3ba31b49a2be4a6edc3e4bb0c64f5e2c989447b97b; tt_scid=89TETJF5BTEHrpHhwf0ZIX1NoLl.ykwaJom.n8fqvAlhvpuvc.HuwNyhBwvDrbDz531a; x-web-secsdk-uid=0ec13115-c221-44d2-bacd-677a19f2bba4; csrftoken=74b092b979df1726041b86b7ff394b52; _ga_QEHZPBE5HH=GS2.1.s1761096053$o1$g0$t1761096053$j60$l0$h0; _ga=GA1.1.483992183.1761096053"
# )

driver = webdriver.Edge(options=options)


driver.get("https://www.baidu.com")

# print(driver.page_source)
# # 通过ID查找文本框
# textarea = driver.find_element(By.ID, 'chat-textarea')    
# # 在文本框中输入查找内容
# textarea.send_keys('python')
# # 通过ID查找发送按钮 id="chat-submit-button"
# subBit = driver.find_element(By.ID, 'chat-submit-button')
# # 点击提交按钮查找内容
# subBit.click()

# 获取热点新闻
""" 方式一 """
lis = driver.find_elements(By.CLASS_NAME,'hotsearch-item')
for li in lis:
    title = li.find_element(By.CLASS_NAME, 'title-content-title').text
    print(title)

""" 方式二 """
# titles = driver.find_elements(By.CLASS_NAME, 'title-content-title')
# for title in titles:
#     print(title.text)
# print(len(title.text))

""" 方式三 （不推荐）"""
# titles = [title.text for title in driver.find_elements(By.CLASS_NAME, 'title-content-title')]
# print(titles)
# print(len(titles))