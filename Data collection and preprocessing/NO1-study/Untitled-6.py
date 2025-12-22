from selenium import webdriver
from selenium.webdriver.edge.options import Options
from selenium.webdriver.common.by import By

options = Options()
options.add_experimental_option('detach', True)  # 让浏览器不自动关闭

driver = webdriver.Edge(options=options)
driver.get("https://www.baidu.com")

# for i in range(5):

titles = []
while(1):

    flag = False

    titleEles = driver.find_elements(By.CLASS_NAME, 'title-content-title')

    for li in titleEles:
        tname = li.text
        if tname in titles:
            flag = True
            break
        titles.append(tname)
        # title = li.find_element(By.CLASS_NAME, 'title-content-title').text
        print(tname)

    if flag:
        break

    # 获取刷新元素
    refresh = driver.find_element(By.CLASS_NAME, 'hot-refresh-text')
    # 设置点击事件
    refresh.click()
