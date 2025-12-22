from selenium import webdriver
 
# 创建一个 Edge WebDriver 实例
driver = webdriver.Edge()
 
# 打开一个网页
driver.get("https://www.baidu.com")
 
# 关闭浏览器
driver.quit()    