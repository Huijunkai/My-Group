# import pandas as pd
# 把数据写入文件
# with open('data.txt', 'a+', encoding='utf-8') as f:
name = '张三'
gender = '男'

import mysql.connector as mysql

conn = mysql.connect(
    host='localhost',       # 主机名
    user='root',            # 用户名
    password='121902',      # 密码
    database='my_data',     # 数据库名      
    charset='utf-8'       # 编码格式
)

cursor = conn.cursor()

sql = 'insert into test (name, gender) values ("'+name+'", "'+gender+'")'
cursor.execute(sql)
conn.commit()
