import pandas as pd

#  读文件
file = pd.read_csv('Data collection and preprocessing/NO1-study/data clean/students.csv', encoding='utf-8')
# print(file)
# # 清洗数据
# file_cleaned = file.drop_duplicates(subset=['学号','姓名'], keep='first',inplace=True)

# print(file_cleaned)
# print(file)    

# 读取前两行
data_head = file.head(2)
print(data_head)

# 读取后两行
data_tail = file.tail(2)
print(data_tail)

# 查看整体数据类型信息
print(file.info())

# 查看数据的分布情况
print(file.describe())
# 查看数值类型
print(file.isna().sum())