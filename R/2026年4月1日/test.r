'
练习题 2：多工作表 Excel 文件合并与分组汇总
题目描述
有一个 Excel 文件 sales.xlsx，包含两个工作表：
Sheet1：记录 2024 年销售数据，列：date, product, quantity
Sheet2：记录 2025 年销售数据，列：date, product, quantity

请完成以下任务：

（1）使用 readxl::read_excel(文件名，sheet="表名称") 分别读取两个工作表。
（2）将两个数据框纵向合并（追加）bind_rows(表1，表2)，并添加一列 year（根据数据来源标记 2024 或 2025）。
（3）计算每个产品在两年间的总销量group_by(产品) %>% summarise(计算总销售量)，并按照总销量降序排列。
（4）将汇总结果写入 Excel 文件 product_summary.xlsx（使用 writexl::write_xlsx(写入的数据，写入的文件名)）。
'
# install.packages('readxl')
# install.packages('writexl')
# install.packages('dplyr')
# install.packages('purrr')

library(readxl,dplyr)
library(writexl)
library(dplyr)

# （1）使用 readxl::read_excel(文件名，sheet="表名称") 分别读取两个工作表。
data_2024 = read_excel('R\\2026年4月1日\\sales.xlsx',sheet="2024年数据")
data_2025 = read_excel('R\\2026年4月1日\\sales.xlsx',sheet="2025年数据")
print(data_2024)
print(data_2024)

#（2）将两个数据框纵向合并（追加）bind_rows(表1，表2)，并添一列 year（根据数据来源标记 2024 或 2025）。
# 添加年份
data_2024 = data_2024 %>% mutate(year=2024)
data_2025 = data_2025 %>% mutate(year=2025)

# 合并两个数据表
datas = bind_rows(data_2024, data_2025)
print(datas)

#（3）计算每个产品在两年间的总销量group_by(产品) %>% summarise(计算总销售量)，并按照总销量降序排列。
product_summary = datas %>% 
  group_by(product) %>% 
  summarise(总销量 = sum(quantity)) %>% 
  arrange(desc(总销量))
print(product_summary) 

#（4）将汇总结果写入 Excel 文件 product_summary.xlsx（使用 writexl::write_xlsx(写入的数据，写入的文件名)）。
writexl::write_xlsx(product_summary, 'R\\2026年4月1日\\product_summary.xlsx')