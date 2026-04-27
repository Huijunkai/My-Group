'
练习题 1：学生成绩 CSV 文件读取与清洗
题目描述
现有文件 students.csv，包含以下字段：
student_id：学号
name：姓名
math：数学成绩（可能含缺失值 NA）
english：英语成绩（可能含缺失值 NA）
science：科学成绩（可能含缺失值 NA）

请完成以下任务：

（1）使用 readr::read_csv() 读取该文件。
（2）将所有缺失的成绩替换为对应科目的平均分（四舍五入保留整数）。
（3）计算每个学生的总分 total_score，并保留到数据框中。
（4）将处理后的数据写入新的 CSV 文件 students_clean.csv。
'

library(readr)
library(dplyr)

#（1）使用 readr::read_csv() 读取该文件。
stus = read_csv('F:/My-study/R/students.csv')
print(stus)

#（2）将所有缺失的成绩替换为对应科目的平均分（四舍五入保留整数round(0)）。
stus = stus %>%
  mutate(
    math = ifelse(is.na(math), round(mean(math, na.rm = TRUE), 0), math),
    english = ifelse(is.na(english), round(mean(english, na.rm = TRUE), 0), english),
    science = ifelse(is.na(science), round(mean(science, na.rm = TRUE), 0), science)
  )
print(stus)


#（3）计算每个学生的总分 total_score，并保留到数据框中。
stus = stus%>%
  mutate(total_score = math + english + science)
print(stus)


#（4）将处理后的数据写入新的 CSV 文件 students_clean.csv。
write_csv(stus, 'F:/My-study/R/students_clean.csv')
