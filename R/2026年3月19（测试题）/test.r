# 创建一个数据框grades，包含'student_id','name','math','english','science'五列，
# 分别表示学生ID、姓名、数学成绩、英语成绩和科学成绩

# 安装依赖包（仅在未安装时）
if (!requireNamespace("dplyr", quietly = TRUE)) install.packages("dplyr")

# 加载dplyr包
library(dplyr)

grades = data.frame(
    student_id = c(1,2,3,4,5),
    name       = c('A1','B1','C1','D1','E1'),
    math       = c(90, 80, 70, 60, 50), #数学
    english    = c(85, 75, 65, 55, 45), #英语
    science    = c(95, 85, 75, 65, 55)  #科学
)

print(grades)

# 1.在数据框中新增total_score，计算每个学生的总分（使用 dplyr，正确地给整个数据框赋值）
grades = grades %>% mutate(total_score = math + english + science)
print(grades)

# 2.筛选出数学成绩 ≥ 90 或总分 ≥ 260 ，输出姓名和数学成绩和总分
# filtered_grades = grades[grades$math >= 90 | grades$total_score >= 260, c('name', 'math', 'total_score')]
# print(filtered_grades)

select_data = grades %>% filter(math >=90 | total_score >= 260) %>% select(name, math, total_score)
print(select_data)


# 3.按总分的高低对数据框进行排序，并保留前3名学生的所有消息
top3 = grades %>% arrange(desc(total_score)) %>% head(3)
