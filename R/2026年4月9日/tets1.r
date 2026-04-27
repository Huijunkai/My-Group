'
第3题：模拟学生成绩数据——多科成绩分析与班级差异
数据：模拟生成一个包含100个学生的数据集，变量如下：
gender：性别（"男","女"），比例各半。
class：班级（"A","B","C"），均匀分布。
math：数学成绩（正态分布，均值75，标准差10，范围0~100）。
chinese：语文成绩（正态分布，均值78，标准差12，范围0~100）。
english：英语成绩（正态分布，均值80，标准差11，范围0~100）。

任务：
（1）计算数学、语文、英语三科成绩的均值、中位数、标准差。
（2）绘制数学成绩的直方图，并添加核密度曲线。
（3）按班级 class 绘制数学成绩的箱线图，并比较不同班级数学成绩的差异。
（4）绘制数学成绩与语文成绩的散点图，添加线性拟合线，并计算二者的相关系数。
'

library(dplyr)
library(ggplot2)

# 1. 生成100名学生模拟数据
n = 100
stus = data.frame(
  gender = sample(c("男", "女"), size = n, replace = TRUE, prob = c(0.5, 0.5)),
  class = sample(c("A", "B", "C"), size = n, replace = TRUE),
  
  math = pmax(pmin(round(rnorm(n, mean = 75, sd = 10)), 100), 0),
  chinese = pmax(pmin(round(rnorm(n, mean = 78, sd = 12)), 100), 0),
  english = pmax(pmin(round(rnorm(n, mean = 80, sd = 11)), 100), 0)
)

# ======================
# （1）三科成绩统计量
# ======================
stat_result = stus %>% summarise(
  math_mean = mean(math),
  math_median = median(math),
  math_sd = sd(math),
  
  chinese_mean = mean(chinese),
  chinese_median = median(chinese),
  chinese_sd = sd(chinese),
  
  english_mean = mean(english),
  english_median = median(english),
  english_sd = sd(english)
)
print("===== 三科成绩统计量 =====")
print(stat_result, digits = 3)

# ======================
# （2）数学成绩直方图 + 核密度曲线
# ======================
p1 = ggplot(stus, aes(x = math)) +
  geom_histogram(aes(y = after_stat(density)), bins = 12, 
                 fill = "lightgreen", color = "black") +
  geom_density(color = "blue", linewidth = 1) +
  labs(title = "数学成绩分布直方图", x = "数学成绩", y = "密度") +
  theme_minimal()
print(p1)

# ======================
# （3）按班级绘制数学成绩箱线图
# ======================
p2 = ggplot(stus, aes(x = class, y = math, fill = class)) +
  geom_boxplot() +
  labs(title = "不同班级数学成绩箱线图", x = "班级", y = "数学成绩") +
  theme_minimal()
print(p2)

# ======================
# （4）数学 vs 语文 散点图 + 拟合线 + 相关系数
# ======================
# 计算相关系数
cor_value = cor(stus$math, stus$chinese)
cat("\n数学与语文成绩相关系数 =", round(cor_value, 3), "\n")

# 绘图
p3 = ggplot(stus, aes(x = math, y = chinese)) +
  geom_point(color = "darkred", alpha = 0.7) +
  geom_smooth(method = "lm", se = FALSE, color = "blue", linewidth = 1) +
  labs(title = "数学-语文成绩散点图",
       x = "数学成绩", y = "语文成绩",
       subtitle = paste("相关系数 =", round(cor_value, 3))) +
  theme_minimal()
print(p3)