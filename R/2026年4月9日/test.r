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

# ===================== 纯R语言实现：学生成绩数据分析 =====================
set.seed(123)  # 固定随机数，结果可复现
n = 100      # 100个学生

# 1. 生成数据
gender = sample(c("男", "女"), size = n, replace = TRUE, prob = c(0.5, 0.5))
class  = sample(c("A", "B", "C"), size = n, replace = TRUE)

# 生成成绩并限制在0-100之间，取整
math    = round(pmin(pmax(rnorm(n, mean = 75, sd = 10), 0), 100))
chinese = round(pmin(pmax(rnorm(n, mean = 78, sd = 12), 0), 100))
english = round(pmin(pmax(rnorm(n, mean = 80, sd = 11), 0), 100))

# 构建数据框
score_data = data.frame(gender, class, math, chinese, english)
cat("数据前6行预览：\n")
print(head(score_data))


# ===================== 任务(1) 三科成绩：均值、中位数、标准差 =====================
cat("\n====================================\n")
cat("(1) 数学、语文、英语成绩统计\n")
cat("====================================\n")

# 数学
cat("数学  均值：", mean(math), "\n")
cat("数学  中位数：", median(math), "\n")
cat("数学  标准差：", sd(math), "\n\n")

# 语文
cat("语文  均值：", mean(chinese), "\n")
cat("语文  中位数：", median(chinese), "\n")
cat("语文  标准差：", sd(chinese), "\n\n")

# 英语
cat("英语  均值：", mean(english), "\n")
cat("英语  中位数：", median(english), "\n")
cat("英语  标准差：", sd(english), "\n")


# ===================== 任务(2) 数学成绩直方图 + 核密度曲线 =====================
cat("\n正在生成图片1：数学成绩分布...\n")
hist(math, 
     prob = TRUE, 
     col = "lightblue", 
     main = "数学成绩直方图与核密度曲线", 
     xlab = "数学成绩")
lines(density(math), col = "red", lwd = 2)  # 核密度曲线


# ===================== 任务(3) 按班级绘制数学成绩箱线图 =====================
cat("正在生成图片2：班级数学成绩箱线图...\n")
boxplot(math ~ class, 
        data = score_data, 
        col = c("pink","lightgreen","lightyellow"), 
        main = "不同班级数学成绩箱线图",
        xlab = "班级",
        ylab = "数学成绩")


# ===================== 任务(4) 散点图 + 拟合线 + 相关系数 =====================
cor_value = cor(math, chinese)
cat("\n(4) 数学与语文成绩相关系数：", round(cor_value, 3), "\n")

cat("正在生成图片3：数学与语文散点图...\n")
plot(math, chinese,
     main = paste("数学与语文成绩散点图（相关系数=", round(cor_value,3),")"),
     xlab = "数学成绩",
     ylab = "语文成绩",
     pch = 16)
abline(lm(chinese ~ math), col = "red", lwd = 2)  # 线性拟合线