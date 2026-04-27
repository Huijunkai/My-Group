'
使用 sleepstudy 数据集（研究睡眠剥夺对反应时间的影响）。
(1)建立线性混合模型：反应时间 Reaction 对天数 Days 固定效应，被试 Subject 为随机截距。
(2)再添加随机斜率（允许每个被试的 Days 效应不同），比较两个模型。
'

if(!require(lme4)) install.packages('lme4'); library(lme4)

# 加载数据
data(sleepstudy)
print(sleepstudy)


# 建立线性混合模型：反应时间 Reaction 对天数 Days 固定效应，被试 Subject 为随机截距。
m1 = lmer(Reaction ~ Days + (1 | Subject), data = sleepstudy)
cat("========== m1：随机截距模型 ==========\n")
print(summary(m1))
# 再添加随机斜率（允许每个被试的days效应不同），比较两个模型。
m2 = lmer(Reaction ~ Days + (Days | Subject), data = sleepstudy)
cat("\n\n========== m2：随机截距+随机斜率模型 ==========\n")
print(summary(m2))

# 模型比较（关键：看哪个模型更好）
cat("\n\n========== 模型比较 ==========\n")
print(anova(m1, m2))