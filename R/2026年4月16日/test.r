'
1.使用mtcars数据集，建立mpg对wt，hp的多元线性回归模型。要求：
（1）输出模型摘要，解释系数意义。
（2）绘制4个诊断图，判断是否满足假设。
（3）添加wt:hp 交互项，用anova比较两个模型。
'

# 加载数据
data(mtcars)

print(mtcars)


# （1）输出模型摘要，解释系数意义。
# 使用基本模型
model1 = lm(mpg ~ wt + hp, data = mtcars)
print(summary(model1))

# （2）绘制4个诊断图，判断是否满足假设。
par(mfrow = c(2,2))
plot(model1)  # 第一张图


# （3）添加wt:hp 交互项，用anova比较两个模型。
model2 = lm(mpg ~ wt * hp, data = mtcars)
print(anova(model1, model2))
plot(model2)  # 第二张图

par(mfrow = c(1,1))