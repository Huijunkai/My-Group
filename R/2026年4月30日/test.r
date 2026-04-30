'
3.使用mtcars预测mpg（回归）。
（1）用gbm训练模型，设置distribution = "gaussian"。
（2）输出变量相对重要性。
（3）用5折交叉验证寻找最优树的数量。
'
# 加载必要的包
if(!require(gbm)) install.packages("gbm"); library(gbm)


# 加载数据
data(mtcars)
print(mtcars)

set.seed(123) # 设置随机种子以确保结果可重复

# 1. 用gbm训练模型，设置distribution = "gaussian"
gbm_model = gbm(mpg ~., data = mtcars, distribution = "gaussian", n.trees = 1000,
                interaction.depth = 3, shrinkage = 0.1, n.minobsinnode = 5)
print(summary(gbm_model))

# 2. 输出变量相对重要性
print(summary(gbm_model))


# 3. 用5折交叉验证寻找最优树的数量
gbm_cv = gbm(mpg ~., data = mtcars, distribution = "gaussian", n.trees = 1000,
              interaction.depth = 3, shrinkage = 0.1, n.minobsinnode = 5,
              cv.folds = 5)
best_trees = gbm.perf(gbm_cv, method = "cv")
print(paste("最优树的数量:", best_trees))