'
练习题1（回归 + 随机森林）
使用 Boston 数据集（MASS包），以 medv（房价中位数）为响应变量，其余变量为预测变量。
（1）划分数据为70%训练集和30%测试集。
（2）使用随机森林（randomForest包）训练回归模型，设置 ntree=500，并计算变量重要性（importance=TRUE）。
（3）用训练好的模型预测测试集，计算均方根误差（RMSE）。
（4）通过5折交叉验证找到最优的 mtry 参数（候选值：2,4,6,8,10）。
'
# 加载必要的包
if(!require(MASS)) install.packages("MASS"); library(MASS)

# 加载数据
data(Boston)
set.seed(123) # 设置随机种子以确保结果可重复

# 1. 划分数据为70%训练集和30%测试集
idx = sample(1:nrow(Boston), size = 0.7 * nrow
(Boston))
train = Boston[idx, ]
test = Boston[-idx, ]
# 2. 使用随机森林（randomForest包）训练回归模型，设置 ntree=500，并计算变量重要性（importance=TRUE）
if(!require(randomForest)) install.packages("randomForest"); library(randomForest)
rf_model = randomForest(medv ~ ., data = train, ntree = 500
, importance = TRUE)
print(rf_model)
print(importance(rf_model))
# 3. 用训练好的模型预测测试集，计算均方根误差（RMSE）
predictions = predict(rf_model, newdata = test)
rmse = sqrt(mean((predictions - test$medv)^2))
print(paste("测试集RMSE:", rmse))
# 4. 通过5折交叉验证找到最优的 mtry 参数（候选值：2,4,6,8,10）
tune_result = tune.randomForest(medv ~ ., data = train, ntree =
500, mtry = c(2, 4, 6, 8, 10), importance = TRUE)
best_mtry = tune_result$best.parameters$mtry
print(paste("最优的 mtry 参数:", best_mtry))