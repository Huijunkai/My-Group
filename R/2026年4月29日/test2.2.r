# 2. 使用 BostonHousing 数据集构建随机森林回归模型
# 背景：BostonHousing 数据集位于 MASS 包中，包含美国波士顿 506 个社区的房价中位数（medv）及其影响因素

if(!require(randomForest)) install.packages("randomForest"); library(randomForest)
if(!require(MASS)) install.packages("MASS"); library(MASS)
if(!require(caret)) install.packages("caret"); library(caret)

# （1）加载并预览 Boston 数据集
cat("\n\n=== 加载 Boston 数据集 ===\n")
data(Boston)
cat("数据集维度:", dim(Boston), "\n")
cat("数据集预览:\n")
print(head(Boston))

# （2）训练随机森林模型
cat("\n=== 训练随机森林模型 ===\n")
set.seed(42)
rf_model = randomForest(medv ~ ., data = Boston, importance = TRUE)

# （3）输出并解读袋外误差
cat("\n随机森林模型信息:\n")
print(rf_model)
cat("\n袋外误差解读:\n")
cat("OOB (Out-of-Bag) 误差是", rf_model$mse[length(rf_model$mse)], "\n")
cat("这表示使用未参与训练的袋外样本计算的均方误差\n")

# （4）绘制变量重要性图
cat("\n=== 变量重要性分析 ===\n")
print(varImpPlot(rf_model, main = "随机森林变量重要性", type = 2))


# （5）使用 caret 包进行超参数调优
cat("\n=== 超参数调优 ===\n")
set.seed(42)
train_control = trainControl(method = "cv", number = 5)
tune_grid = expand.grid(mtry = seq(1, ncol(Boston)-1, by = 1))

rf_tune = train(medv ~ ., data = Boston, 
                 method = "rf", 
                 trControl = train_control, 
                 tuneGrid = tune_grid,
                 importance = TRUE)

print(rf_tune)

# （6）输出最佳 mtry 值和模型性能
cat("\n最佳 mtry 值:", rf_tune$bestTune$mtry, "\n")
cat("调优后的模型性能:\n")
print(rf_tune$results[rf_tune$bestTune$mtry, ])

cat("\n最终结果:\n")
cat("最佳 mtry:", rf_tune$bestTune$mtry, "\n")
cat("RMSE:", rf_tune$results$RMSE[rf_tune$bestTune$mtry], "\n")
cat("R2:", rf_tune$results$Rsquared[rf_tune$bestTune$mtry], "\n")