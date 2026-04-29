'
2.使用iris数据集，建立决策树预测物种
（1）训练随机森林（默认参数），输出OOB误差。
（2）绘制变量重要性图
（3）使用caret对mtry进行调优
'
if (!require("randomForest", quietly = TRUE)) {
  install.packages("randomForest")
  library(randomForest)
}
if (!require("caret", quietly = TRUE)) {
  install.packages("caret")
  library(caret)
}

set.seed(123)

# （1）训练随机森林（默认参数），输出OOB误差。
rf_default = randomForest(Species ~., data = iris)
print(rf_default)
# （2）绘制变量重要性图
importance(rf_default)
print(varImpPlot(rf_default))

# （3）使用caret对mtry进行调优
control = trainControl(method = 'cv', number = 5)
tuneGrid = expand.grid(mtry = 1:4)
rf_tune = train(Species ~., data = iris, method = 'rf', trControl = control, tuneGrid = tuneGrid)
rf_tune$bestTune
print(rf_tune)