'
练习题 7（新数据集）：神经网络分类 —— 使用 Sonar 数据集
数据集说明
mlbench::Sonar 包含 208 个样本，60 个数值特征（声纳回波强度），目标变量 Class 为 "R"（岩石）或 "M"（金属圆柱体）。这是一个二分类问题。
题目
使用 nnet 对 Sonar 数据集进行分类。要求：
（1）数据划分：按 70% 训练、30% 测试划分。
（2）训练网络：设置隐藏层节点数 size = 5，最大迭代次数 maxit = 200。
（3）预测并计算准确率。
（4）尝试不同的 size（3, 5, 10）和 decay（0, 0.01, 0.1）组合，比较测试集准确率，找出最优参数组合。
'

library(mlbench)
library(nnet)
library(caret)

data(Sonar)

set.seed(123)
train_index = createDataPartition(Sonar$Class, p = 0.7, list = FALSE)
train_data = Sonar[train_index, ]
test_data = Sonar[-train_index, ]

model = nnet(Class ~ ., data = train_data, size = 5, maxit = 200, trace = FALSE)

pred = predict(model, test_data, type = "class")
accuracy = sum(pred == test_data$Class) / nrow(test_data)
cat("准确率: ", accuracy, "\n")

sizes = c(3, 5, 10)
decays = c(0, 0.01, 0.1)
results = data.frame(size = numeric(), decay = numeric(), accuracy = numeric())

for (size in sizes) {
  for (decay in decays) {
    model = nnet(Class ~ ., data = train_data, size = size, decay = decay, maxit = 200, trace = FALSE)
    pred = predict(model, test_data, type = "class")
    acc = sum(pred == test_data$Class) / nrow(test_data)
    results = rbind(results, data.frame(size = size, decay = decay, accuracy = acc))
  }
}

cat("\n参数组合结果:\n")
print(results)

best_index = which.max(results$accuracy)
cat("\n最优参数组合:\n")
print(results[best_index, ])