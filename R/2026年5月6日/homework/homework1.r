"
练习题 5（新数据集）：KNN 分类 —— 使用 Sonar 数据集
数据集说明
mlbench::Sonar 包含 208 个样本，60 个数值特征（声纳回波强度），目标变量 Class 为 R（岩石）或 M（金属圆柱体）。这是一个经典的二分类问题。
题目
使用 Sonar 数据集，实现 KNN 分类。要求：
（1）数据标准化：对全部 60 个特征进行标准化（scale）。
（2）数据划分：按 70% 训练、30% 测试划分（分层抽样，保持类别比例）。
（3）交叉验证选 K：使用 5 折交叉验证（仅在训练集上）选择最佳 K 值（范围 1~21 的奇数）。
（4）最终评估：用最佳 K 值在测试集上评估，输出混淆矩阵、准确率、Kappa 值。
"
# 导入包
if(!require(class)) install.packages("class"); library(class)
if(!require(caret)) install.packages("caret"); library(caret)
if(!require(mlbench)) install.packages("mlbench"); library(mlbench)

set.seed(123)
# 加载数据
data(Sonar)
# （1）数据标准化：对全部 60 个特征进行标准化（scale）。
sonar_std = Sonar
sonar_std[, 1:60] = scale(Sonar[, 1:60])

# （2）数据划分：按 70% 训练、30% 测试划分（分层抽样，保持类别比例）。
train_index = createDataPartition(sonar_std$Class, p = 0.7, list = FALSE)
train_x = sonar_std[train_index, 1:60]
train_y = sonar_std[train_index, 61]
test_x = sonar_std[-train_index, 1:60]
test_y = sonar_std[-train_index, 61]

# （3）交叉验证选 K：使用 5 折交叉验证（仅在训练集上）选择最佳 K 值（范围 1~21 的奇数）。
k_values = seq(1, 21, by = 2)
cv_acc = numeric(length(k_values))
folds = createFolds(train_y, k = 5, returnTrain = FALSE)

for (i in seq_along(k_values)) {
  k = k_values[i]
  fold_acc = numeric(length(folds))
  for (j in seq_along(folds)) {
    val_idx = folds[[j]]
    cv_train_x = train_x[-val_idx, , drop = FALSE]
    cv_train_y = train_y[-val_idx]
    cv_val_x = train_x[val_idx, , drop = FALSE]
    cv_val_y = train_y[val_idx]
    pred_cv = knn(train = cv_train_x, test = cv_val_x, cl = cv_train_y, k = k)
    fold_acc[j] = mean(pred_cv == cv_val_y)
  }
  cv_acc[i] = mean(fold_acc)
}

best_k = k_values[which.max(cv_acc)]
cat("最佳K值:", best_k, "\n")

# （4）最终评估：用最佳 K 值在测试集上评估，输出混淆矩阵、准确率、Kappa 值。
final_pred = knn(train = train_x, test = test_x, cl = train_y, k = best_k)
cm = confusionMatrix(final_pred, test_y)
print(cm)
cat("准确率:", cm$overall['Accuracy'], "\n")
cat("Kappa值:", cm$overall['Kappa'], "\n")
