'
2.使用R语言中的iris数据集，将Species变量转换为二分类问题（如setosa vs 其他），构建逻辑回归模型：
（1）使用glm(family = binomial)拟合模型
（2）预测测试集的概率并设定分类阈值为0.6
（3）构建混淆矩阵并计算准确率、灵敏度和特异性
（4）绘制ROC曲线并计算AUC值
'

# 加载必要的包
library(pROC)

# 加载数据集
data(iris)

# 查看数据集结构
str(iris)

# 将 Species 变量转换为二分类问题（setosa vs 其他）
iris$is_setosa = ifelse(iris$Species == "setosa", 1, 0)

# 划分训练集和测试集
set.seed(123)  # 设置随机种子，保证结果可重复
train_index = sample(1:nrow(iris), 0.7 * nrow(iris))
train_data = iris[train_index, ]
test_data = iris[-train_index, ]

# （1）使用 glm(family = binomial) 拟合模型
logit_model = glm(is_setosa ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width, 
                   data = train_data, family = binomial)

# 查看模型摘要
summary(logit_model)

# （2）预测测试集的概率并设定分类阈值为 0.6
test_prob = predict(logit_model, newdata = test_data, type = "response")
test_pred = ifelse(test_prob >= 0.6, 1, 0)

# （3）构建混淆矩阵并计算准确率、灵敏度和特异性
conf_matrix = table(Actual = test_data$is_setosa, Predicted = test_pred)
print("混淆矩阵：")
print(conf_matrix)

# 计算准确率
accuracy = sum(diag(conf_matrix)) / sum(conf_matrix)

# 计算灵敏度（召回率）
sensitivity = conf_matrix[2, 2] / sum(conf_matrix[2, ])

# 计算特异性
specificity = conf_matrix[1, 1] / sum(conf_matrix[1, ])

print(paste("准确率：", round(accuracy, 4)))
print(paste("灵敏度：", round(sensitivity, 4)))
print(paste("特异性：", round(specificity, 4)))

# （4）绘制 ROC 曲线并计算 AUC 值
roc_obj = roc(test_data$is_setosa, test_prob)
auc_value = auc(roc_obj)

# 绘制 ROC 曲线
plot(roc_obj, main = paste("ROC 曲线 (AUC =", round(auc_value, 4), ")"), 
     col = "blue", lwd = 2)

# 输出 AUC 值
print(paste("AUC 值：", round(auc_value, 4)))
