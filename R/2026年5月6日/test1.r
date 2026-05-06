"
6.使用iris数据集，朴素贝叶斯分类
（1）训练模型（默认拉普拉斯= 0）
（2）计算预测的后验概率
（3）输出混淆矩阵和准确率
"

# 导入包
if(!require(e1071)) install.packages('e1071'); library(e1071)
if(!require(caret)) install.packages('caret'); library(caret)

# 加载数据
data(iris)
print(iris)

# 划分数据集
idx = sample(1:nrow(iris), size = 0.7*nrow(iris))
train = iris[idx, ]
test = iris[-idx, ]
print(train)
print(test)

# （1）训练模型（默认拉普拉斯= 0）
nb_model = naiveBayes(Species ~ ., data = train)
print(summary(nb_model))

# （2）计算预测的后验概率
pried_class = predict(nb_model, test)
pred_prob = predict(nb_model, test, type = 'raw')
print(head(pred_prob))

# （3）输出混淆矩阵和准确率
confusionMatrix(pried_class, test$Species)
# 混淆矩阵
print(table(pried_class, test$Species))
# 准确率
print(paste("准确率:", mean(pried_class == test$Species)))