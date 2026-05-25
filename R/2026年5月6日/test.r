"
5.使用iris数据集，实现KNN分类
（1）标准化数值变量。
（2）使用5折交叉验证选择最佳k值（1~20）
（3）在测试集上评估最佳k的准确率。
"
# 导入包
if(!require(class)) install.packages("class");library(class)
if(!require(caret)) install.packages("caret");library(caret)

# 加载数据
data(iris)
print(iris)


# （1）标准化数值变量。
iris_std = iris
iris_std[,1:4] = scale(iris[,1:4])
print(iris_std)
# （2）使用5折交叉验证选择最佳k值（1~20）
# 数据的划分
train_index = createDataPartition(iris_std$Species, p = 0.7, list = FALSE)
train_x = iris_std[train_index, 1:4]
train_y = iris_std[train_index, 5]
print(train_x)
print(train_y)

test_x = iris_std[-train_index, 1:4]
test_y = iris_std[-train_index, 5]
print(test_x)
print(test_y)

# 使用交叉验证选择最佳k值
k_values = 1:30
acc = numeric(length(k_values))
for(k in k_values){
    pred = knn(train = train_x, test = train_x, cl = train_y, k = k)
    acc[k] = mean(pred == train_y)
}
best_k = which.max(acc)
print(paste("最佳k值:", best_k))
# （3）在测试集上评估最佳k的准确率。
final_ped = knn(train = train_x, test = test_x, cl = train_y, k = best_k)
mean(final_ped == test_y)
print(paste("测试集上的准确率:", mean(final_ped == test_y)))