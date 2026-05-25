'
1.使用iris数据集，建立决策树预测物种
（1）使用rpary训练树，并绘制树结构。
（2）通过交叉验证选择最佳cp值
（3）对树进行剪枝，计算测试集准确率(70%训练，30%测试)
'

# 自动安装并加载
if (!require("rpart.plot", quietly = TRUE)) {
  install.packages("rpart.plot")
  library(rpart.plot)
}

set.seed(123)

# 加载数据
data(iris)
# print(iris)

# 划分数据集
idx = sample(1:nrow(iris), 0.7*nrow(iris))
train = iris[idx,]
test = iris[-idx,]
# print(length(test))

# （1）使用rpary训练树，并绘制树结构。
tree = rpart(Species ~ . ,data = train, method = 'class')
print(rpart.plot(tree))

# （2）通过交叉验证选择最佳cp值
printcp(tree)
plotcp(tree)

best_cp = tree$cptable[which.min(tree$cptable[,'xerror'])]
# （3）对树进行剪枝，计算测试集准确率(70%训练，30%测试)
proued_tree = prune(tree, cp = best_cp)

pred = predict(proued_tree, test, type = 'class')
print((pred == test$Species))