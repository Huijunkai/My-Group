'
4.使用iris数据集，用SVM进行分类（径向基核）
（1）用tune.svm寻找最佳cost和gamma参数
（2）用最佳参数重新训练，并计算测试集准确率（70%训练，30%测试）
'

# 加载必要的包
if(!require(e1071)) install.packages("e1071"); library(e1071)

# 加载数据
data(iris)
print(iris)

set.seed(123) #设置随机种子

# 划分数据集
idx = sample(1:nrow(iris), size = 0.7 * nrow(iris))
train = iris[idx, ]
test = iris[-idx, ]

# 1. 用tune.svm寻找最佳cost和gamma参数
tune_result = tune.svm(Species ~ .,
                       data = train,
                       kernel = "radial",
                       cost = 10^(-1:2), 
                       gamma = 10^(-1:2))

best_params = tune_result$best.parameters

# 2. 用最佳参数重新训练，并计算测试集准确率

svm_model = svm(Species ~ ., data = train, kernel = "radial",
                cost = best_params$cost, gamma = best_params$gamma)
predictions = predict(svm_model, newdata = test)
confusionMatrix 