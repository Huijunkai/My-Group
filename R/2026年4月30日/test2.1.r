'
练习题2（分类 + 逻辑回归 + 正则化）
使用 PimaIndiansDiabetes 数据集（mlbench包），预测 diabetes（是否患糖尿病）。
（1）划分数据为80%训练集和20%测试集。
（2）使用 glmnet 包进行带L2正则化的逻辑回归（岭回归），通过10折交叉验证选择最优的 lambda 值（cv.glmnet）。
（3）用最优模型在测试集上预测，计算AUC值（pROC包）。
（4）输出系数不为零的变量名（若使用L1正则化则输出稀疏解，但题目指定L2，所以所有系数均非零，可输出系数绝对值较大的前5个变量）。
'

# 加载必要的包
if(!require(mlbench)) install.packages("mlbench"); library(mlbench)
if(!require(glmnet)) install.packages("glmnet"); library(glmnet)
if(!require(pROC)) install.packages("pROC"); library(pROC)

# 加载数据
data(PimaIndiansDiabetes)
set.seed(123) # 设置随机种子以确保结果可重复

# 1. 划分数据为80%训练集和20%测试集
idx = sample(1:nrow(PimaIndiansDiabetes), size = 0.8 * nrow(PimaIndiansDiabetes))
train = PimaIndiansDiabetes[idx, ]
test = PimaIndiansDiabetes[-idx, ]

# 2. 使用 glmnet 包进行带L2正则化的逻辑回归（岭回归），通过10折交叉验证选择最优的 lambda 值（cv.glmnet）
x_train = as.matrix(train[, -ncol(train)])
y_train = as.numeric(train$diabetes) - 1 # 转换为0/1
cv_fit = cv.glmnet(x_train, y_train, alpha = 0, family = "binomial")
best_lambda = cv_fit$lambda.min
print(paste("最优的 lambda 值:", best_lambda))

# 3. 用最优模型在测试集上预测，计算AUC值（pROC包）
x_test = as.matrix(test[, -ncol(test)])
y_test = as.numeric(test$diabetes) - 1
pred_prob = predict(cv_fit, newx = x_test, s = "lambda.min", type = "response")
roc_obj = roc(y_test, as.vector(pred_prob))
auc_value = auc(roc_obj)
print(paste("测试集AUC值:", auc_value))

# 4. 输出系数不为零的变量名（L2正则化所有系数均非零，输出绝对值较大的前5个变量）
coef_matrix = as.matrix(coef(cv_fit, s = "lambda.min"))
coef_df = data.frame(variable = rownames(coef_matrix), coefficient = coef_matrix[, 1])
coef_df = coef_df[order(abs(coef_df$coefficient), decreasing = TRUE), ]
top_variables = head(coef_df, 5)
print("系数绝对值较大的前5个变量:")
print(top_variables)
