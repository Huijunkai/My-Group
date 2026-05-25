# 1. 使用 mtcars 数据集构建回归决策树
# 背景：mtcars 数据集记录了 32 辆不同汽车的 11 个性能指标，如马力（hp）、重量（wt）、气缸数（cyl）等。

if(!require(rpart)) install.packages("rpart"); library(rpart)
if(!require(rpart.plot)) install.packages("rpart.plot"); library(rpart.plot)
if(!require(Metrics)) install.packages("Metrics"); library(Metrics)

# 加载数据集
data(mtcars)

# （1）将 mtcars 数据按照 70% 训练集和 30% 测试集拆分
set.seed(42)
idx = sample(1:nrow(mtcars), 0.7 * nrow(mtcars))
train = mtcars[idx, ]
test = mtcars[-idx, ]

cat("=== 数据集拆分 ===\n")
cat("训练集样本数:", nrow(train), "\n")
cat("测试集样本数:", nrow(test), "\n")

# （2）使用 rpart 包建立回归决策树
cat("\n=== 构建回归决策树 ===\n")
tree_model = rpart(mpg ~ ., data = train, method = "anova")

# （3）打印并绘制初始决策树的结构
cat("\n初始决策树结构:\n")
print(tree_model)
rpart.plot(tree_model, main = "初始回归决策树", type = 2, extra = 101)

# （4）通过 10 折交叉验证选择最佳 cp 值
cat("\n=== 交叉验证选择最佳 cp 值 ===\n")
print(tree_model$cptable)

best_cp = tree_model$cptable[which.min(tree_model$cptable[, "xerror"]), "CP"]
cat("最佳 cp 值:", best_cp, "\n")

# （5）使用最佳 cp 值对初始决策树进行剪枝
cat("\n=== 剪枝决策树 ===\n")
pruned_tree = prune(tree_model, cp = best_cp)
print(pruned_tree)
rpart.plot(pruned_tree, main = "剪枝后的回归决策树", type = 2, extra = 101)

# （6）在测试集上预测并计算 RMSE
cat("\n=== 模型评估 ===\n")
pred = predict(pruned_tree, test)
rmse_value = rmse(test$mpg, pred)
cat("测试集 RMSE:", rmse_value, "\n")