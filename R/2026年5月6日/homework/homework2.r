"
练习题 6（新数据集）：朴素贝叶斯 —— 使用 PimaIndiansDiabetes2 数据集
数据集说明
mlbench::PimaIndiansDiabetes2 包含 768 名印第安女性的医疗记录，目标变量 diabetes 为 pos（糖尿病阳性）或 neg（阴性）。特征包括怀孕次数、血糖浓度、血压等。数据中存在缺失值。
题目
使用 PimaIndiansDiabetes2 数据集，实现朴素贝叶斯分类。要求：
（1）缺失值处理：删除任意含缺失值的行。
（2）数据划分：按 75% 训练、25% 测试划分。
（3）训练模型：使用 e1071::naiveBayes，设置 laplace = 0.5。
（4）后验概率：输出测试集前 6 个样本的预测类别和后验概率。
（5）模型评估：输出混淆矩阵、准确率、ROC 曲线及 AUC 值。
"

# 导入包
if(!require(e1071)) install.packages('e1071'); library(e1071)
if(!require(caret)) install.packages('caret'); library(caret)
if(!require(mlbench)) install.packages('mlbench'); library(mlbench)

# 加载数据
data(PimaIndiansDiabetes2)
cat("原始数据集样本数:", nrow(PimaIndiansDiabetes2), "特征数:", ncol(PimaIndiansDiabetes2), "\n")

# （1）缺失值处理：删除任意含缺失值的行。
pima_clean = na.omit(PimaIndiansDiabetes2)
cat("清洗后样本数:", nrow(pima_clean), "\n")

# （2）数据划分：按 75% 训练、25% 测试划分。
train_index = createDataPartition(pima_clean$diabetes, p = 0.75, list = FALSE)
train = pima_clean[train_index, ]
test = pima_clean[-train_index, ]
cat("训练集样本数:", nrow(train), "测试集样本数:", nrow(test), "\n")

# （3）训练模型：使用 e1071::naiveBayes，设置 laplace = 0.5。
nb_model = naiveBayes(diabetes ~ ., data = train, laplace = 0.5)
print(summary(nb_model))

# （4）后验概率：输出测试集前 6 个样本的预测类别和后验概率。
pred_class = predict(nb_model, test)
pred_prob = predict(nb_model, test, type = 'raw')
cat("前6个测试样本预测类别:\n")
print(head(pred_class))
cat("前6个测试样本后验概率:\n")
print(head(pred_prob))

# （5）模型评估：输出混淆矩阵、准确率、ROC 曲线及 AUC 值。
cm = confusionMatrix(pred_class, test$diabetes)
print(cm)
cat("准确率:", cm$overall['Accuracy'], "\n")

# ROC 曲线及 AUC 值
if(!require(pROC)) install.packages('pROC'); library(pROC)
roc_obj = roc(test$diabetes, pred_prob[, "pos"])
auc_value = auc(roc_obj)
cat("AUC值:", auc_value, "\n")
plot(roc_obj, main = "ROC Curve", col = "blue")
