"
7.使用nnet对iris分类
（1）设置隐藏层节点数为5， 训练网络。
（2）预测并计算准确率
（3）尝试不同的size 和 decay， 比较效果。
"

# 导入包
if(!require(nnet)) install.packages("nnet"); library(nnet)


# 加载数据

data(iris)
print(iris)

# 划分数据集
idx = sample(1:nrow(iris), nrow(iris)*0.7)
train = iris[idx, ]
test = iris[-idx, ]

# （1）设置隐藏层节点数为5， 训练网络。
nn_model = nnet(Species ~., 
                data = train, 
                size = 5, 
                decay = 0.1, 
                maxit = 200,
                trace = FALSE)
print(summary(nn_model))
# （2）预测并计算准确率
pred = predict(nn_model, test, type = "class")
mean(pred == test$Species)
# 打印准确率
print(paste("准确率:", round(mean(pred == test$Species) * 100, 2), "%")) 
# （3）尝试不同的size 和 decay， 比较效果。
size = c(3, 6, 10)
decay = c(0, 0.01, 0.1)
best_acc = 0
for (s in size){
    for (d in decay){
        m = nnet(Species ~., 
                 data = train, 
                 size = s, 
                 decay = d, 
                 maxit = 200,
                 trace = FALSE)
        acc = mean(predict(m, test, type = "class") == test$Species)
        print(paste("size:", s, "decay:", d, "准确率:", round(acc * 100, 2), "%"))
        if (acc > best_acc){
            best_acc = acc
            best_model = m
        }
    }
}

print(paste("最佳模型准确率:", round(best_acc * 100, 2), "%"))