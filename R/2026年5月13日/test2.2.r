'
练习题 8（新数据集）：PCA + K-means 聚类 —— 使用 mtcars 数据集
数据集说明
mtcars 包含 32 种汽车型号的 11 个性能指标（如油耗、马力、重量等）。本题使用前 4 个数值变量进行 PCA 降维，再聚类分析。
题目
使用 mtcars 的前 4 个数值变量（mpg, cyl, disp, hp）进行 PCA。要求：
（1）执行 PCA：对前 4 个变量进行主成分分析（需标准化）。
（2）提取主成分：输出各主成分的标准差、方差解释比例和累计比例。
（3）绘制碎石图（方差解释比例）。
（4）使用前两个主成分进行 K-means 聚类（k = 3），将聚类结果与汽车的 气缸数（cyl） 比较（将 cyl 转换为因子：4缸、6缸、8缸），绘制主成分散点图，用不同颜色和形状区分真实气缸数和聚类标签。
'
library(ggplot2)

data(mtcars)

data = mtcars[, c("mpg", "cyl", "disp", "hp")]

scaled_data = scale(data[, -2])

pca_result = prcomp(scaled_data, scale. = FALSE)

cat("主成分标准差:\n")
print(pca_result$sdev)

variance = pca_result$sdev^2
prop_var = variance / sum(variance)
cum_var = cumsum(prop_var)

cat("\n方差解释比例:\n")
print(prop_var)

cat("\n累计方差解释比例:\n")
print(cum_var)

pca_summary = data.frame(
  PC = paste0("PC", 1:length(prop_var)),
  Variance = prop_var
)

print(ggplot(pca_summary, aes(x = PC, y = Variance)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  labs(title = "碎石图", x = "主成分", y = "方差解释比例") +
  theme_minimal()
)

pc_data = as.data.frame(pca_result$x[, 1:2])
pc_data$cyl = factor(data$cyl, labels = c("4缸", "6缸", "8缸"))

set.seed(123)
kmeans_result = kmeans(scaled_data, centers = 3, nstart = 20)
pc_data$cluster = factor(kmeans_result$cluster)

print(ggplot(pc_data, aes(x = PC1, y = PC2, color = cyl, shape = cluster)) +
  geom_point(size = 3) +
  labs(title = "主成分散点图", x = "第一主成分", y = "第二主成分", 
       color = "真实气缸数", shape = "聚类标签") +
  theme_minimal()
)