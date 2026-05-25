"
8.使用iris的前四个数值变量进行PCA
（1）执行PCA并提取主成分。
（2）绘制碎石图（反差解释比例）
（3）使用前两个主成分进行K-means聚类（k= 3），并于真实物种比较。
"

# 加载包
if(! require(ggplot2)) install.packages("gglot2"); library(ggplot2)


# 加载数据
data(iris)

iris_num = iris[,1:4]

# （1）执行PCA并提取主成分。
pca_ret = prcomp(iris_num, scale. = TRUE)
print(summary(pca_ret))
# （2）绘制碎石图（反差解释比例）
print(screeplot(pca_ret, type = 'line', main = '碎石图'))
# （3）使用前两个主成分进行K-means聚类（k= 3），并于真实物种比较。
# 提取前两个主成分
pc_scores = as.data.frame(pca_ret$x[,1:2])
pc_scores$Species = iris$Species

# k-means聚类分析
kmeans_out = kmeans(pc_scores[,1:2], centers = 3)
pc_scores $ cluster = as.factor(kmeans_out$cluster)

# 聚类可视化
print(ggplot(pc_scores, aes(x = PC1, y = PC2, 
        color = Species,
        shape = cluster)) +
        geom_point(size = 3) +
        labs(titel = 'PCA降维后真实物种 vs 聚类结果')
)