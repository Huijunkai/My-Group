import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
plt.rcParams['font.sans-serif'] = ['SimHei']  # 支持中文显示


np.random.seed(42)
n_samples = 400

# 1. 生成2个重叠的高斯成分数据
mean1 = [1, 1]
cov1 = [[2, 1.5], [1.5, 2]]
data1 = np.random.multivariate_normal(mean1, cov1, n_samples)

mean2 = [4, 4]
cov2 = [[2, -1], [-1, 2]]
data2 = np.random.multivariate_normal(mean2, cov2, n_samples)

X = np.vstack([data1, data2])
y_true = np.hstack([np.zeros(n_samples), np.ones(n_samples)])

# 2. 分别训练GMM和K-Means
gmm = GaussianMixture(n_components=2, random_state=42)
y_gmm = gmm.fit_predict(X)

kmeans = KMeans(n_clusters=2, random_state=42)
y_kmeans = kmeans.fit_predict(X)

# 3. 可视化聚类结果（含GMM等概率密度轮廓线）
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# K-Means聚类结果
ax1.scatter(X[:, 0], X[:, 1], c=y_kmeans, cmap='coolwarm', s=30, alpha=0.7)
ax1.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
            marker='*', s=200, c='black', label='聚类中心')
ax1.set_title('K-Means聚类结果')
ax1.set_xlabel('特征1')
ax1.set_ylabel('特征2')
ax1.legend()
ax1.grid(alpha=0.3)

# GMM聚类结果+等概率密度轮廓线
ax2.scatter(X[:, 0], X[:, 1], c=y_gmm, cmap='coolwarm', s=30, alpha=0.7)
ax2.scatter(gmm.means_[:, 0], gmm.means_[:, 1], 
            marker='*', s=200, c='black', label='高斯成分均值')

# 生成网格点绘制等概率轮廓
x_min, x_max = X[:, 0].min()-1, X[:, 0].max()+1
y_min, y_max = X[:, 1].min()-1, X[:, 1].max()+1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
zz = gmm.score_samples(np.c_[xx.ravel(), yy.ravel()])
zz = zz.reshape(xx.shape)
contour = ax2.contour(xx, yy, zz, levels=5, colors='gray', alpha=0.6)
ax2.clabel(contour, inline=True, fontsize=8)

ax2.set_title('GMM聚类结果（含等概率密度轮廓线）')
ax2.set_xlabel('特征1')
ax2.set_ylabel('特征2')
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# 4. 输出对比分析
print("=== GMM与K-Means聚类优势对比 ===")
print("1. K-Means假设数据呈球形分布且各向同性，无法处理椭圆分布的重叠数据；")
print("2. GMM考虑了数据的协方差结构，能拟合椭圆分布，且输出的是概率隶属度而非硬分类；")
print("3. 等概率密度轮廓可直观展示数据的分布特征，更适合重叠聚类场景。")