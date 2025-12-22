import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
plt.rcParams['font.sans-serif'] = ['SimHei']  # 支持中文显示

# 1. 数据加载与预处理
iris = load_iris()
X = iris.data  # 4个特征
y_true = iris.target  # 真实标签

# 特征标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. 训练K均值模型（K=3）
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
y_pred = kmeans.fit_predict(X_scaled)

# 3. 计算调整兰德指数（ARI）
ari = adjusted_rand_score(y_true, y_pred)
print(f"调整兰德指数（ARI）：{ari:.4f}")  # 评估聚类与真实标签的匹配度

# 4. PCA降维（4维→2维）用于可视化
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 5. 可视化：真实标签 vs 聚类结果
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# 真实标签分布图
ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=y_true, cmap='tab10', s=60, edgecolors='black')
ax1.set_title('鸢尾花数据真实标签分布', fontsize=14)
ax1.set_xlabel('PCA维度1', fontsize=12)
ax1.set_ylabel('PCA维度2', fontsize=12)
ax1.grid(alpha=0.3)

# 聚类结果分布图
scatter = ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=y_pred, cmap='tab10', s=60, edgecolors='black')
ax2.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
           c='red', marker='*', s=200, label='聚类中心', edgecolors='black')
ax2.set_title(f'K均值聚类结果（K=3, ARI={ari:.4f}）', fontsize=14)
ax2.set_xlabel('PCA维度1', fontsize=12)
ax2.set_ylabel('PCA维度2', fontsize=12)
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.show()