import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
plt.rcParams['font.sans-serif'] = ['SimHei']  # 支持中文显示

# 1. 模拟用户消费数据
np.random.seed(42)  # 固定随机种子，保证结果可复现

# 簇1：低消费、低频率（潜在用户）
cluster1 = np.random.normal(loc=[50, 2], scale=[15, 0.8], size=(180, 2))
# 簇2：中消费、中频率（核心用户）
cluster2 = np.random.normal(loc=[200, 8], scale=[30, 1.5], size=(220, 2))
# 簇3：高消费、高频率（高价值用户）
cluster3 = np.random.normal(loc=[500, 15], scale=[50, 2], size=(100, 2))

# 合并数据
X_consume = np.vstack([cluster1, cluster2, cluster3])

# 2. 数据预处理：归一化
scaler = MinMaxScaler()
X_consume_scaled = scaler.fit_transform(X_consume)

# 3. 肘部法则确定最优K
inertia = []
k_range = range(1, 11)
for k in k_range:
    kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans_temp.fit(X_consume_scaled)
    inertia.append(kmeans_temp.inertia_)  # 簇内平方和

# 绘制肘部法则图
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(k_range, inertia, 'o-', color='blue', linewidth=2, markersize=8)
plt.axvline(x=3, color='red', linestyle='--', label='最优K=3')
plt.title('肘部法则选择最优聚类数K', fontsize=14)
plt.xlabel('聚类数K', fontsize=12)
plt.ylabel('簇内平方和（惯性）', fontsize=12)
plt.legend()
plt.grid(alpha=0.3)

# 4. 训练K均值模型（K=3）
kmeans_consume = KMeans(n_clusters=3, random_state=42, n_init=10)
y_consume_pred = kmeans_consume.fit_predict(X_consume_scaled)

# 5. 统计各群体用户数量占比
unique, counts = np.unique(y_consume_pred, return_counts=True)
total = len(y_consume_pred)
ratios = (counts / total * 100).round(1)

# 6. 可视化聚类结果（原始特征空间）
plt.subplot(1, 2, 2)
# 绘制用户散点图
for i in range(3):
    mask = y_consume_pred == i
    plt.scatter(X_consume[mask, 0], X_consume[mask, 1], 
               label=f'群体{i+1}（{ratios[i]}%）', s=60, edgecolors='black', alpha=0.7)

# 绘制聚类中心（反归一化回原始尺度）
centers_scaled = kmeans_consume.cluster_centers_
centers = scaler.inverse_transform(centers_scaled)
plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='*', s=200, label='聚类中心', edgecolors='black')

plt.title('用户消费行为分群结果（K=3）', fontsize=14)
plt.xlabel('平均消费金额（元）', fontsize=12)
plt.ylabel('消费频率（次/月）', fontsize=12)
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# 输出各群体统计信息
for i, (count, ratio) in enumerate(zip(counts, ratios)):
    print(f'群体{i+1}：{count}个用户，占比{ratio}%')