import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
plt.rcParams['font.sans-serif'] = ['SimHei']  # 支持中文显示

# 1. 模拟城市经纬度数据
np.random.seed(42)

# 4个真实地理集群中心（纬度N，经度E）
cluster_centers_true = np.array([
    [32.5, 105.2],  # 西南区域
    [38.1, 110.3],  # 西北区域
    [35.7, 118.5],  # 华北区域
    [42.3, 122.1]   # 东北区域
])
cluster_sizes = [28, 22, 25, 25]  # 每个区域城市数量

# 生成带噪声的城市数据
X_city = []
for center, size in zip(cluster_centers_true, cluster_sizes):
    cluster = np.random.normal(loc=center, scale=[0.8, 1.2], size=(size, 2))
    X_city.append(cluster)
X_city = np.vstack(X_city)  # (100, 2)：每行[纬度, 经度]

# 2. 训练K均值模型（K=4）
kmeans_city = KMeans(n_clusters=4, random_state=42, n_init=10)
y_city_pred = kmeans_city.fit_predict(X_city)

# 3. 统计各区域城市数量及中心经纬度（反归一化，直接使用原始经纬度）
unique, counts = np.unique(y_city_pred, return_counts=True)
centers_city = kmeans_city.cluster_centers_  # 聚类中心（纬度，经度）

# 4. 可视化地理聚类结果（模拟地图背景）
plt.figure(figsize=(12, 8))

# 绘制各区域城市
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
for i in range(4):
    mask = y_city_pred == i
    plt.scatter(X_city[mask, 1], X_city[mask, 0],  # 经度为x轴，纬度为y轴（符合地图习惯）
               c=colors[i], label=f'区域{i+1}（{counts[i]}个城市）', 
               s=80, edgecolors='black', alpha=0.7)

# 绘制区域中心城市（聚类中心）
plt.scatter(centers_city[:, 1], centers_city[:, 0], 
           c='black', marker='*', s=300, label='区域中心城市', edgecolors='yellow', linewidth=2)

# 添加经纬度标签
for i, (lat, lon) in enumerate(centers_city):
    plt.annotate(f'中心({lat:.1f}°N,{lon:.1f}°E)', 
                xy=(lon, lat), xytext=(5, 5), textcoords='offset points',
                fontsize=10, fontweight='bold')

plt.title('城市经纬度聚类结果（模拟区域中心城市筛选）', fontsize=16)
plt.xlabel('经度（°E）', fontsize=14)
plt.ylabel('纬度（°N）', fontsize=14)
plt.xlim(100, 125)  # 限定经度范围
plt.ylim(30, 45)    # 限定纬度范围
plt.grid(alpha=0.3)
plt.legend(fontsize=12)
plt.show()

# 输出各区域统计信息
print("各地理区域统计：")
for i, (count, (lat, lon)) in enumerate(zip(counts, centers_city)):
    print(f'区域{i+1}：{count}个城市，中心经纬度：({lat:.1f}°N, {lon:.1f}°E)')