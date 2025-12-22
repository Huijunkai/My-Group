import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
plt.rcParams['font.sans-serif'] = ['SimHei']  # 支持中文显示


# 设置随机种子保证结果可复现
np.random.seed(42)
n_samples = 300

# 1. 生成3个高斯成分的二维模拟数据
mean1 = [0, 2]
cov1 = [[1, 0.5], [0.5, 1]]
data1 = np.random.multivariate_normal(mean1, cov1, n_samples)

mean2 = [5, 5]
cov2 = [[1, -0.3], [-0.3, 1]]
data2 = np.random.multivariate_normal(mean2, cov2, n_samples)

mean3 = [2, -3]
cov3 = [[1, 0], [0, 1.5]]
data3 = np.random.multivariate_normal(mean3, cov3, n_samples)

# 合并数据和真实标签
X = np.vstack([data1, data2, data3])
y_true = np.hstack([np.zeros(n_samples), np.ones(n_samples), np.full(n_samples, 2)])

# 2. 拟合GMM模型（指定成分数为3）
gmm = GaussianMixture(n_components=3, random_state=42)
y_pred = gmm.fit_predict(X)

# 3. 可视化原始数据和拟合结果
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# 原始数据（按真实类别着色）
sc1 = ax1.scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis', s=30, alpha=0.8)
ax1.set_title('原始数据（按真实类别着色）')
ax1.set_xlabel('特征1')
ax1.set_ylabel('特征2')
ax1.legend(*sc1.legend_elements(), title='真实类别')

# GMM拟合结果（按预测类别着色）
sc2 = ax2.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis', s=30, alpha=0.8)
ax2.set_title('GMM拟合结果（按预测类别着色）')
ax2.set_xlabel('特征1')
ax2.set_ylabel('特征2')
ax2.legend(*sc2.legend_elements(), title='预测类别')

plt.tight_layout()
plt.show()

# 4. 输出模型参数
print("=== 各高斯成分的均值 ===")
for i, mean in enumerate(gmm.means_):
    print(f"成分{i+1}均值：{mean}")

print("\n=== 各高斯成分的协方差矩阵 ===")
for i, cov in enumerate(gmm.covariances_):
    print(f"成分{i+1}协方差：\n{cov}")

print(f"\n模型对数似然值：{gmm.score(X) * len(X):.2f}")  # score返回平均对数似然，乘以样本数得总对数似然