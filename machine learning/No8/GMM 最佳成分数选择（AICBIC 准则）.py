import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
plt.rcParams['font.sans-serif'] = ['SimHei']  # 支持中文显示


# 复用实验一生成的数据
np.random.seed(42)
n_samples = 300
mean1 = [0, 2]
cov1 = [[1, 0.5], [0.5, 1]]
data1 = np.random.multivariate_normal(mean1, cov1, n_samples)
mean2 = [5, 5]
cov2 = [[1, -0.3], [-0.3, 1]]
data2 = np.random.multivariate_normal(mean2, cov2, n_samples)
mean3 = [2, -3]
cov3 = [[1, 0], [0, 1.5]]
data3 = np.random.multivariate_normal(mean3, cov3, n_samples)
X = np.vstack([data1, data2, data3])

# 1. 测试k从1到8的GMM模型
k_range = range(1, 9)
aic_scores = []
bic_scores = []

for k in k_range:
    gmm = GaussianMixture(n_components=k, random_state=42)
    gmm.fit(X)
    aic_scores.append(gmm.aic(X))
    bic_scores.append(gmm.bic(X))

# 2. 找到AIC/BIC最小值对应的最佳k
best_k_aic = k_range[np.argmin(aic_scores)]
best_k_bic = k_range[np.argmin(bic_scores)]

# 3. 可视化AIC/BIC随k的变化
plt.figure(figsize=(10, 6))
plt.plot(k_range, aic_scores, marker='o', label=f'AIC (最佳k={best_k_aic})')
plt.plot(k_range, bic_scores, marker='s', label=f'BIC (最佳k={best_k_bic})')
plt.title('GMM成分数选择（AIC/BIC准则）')
plt.xlabel('GMM成分数k')
plt.ylabel('AIC/BIC值')
plt.legend()
plt.grid(alpha=0.3)
plt.annotate(f'最小值({best_k_aic}, {np.min(aic_scores):.0f})', 
             xy=(best_k_aic, np.min(aic_scores)), 
             xytext=(best_k_aic+0.5, np.min(aic_scores)+500),
             arrowprops=dict(arrowstyle='->'))
plt.annotate(f'最小值({best_k_bic}, {np.min(bic_scores):.0f})', 
             xy=(best_k_bic, np.min(bic_scores)), 
             xytext=(best_k_bic+0.5, np.min(bic_scores)-500),
             arrowprops=dict(arrowstyle='->'))
plt.show()

print(f"AIC准则最佳成分数k：{best_k_aic}")
print(f"BIC准则最佳成分数k：{best_k_bic}")
print(f"验证结果：最佳k与真实成分数3{'一致' if best_k_aic==3 and best_k_bic==3 else '不一致'}")