import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
plt.rcParams['font.sans-serif'] = ['SimHei']  # 支持中文显示


np.random.seed(42)
# 1. 构造数据集参数
n_normal_total = 1500  # 正常用户总数
n_abnormal = 200       # 异常用户总数

# 正常用户群体
## 低频低额（普通休闲用户）
n1 = n_normal_total // 3
mean1 = [80, 2]
cov1 = [[500, 50], [50, 1]]
data1 = np.random.multivariate_normal(mean1, cov1, n1)
label1 = np.full(n1, 0)

## 中频中额（核心消费用户）
n2 = n_normal_total // 3
mean2 = [200, 5]
cov2 = [[1500, 100], [100, 2]]
data2 = np.random.multivariate_normal(mean2, cov2, n2)
label2 = np.full(n2, 1)

## 高频小额（日常刚需用户）
n3 = n_normal_total - n1 - n2
mean3 = [30, 10]
cov3 = [[100, 30], [30, 1.5]]
data3 = np.random.multivariate_normal(mean3, cov3, n3)
label3 = np.full(n3, 2)

# 合并正常用户数据并过滤负数（消费金额/频次不能为负）
X_normal = np.vstack([data1, data2, data3])
label_normal_true = np.hstack([label1, label2, label3])
valid_idx = (X_normal[:, 0] > 0) & (X_normal[:, 1] > 0)
X_normal = X_normal[valid_idx]
label_normal_true = label_normal_true[valid_idx]
n_normal_total = len(X_normal)  # 修正过滤后数量

# 异常用户群体
## 盗刷（高额低频）
n_ab1 = n_abnormal // 2
mean_ab1 = [1500, 0.5]
cov_ab1 = [[50000, -100], [-100, 0.2]]
data_ab1 = np.random.multivariate_normal(mean_ab1, cov_ab1, n_ab1)

## 刷单（高额高频）
n_ab2 = n_abnormal - n_ab1
mean_ab2 = [500, 20]
cov_ab2 = [[20000, 500], [500, 5]]
data_ab2 = np.random.multivariate_normal(mean_ab2, cov_ab2, n_ab2)

X_abnormal = np.vstack([data_ab1, data_ab2])
X_abnormal = X_abnormal[(X_abnormal[:, 0] > 0) & (X_abnormal[:, 1] > 0)]  # 过滤负数

# 2. 用正常用户训练GMM模型（拟合3类正常群体）
gmm_normal = GaussianMixture(n_components=3, random_state=42)
gmm_normal.fit(X_normal)
y_normal_pred = gmm_normal.predict(X_normal)

# 3. 可视化正常群体与异常群体分布
plt.figure(figsize=(12, 8))
# 绘制正常用户
scatter0 = plt.scatter(X_normal[label_normal_true==0, 0], X_normal[label_normal_true==0, 1], 
                       c='skyblue', s=40, alpha=0.7, label='低频低额（普通休闲）')
scatter1 = plt.scatter(X_normal[label_normal_true==1, 0], X_normal[label_normal_true==1, 1], 
                       c='orange', s=40, alpha=0.7, label='中频中额（核心消费）')
scatter2 = plt.scatter(X_normal[label_normal_true==2, 0], X_normal[label_normal_true==2, 1], 
                       c='lightgreen', s=40, alpha=0.7, label='高频小额（日常刚需）')
# 绘制异常用户
scatter_ab1 = plt.scatter(data_ab1[:, 0], data_ab1[:, 1], 
                         c='red', marker='x', s=60, alpha=0.8, label='异常：盗刷（高额低频）')
scatter_ab2 = plt.scatter(data_ab2[:, 0], data_ab2[:, 1], 
                         c='purple', marker='x', s=60, alpha=0.8, label='异常：刷单（高额高频）')

plt.title('电商用户消费行为分布（正常群体+异常行为）')
plt.xlabel('单次消费金额（元）')
plt.ylabel('周消费频次')
plt.legend(loc='upper right')
plt.grid(alpha=0.3)
plt.show()

# 4. 统计正常用户占比并分析差异
normal_counts = np.bincount(y_normal_pred)
normal_ratios = normal_counts / len(X_normal) * 100
print("=== 三类正常用户占比 ===")
for i, ratio in enumerate(normal_ratios):
    group_names = ['低频低额', '中频中额', '高频小额']
    print(f"{group_names[i]}用户占比：{ratio:.2f}%")

print("\n=== 异常用户与正常群体的行为差异 ===")
print("1. 盗刷用户：单次消费金额远超正常群体（1500元+），但周消费频次极低（<1次），与正常群体分布完全脱节；")
print("2. 刷单用户：单次消费金额较高（500元+）且周消费频次极高（20次+），显著偏离正常用户的频次上限；")
print("3. 正常用户的消费金额与频次呈温和正/负相关，而异常用户呈现极端的“高额低频”或“高额高频”组合。")