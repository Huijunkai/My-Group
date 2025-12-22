from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

# ---------------------- 步骤1：生成moons数据集并划分训练集、测试集 ----------------------
X, y = make_moons(n_samples=100, noise=0.5, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=42
)

# ---------------------- 实验（1）：K值对K近邻算法的影响（欧氏距离） ----------------------
k_values = range(1, 51)
accuracies_euclidean = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean', weights='uniform')
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies_euclidean.append(accuracy)

# 可视化K值与测试精度的关系
plt.figure(figsize=(10, 6))
plt.plot(k_values, accuracies_euclidean, marker='o', linestyle='-', color='blue')
plt.xlabel('K值')
plt.ylabel('测试精度')
plt.title('K值对K近邻算法（欧氏距离）的影响')
plt.grid(True)
plt.show()

# ---------------------- 实验（2）：距离类型对K近邻算法的影响 ----------------------
k_list = [5, 10, 20]
metrics = ['euclidean', 'manhattan']
metric_names = ['欧氏距离', '曼哈顿距离']
results = []

for metric, metric_name in zip(metrics, metric_names):
    accs = []
    for k in k_list:
        knn = KNeighborsClassifier(n_neighbors=k, metric=metric, weights='uniform')
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        accs.append(acc)
    results.append(accs)

# 可视化距离类型与测试精度的关系
plt.figure(figsize=(10, 6))
x = np.arange(len(k_list))
width = 0.35
for i, (metric_name, accs) in enumerate(zip(metric_names, results)):
    plt.bar(x + i * width, accs, width, label=metric_name)
plt.xlabel('K值')
plt.ylabel('测试精度')
plt.title('距离类型对K近邻算法的影响')
plt.xticks(x + width/2, k_list)
plt.legend()
plt.grid(True, axis='y')
plt.show()