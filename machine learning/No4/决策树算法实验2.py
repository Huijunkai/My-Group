import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 1. 数据准备
iris = datasets.load_iris()
X = iris.data[:, [0, 2]]  # 花萼长度、花瓣长度
y = iris.target
feature_names = [iris.feature_names[0], iris.feature_names[2]]  # 特征名称
class_names = iris.target_names  # 类别名称：setosa、versicolor、virginica

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=42
)

# 2. 实验1：决策树深度对精度的影响（柱状图+折线图结合）
depths = [2, 6, 10, 15, 20]
train_acc = []
test_acc = []
node_counts = []  # 记录每个深度的决策树节点数（反映复杂度）

for depth in depths:
    dt = DecisionTreeClassifier(max_depth=depth, random_state=42)
    dt.fit(X_train, y_train)
    # 精度计算
    train_acc.append(accuracy_score(y_train, dt.predict(X_train)))
    test_acc.append(accuracy_score(y_test, dt.predict(X_test)))
    # 节点数统计
    node_counts.append(dt.tree_.node_count)

# 可视化：深度-精度-复杂度关系
fig, ax1 = plt.subplots(figsize=(10, 6))

# 精度折线+柱状图
x = np.arange(len(depths))
width = 0.35
ax1.bar(x - width/2, train_acc, width, label='训练集精度', color='skyblue', alpha=0.7)
ax1.bar(x + width/2, test_acc, width, label='测试集精度', color='lightcoral', alpha=0.7)
ax1.plot(x, train_acc, 'o-', color='blue', linewidth=2)
ax1.plot(x, test_acc, 's-', color='red', linewidth=2)
ax1.set_xlabel('决策树深度', fontsize=10)
ax1.set_ylabel('精度', fontsize=10)
ax1.set_xticks(x)
ax1.set_xticklabels(depths)
ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.3)

# 节点数（复杂度）次坐标轴
ax2 = ax1.twinx()
ax2.plot(x, node_counts, 'D-', color='darkgreen', linewidth=2, label='节点数（复杂度）')
ax2.set_ylabel('决策树节点数', fontsize=10)
ax2.legend(loc='upper right')

plt.title('决策树深度对精度和复杂度的影响', fontsize=12)
plt.tight_layout()
plt.savefig('鸢尾花深度-精度-复杂度.png', dpi=300)
plt.show()

# 3. 实验2：最优深度决策树的分类边界（训练集+测试集对比）
optimal_depth = 6  # 从上述结果中选择最优深度
dt_opt = DecisionTreeClassifier(max_depth=optimal_depth, random_state=42)
dt_opt.fit(X_train, y_train)

# 绘制决策边界的工具函数
def plot_decision_boundary(model, X, y, title, feature_names):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', s=30, edgecolor='k')
    plt.xlabel(feature_names[0], fontsize=10)
    plt.ylabel(feature_names[1], fontsize=10)
    plt.title(title, fontsize=12)
    plt.legend(*scatter.legend_elements(), title="类别")
    plt.tight_layout()
    return plt

# 训练集分类边界
plot_decision_boundary(dt_opt, X_train, y_train, f'最优深度{optimal_depth}：训练集分类边界', feature_names)
plt.savefig('鸢尾花训练集分类边界.png', dpi=300)
plt.show()

# 测试集分类边界
plot_decision_boundary(dt_opt, X_test, y_test, f'最优深度{optimal_depth}：测试集分类边界', feature_names)
plt.savefig('鸢尾花测试集分类边界.png', dpi=300)
plt.show()

# 4. 可视化3：最优模型的混淆矩阵（多类别分类效果）
y_pred_opt = dt_opt.predict(X_test)
cm = confusion_matrix(y_test, y_pred_opt)
cm_df = pd.DataFrame(
    cm,
    index=[f'真实：{class_names[i]}' for i in range(len(class_names))],
    columns=[f'预测：{class_names[i]}' for i in range(len(class_names))]
)

plt.figure(figsize=(8, 6))
sns.heatmap(cm_df, annot=True, fmt='d', cmap='YlOrRd', cbar=False)
plt.title(f'最优深度{optimal_depth}：测试集混淆矩阵', fontsize=12)
plt.tight_layout()
plt.savefig('鸢尾花混淆矩阵.png', dpi=300)
plt.show()

# 5. 可视化4：最优决策树结构
plt.figure(figsize=(15, 10))
plot_tree(
    dt_opt,
    feature_names=feature_names,
    class_names=class_names,
    filled=True,
    rounded=True,
    fontsize=9
)
plt.title(f'鸢尾花分类决策树结构（深度={optimal_depth}）', fontsize=14)
plt.tight_layout()
plt.savefig('鸢尾花决策树结构.png', dpi=300, bbox_inches='tight')
plt.show()

# 输出核心结果
print(f"最优深度{optimal_depth}的测试集精度：{accuracy_score(y_test, y_pred_opt):.4f}")
print(f"决策树节点数（复杂度）：{dt_opt.tree_.node_count}")