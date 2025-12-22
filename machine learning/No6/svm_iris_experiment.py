import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler

# 1. 加载Iris数据集
iris = datasets.load_iris()
X = iris.data  # 特征
y = iris.target  # 标签
feature_names = iris.feature_names
target_names = iris.target_names

# 2. 数据预处理（标准化）
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)

# 4. 定义不同核函数的SVM模型
kernels = ['linear', 'poly', 'rbf']
models = {kernel: SVC(kernel=kernel, random_state=42) for kernel in kernels}

# 5. 训练+评估模型（新增交叉验证）
results = {}
cv_accuracies = []  # 交叉验证准确率
test_accuracies = []  # 测试集准确率
for kernel, model in models.items():
    # 交叉验证（5折）
    cv_acc = cross_val_score(model, X_train, y_train, cv=5).mean()
    cv_accuracies.append(cv_acc)
    # 训练+测试集评估
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    test_accuracies.append(test_acc)
    # 存储结果
    report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    results[kernel] = {
        'cv_accuracy': cv_acc,
        'test_accuracy': test_acc,
        'report': report,
        'confusion_matrix': cm
    }
    # 打印结果
    print(f"===== 核函数：{kernel} =====")
    print(f"交叉验证准确率：{cv_acc:.4f}")
    print(f"测试集准确率：{test_acc:.4f}")
    print("分类报告：")
    print(classification_report(y_test, y_pred, target_names=target_names))
    print("混淆矩阵：")
    print(cm, "\n")

# 6. 可视化1：不同核函数的交叉验证+测试集准确率对比
plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文显示
plt.rcParams['axes.unicode_minus'] = False
x = np.arange(len(kernels))
width = 0.35
fig, ax = plt.subplots(figsize=(8, 5))
rects1 = ax.bar(x - width/2, cv_accuracies, width, label='交叉验证准确率', color='skyblue')
rects2 = ax.bar(x + width/2, test_accuracies, width, label='测试集准确率', color='orange')
# 添加标签、标题和图例
ax.set_title('不同核函数的SVM准确率对比')
ax.set_xlabel('核函数')
ax.set_ylabel('准确率')
ax.set_xticks(x)
ax.set_xticklabels(kernels)
ax.legend()
# 标注数值
for rect in rects1:
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2., height + 0.005,
            f'{height:.4f}', ha='center', va='bottom')
for rect in rects2:
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2., height + 0.005,
            f'{height:.4f}', ha='center', va='bottom')
ax.set_ylim(0, 1.05)
plt.tight_layout()
plt.show()

# 7. 可视化2：最优模型（linear）的混淆矩阵（带颜色渐变）
plt.figure(figsize=(6, 5))
cm = results['linear']['confusion_matrix']
im = plt.imshow(cm, cmap='Blues', interpolation='nearest')
plt.colorbar(im, label='样本数量')
plt.title('最优模型（LINEAR）混淆矩阵')
plt.xticks(range(len(target_names)), target_names)
plt.yticks(range(len(target_names)), target_names)
# 标注数值
for i in range(len(target_names)):
    for j in range(len(target_names)):
        plt.text(j, i, cm[i, j], ha='center', va='center', 
                 color='white' if cm[i, j] > cm.max()/2 else 'black')
plt.xlabel('预测标签')
plt.ylabel('真实标签')
plt.tight_layout()
plt.show()

# 8. 可视化3：线性核SVM特征重要性排序
if 'linear' in models:
    linear_coef = models['linear'].coef_
    avg_coef = np.mean(np.abs(linear_coef), axis=0)  # 多分类特征权重取平均绝对值
    # 按特征重要性排序
    sorted_idx = np.argsort(avg_coef)[::-1]
    sorted_features = [feature_names[i] for i in sorted_idx]
    sorted_importance = avg_coef[sorted_idx]
    
    plt.figure(figsize=(8, 4))
    plt.bar(sorted_features, sorted_importance, color='lightgreen')
    plt.title('线性核SVM特征重要性排序')
    plt.xlabel('特征')
    plt.ylabel('特征重要性（归一化权重绝对值）')
    plt.xticks(rotation=15)
    # 标注数值
    for i, v in enumerate(sorted_importance):
        plt.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    plt.tight_layout()
    plt.show()