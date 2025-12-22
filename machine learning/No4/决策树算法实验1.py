import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文乱码
plt.rcParams['axes.unicode_minus'] = False

# 1. 数据准备与编码）
data = {
    '天气': ['晴天', '晴天', '阴天', '雨天', '雨天', '雨天', '阴天', '晴天', '晴天', '雨天', '雨天', '晴天', '雨天', '阴天', '阴天', '晴天'],
    '温度': ['热', '热', '热', '温和', '冷', '冷', '冷', '温和', '冷', '温和', '温和', '冷', '冷', '温和', '冷', '热'],
    '风': ['无风', '有风', '无风', '无风', '无风', '有风', '有风', '无风', '无风', '无风', '有风', '无风', '有风', '无风', '无风', '有风'],
    '是否打球': ['否', '否', '是', '是', '是', '否', '是', '否', '是', '是', '否', '是', '否', '是', '是', '否']
}
df = pd.DataFrame(data)


label_maps = {}
for col in ['天气', '温度', '风', '是否打球']:
    le = LabelEncoder()
    df[col + '_编码'] = le.fit_transform(df[col])
    label_maps[col] = dict(zip(range(len(le.classes_)), le.classes_))

X = df[['天气_编码', '温度_编码', '风_编码']]
y = df['是否打球_编码']
feature_names = ['天气', '温度', '风']
class_names = [label_maps['是否打球'][0], label_maps['是否打球'][1]]  # ['否', '是']

# 2. 划分数据集并训练模型
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
dt_model = DecisionTreeClassifier(max_depth=3, random_state=42)  # 限制深度使结构更清晰
dt_model.fit(X_train, y_train)
y_pred = dt_model.predict(X_test)

# 3. 可视化1：决策树结构（清晰展示分裂逻辑）
plt.figure(figsize=(12, 8))
plot_tree(
    dt_model,
    feature_names=feature_names,
    class_names=class_names,
    filled=True,  # 按类别填充颜色
    rounded=True,  # 圆角矩形
    fontsize=10
)
plt.title('小明打网球决策树结构', fontsize=14)
plt.tight_layout()
plt.savefig('网球决策树结构.png', dpi=300, bbox_inches='tight')
plt.show()

# 4. 可视化2：特征重要性（哪个因素对打球决策影响最大）
feature_importance = pd.DataFrame({
    '特征': feature_names,
    '重要性': dt_model.feature_importances_
}).sort_values('重要性', ascending=False)

plt.figure(figsize=(8, 5))
sns.barplot(x='重要性', y='特征', data=feature_importance, palette='Blues_r')
plt.title('打球决策的特征重要性排名', fontsize=12)
plt.xlabel('重要性得分', fontsize=10)
plt.ylabel('特征', fontsize=10)
plt.tight_layout()
plt.savefig('网球特征重要性.png', dpi=300)
plt.show()

# 5. 可视化3：测试集预测混淆矩阵（直观展示预测准确性）
cm = confusion_matrix(y_test, y_pred)
# 转换为中文标签
cm_df = pd.DataFrame(
    cm,
    index=[f'真实：{class_names[i]}' for i in range(len(class_names))],
    columns=[f'预测：{class_names[i]}' for i in range(len(class_names))]
)

plt.figure(figsize=(6, 4))
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Greens', cbar=False)
plt.title('测试集预测混淆矩阵', fontsize=12)
plt.tight_layout()
plt.savefig('网球混淆矩阵.png', dpi=300)
plt.show()

# 6. 输出核心结果
print(f"预测精度：{accuracy_score(y_test, y_pred):.2f}")
print("\n特征重要性：")
print(feature_importance)