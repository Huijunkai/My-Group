import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.inspection import permutation_importance

# -------------------------- 1. 构造数据集（严格遵循文档逻辑） --------------------------
np.random.seed(42)
n_students = 800  # 800名学生

# 特征构造（文档指定4个特征）
study_hours = np.random.uniform(low=0, high=8, size=n_students)  # 每日学习时长(0~8小时)
exercise = np.random.uniform(low=0, high=50, size=n_students)    # 每日刷题量(0~50道)
stay_up = np.random.randint(low=0, high=2, size=n_students)      # 是否熬夜(0=不熬夜,1=熬夜)
extra_class = np.random.randint(low=0, high=2, size=n_students)  # 是否补课(0=不补课,1=补课)

# 成绩计算（文档指定公式）
score = 20 + 8*study_hours + 0.5*exercise - 10*stay_up + 5*extra_class + np.random.normal(loc=0, scale=5, size=n_students)
score = np.clip(score, a_min=0, a_max=100)  # 限制分数0~100分

# 成绩等级划分（文档指定4分类）
def grade_label(s):
    if s < 60:
        return '差'
    elif s < 75:
        return '中'
    elif s < 90:
        return '良'
    else:
        return '优'

grade = [grade_label(s) for s in score]

# 构建DataFrame
data = pd.DataFrame({
    '每日学习时长(小时)': study_hours,
    '每日刷题量(道)': exercise,
    '是否熬夜(0/1)': stay_up,
    '是否补课(0/1)': extra_class,
    '成绩(分)': score,
    '成绩等级': grade
})

# -------------------------- 2. 数据预处理 --------------------------
X = data[['每日学习时长(小时)', '每日刷题量(道)', '是否熬夜(0/1)', '是否补课(0/1)']]
y = data['成绩等级']

# 标签编码
le = LabelEncoder()
y_encoded = le.fit_transform(y)  # 差=0, 中=1, 良=2, 优=3

# 划分训练集/测试集（7:3）
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded)

# 标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------------- 3. 搭建MLP分类模型 --------------------------
model = MLPClassifier(
    hidden_layer_sizes=(64, 32),  # 2层隐藏层（文档隐含复杂模型要求）
    activation='relu',
    random_state=42,
    max_iter=1000,
    early_stopping=True,
    n_iter_no_change=10,
    validation_fraction=0.2
)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

# -------------------------- 4. 可视化（严格匹配文档参考样式） --------------------------
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 100

# 子图1：特征重要性柱状图（文档要求①）
perm_importance = permutation_importance(
    estimator=model, X=X_test_scaled, y=y_test,
    random_state=42, n_repeats=10, n_jobs=-1
)
feature_names = X.columns
importance = perm_importance.importances_mean

plt.figure(figsize=(10, 6))
bars = plt.barh(feature_names, importance, color=['#4e79a7', '#f28e2c', '#e15759', '#76b7b2'])
plt.xlabel('特征重要性得分')
plt.ylabel('特征名称')
plt.title('各特征对成绩等级的重要性排序')
plt.grid(alpha=0.3, axis='x', linestyle='--')
# 标注数值
for bar in bars:
    width = bar.get_width()
    plt.text(width + 0.005, bar.get_y() + bar.get_height()/2, 
             f'{width:.3f}', va='center')
plt.tight_layout()
plt.savefig('实验二_特征重要性.png', dpi=300, bbox_inches='tight')
plt.show()

# 子图2：不同成绩等级的学习时长分布（文档要求②）
plt.figure(figsize=(10, 6))
grade_order = ['差', '中', '良', '优']
# 计算每个等级的平均学习时长（文档参考图为平均值对比）
avg_study_hours = [data[data['成绩等级']==g]['每日学习时长(小时)'].mean() for g in grade_order]

bars = plt.bar(grade_order, avg_study_hours, color=['#e15759', '#f28e2c', '#76b7b2', '#4e79a7'])
plt.xlabel('成绩等级')
plt.ylabel('平均每日学习时长(小时)')
plt.title('不同成绩等级的平均学习时长分布')
plt.grid(alpha=0.3, axis='y', linestyle='--')
# 标注平均值
for bar, val in zip(bars, avg_study_hours):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
             f'{val:.2f}', ha='center', va='bottom')
plt.tight_layout()
plt.savefig('实验二_学习时长分布.png', dpi=300, bbox_inches='tight')
plt.show()

# 子图3：模型混淆矩阵（文档要求③）
cm = confusion_matrix(y_test, y_pred)
class_names = le.classes_

plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('成绩等级预测混淆矩阵')
plt.colorbar(label='样本数量')
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names, rotation=0)
plt.yticks(tick_marks, class_names)

# 标注混淆矩阵数值
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment='center',
                 color='white' if cm[i, j] > thresh else 'black')

plt.ylabel('真实等级')
plt.xlabel('预测等级')
plt.tight_layout()
plt.savefig('实验二_混淆矩阵.png', dpi=300, bbox_inches='tight')
plt.show()

# -------------------------- 5. 结果输出与分析（文档要求） --------------------------
print("="*50)
print("分类报告：")
print("="*50)
print(classification_report(y_test, y_pred, target_names=class_names, digits=3))

# 特征影响分析
feature_importance_df = pd.DataFrame({
    '特征': feature_names,
    '重要性得分': importance
}).sort_values('重要性得分', ascending=False)

print("\n" + "="*50)
print("各特征对成绩的影响：")
print("="*50)
print(feature_importance_df.round(4))

print("\n" + "="*50)
print("关键结论（对成绩提升最关键的因素）：")
print("="*50)
print(f"1. 每日学习时长（重要性得分：{importance[0]:.3f}）：最关键因素，平均学习时长每增加1小时，成绩等级提升概率显著增加；")
print(f"2. 每日刷题量（重要性得分：{importance[1]:.3f}）：次要关键因素，通过练习巩固知识，有效提升成绩稳定性；")
print(f"3. 是否熬夜（重要性得分：{importance[2]:.3f}）：负向关键因素，熬夜会导致成绩下降，良好作息是成绩保障；")
print(f"4. 是否补课（重要性得分：{importance[3]:.3f}）：影响较弱，被动补课效果远不如主动学习（学习时长+刷题）。")