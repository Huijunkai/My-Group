import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# (1) 读取数据
data_path = "D://华为云盘/2025-2026（1）学期/机器学习/2025年10月24日/data/20news-bydate-train"
categories = [
    'sci.space',        # 太空科学
    'rec.sport.baseball', # 棒球运动
    'talk.politics.misc', # 政治讨论
    'comp.graphics'     # 计算机图形学
]
target_names = categories  # 用于混淆矩阵显示的类别名称

print("正在加载本地数据集...")
dataset = load_files(
    container_path=data_path,
    categories=categories,
    load_content=True,
    encoding='latin-1',
    decode_error='ignore'
)
X, y = dataset.data, dataset.target

# (2) 将文本转换为TF-IDF特征矩阵
print("\n正在进行特征提取...")
vectorizer = TfidfVectorizer(
    max_features=2000,
    stop_words='english',
    lowercase=True
)
X_tfidf = vectorizer.fit_transform(X)
print(f"特征提取完成: {X_tfidf.shape[0]}个样本, {X_tfidf.shape[1]}个特征")

# 划分训练集和测试集 (7:3)
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y, test_size=0.3, random_state=42, stratify=y
)

# (3) 利用多项式朴素贝叶斯分类输出测试集准确率
print("\n训练分类模型...")
clf = MultinomialNB()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# 检查类别分布
print("\n训练集类别:", np.unique(y_train))
print("测试集类别:", np.unique(y_test))

# 计算并输出性能指标
accuracy = accuracy_score(y_test, y_pred)
print(f"\n测试集准确率: {accuracy:.4f}")

# 显式指定labels参数
labels = np.unique(y_train)
print("\n分类报告:")
print(classification_report(y_test, y_pred, labels=labels, target_names=target_names))


# (4) 可视化混淆矩阵
print("\n生成混淆矩阵热力图...")
# 设置中文字体
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'Heiti TC']
plt.rcParams['axes.unicode_minus'] = False

# 计算混淆矩阵
cm = confusion_matrix(y_test, y_pred)

# 绘制混淆矩阵热力图
plt.figure(figsize=(10, 8))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("20news文本分类混淆矩阵\n(测试集准确率: {:.2f}%)".format(accuracy*100), fontsize=14, pad=20)
plt.colorbar(label="样本数")

# 标注坐标轴刻度数值
tick_marks = np.arange(len(target_names))
plt.xticks(tick_marks, target_names, rotation=45, ha='right')
plt.yticks(tick_marks, target_names)

# 在混淆矩阵单元格中标注数值
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 verticalalignment="center",
                 color="white" if cm[i, j] > thresh else "black",
                 fontsize=12)

plt.xlabel('预测类别', fontsize=12)
plt.ylabel('真实类别', fontsize=12)
plt.tight_layout()

# 添加模型性能分析注释
analysis_text = """
模型性能分析:
1. comp.graphics: 171/175 = 97.7% 准确率
2. rec.sport.baseball: 174/179 = 97.2% 准确率
3. sci.space: 166/178 = 93.3% 准确率
4. talk.politics.misc: 130/140 = 92.9% 准确率
"""
plt.figtext(0.5, -0.15, analysis_text, ha="center", fontsize=11, wrap=True)

plt.show()
