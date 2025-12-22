import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
# 关键：添加 classification_report 导入
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings('ignore')
# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'使用设备: {device}')

# ------------------------------------------------------------------------------
# 1. 数据集构造
# ------------------------------------------------------------------------------
np.random.seed(42)
n_samples = 2000

# 特征构造
price = np.random.uniform(low=10, high=1000, size=n_samples)  # 商品价格(10-1000元)
discount = np.random.uniform(low=0.5, high=1.0, size=n_samples)  # 促销折扣(0.5-1.0)
comments = np.random.randint(low=0, high=1001, size=n_samples)  # 累计评论数(0-1000)
on_shelf_days = np.random.randint(low=1, high=366, size=n_samples)  # 上架时长(1-365天)

# 构造销量得分（确保四类分布均衡）
sales_score = (1 - discount) * 35 + (comments / 1000) * 30 + \
              (1 - abs(price - 200) / 800) * 20 + (1 - abs(on_shelf_days - 90) / 275) * 15 + \
              np.random.normal(loc=0, scale=8, size=n_samples)
sales_score = np.clip(sales_score, a_min=0, a_max=100)  # 限制得分范围


# 销量等级划分函数
def get_sales_grade(score):
    if score < 25:
        return '低销'
    elif score < 50:
        return '中销'
    elif score < 75:
        return '高销'
    else:
        return '爆销'


sales_grade = [get_sales_grade(s) for s in sales_score]

# 构建DataFrame
data = pd.DataFrame({
    '商品价格(元)': price,
    '促销折扣': discount,
    '评论数': comments,
    '上架时长(天)': on_shelf_days,
    '销量得分': sales_score,
    '销量等级': sales_grade
})

# 查看类别分布
print("销量等级分布：")
print(data['销量等级'].value_counts())

# ------------------------------------------------------------------------------
# 2. 数据预处理
# ------------------------------------------------------------------------------
# 标签编码（文本转数字）
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(data['销量等级'])  # 低销:0, 中销:1, 高销:2, 爆销:3

# 特征分离
X = data[['商品价格(元)', '促销折扣', '评论数', '上架时长(天)']].values

# 划分训练集、验证集、测试集（6:2:2）
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# 特征标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)


# 自定义Dataset类
class SalesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)  # 多分类标签为long类型

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# 创建DataLoader
batch_size = 32
train_dataset = SalesDataset(X_train_scaled, y_train)
val_dataset = SalesDataset(X_val_scaled, y_val)
test_dataset = SalesDataset(X_test_scaled, y_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# ------------------------------------------------------------------------------
# 3. 模型定义（含Dropout防止过拟合）
# ------------------------------------------------------------------------------
class SalesNet(nn.Module):
    def __init__(self, input_dim=4, num_classes=4):
        super(SalesNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),  # Dropout层防止过拟合
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, num_classes)  # 多分类输出（无激活函数，CrossEntropyLoss自带Softmax）
        )

    def forward(self, x):
        return self.layers(x)


# 模型初始化
model = SalesNet().to(device)

# ------------------------------------------------------------------------------
# 4. 训练配置
# ------------------------------------------------------------------------------
criterion = nn.CrossEntropyLoss()  # 多分类交叉熵损失
optimizer = optim.Adam(model.parameters(), lr=1e-3)
epochs = 50

# 训练记录
train_losses = []
val_losses = []
train_accs = []
val_accs = []

# ------------------------------------------------------------------------------
# 5. 训练与验证流程
# ------------------------------------------------------------------------------
for epoch in range(epochs):
    # 训练阶段
    model.train()
    train_loss = 0.0
    train_correct = 0
    total_train = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # 反向传播与优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 统计损失与准确率
        train_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)  # 取概率最大的类别
        train_correct += (predicted == labels).sum().item()
        total_train += labels.size(0)

    # 计算训练集指标
    avg_train_loss = train_loss / total_train
    train_acc = train_correct / total_train
    train_losses.append(avg_train_loss)
    train_accs.append(train_acc)

    # 验证阶段
    model.eval()
    val_loss = 0.0
    val_correct = 0
    total_val = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            val_correct += (predicted == labels).sum().item()
            total_val += labels.size(0)

    # 计算验证集指标
    avg_val_loss = val_loss / total_val
    val_acc = val_correct / total_val
    val_losses.append(avg_val_loss)
    val_accs.append(val_acc)

    # 打印日志
    if (epoch + 1) % 5 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], 训练Loss: {avg_train_loss:.4f}, 训练准确率: {train_acc:.4f}, '
              f'验证Loss: {avg_val_loss:.4f}, 验证准确率: {val_acc:.4f}')

# ------------------------------------------------------------------------------
# 6. 测试集评估
# ------------------------------------------------------------------------------
model.eval()
y_pred = []
y_true = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        y_pred.extend(predicted.cpu().numpy())
        y_true.extend(labels.numpy())

y_pred = np.array(y_pred)
y_true = np.array(y_true)

# 计算多分类指标（加权平均）
test_acc = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')

print("\n=== 商品销量等级分类测试集评估结果 ===")
print(f"准确率: {test_acc:.4f}")
print(f"加权精确率: {precision:.4f}")
print(f"加权召回率: {recall:.4f}")
print(f"加权F1分数: {f1:.4f}")

# ------------------------------------------------------------------------------
# 7. 可视化
# ------------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('商品销量等级分类分析结果', fontsize=16)

# （1）训练/验证损失曲线
axes[0, 0].plot(range(1, epochs + 1), train_losses, label='训练Loss', color='blue', linewidth=2)
axes[0, 0].plot(range(1, epochs + 1), val_losses, label='验证Loss', color='red', linewidth=2)
axes[0, 0].set_xlabel('训练轮数(Epoch)')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].set_title('Loss变化曲线')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# （2）混淆矩阵
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=axes[0, 1],
            xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
axes[0, 1].set_xlabel('预测标签')
axes[0, 1].set_ylabel('真标签')
axes[0, 1].set_title('混淆矩阵')

# （3）训练/验证准确率曲线
axes[1, 0].plot(range(1, epochs + 1), train_accs, label='训练准确率', color='blue', linewidth=2)
axes[1, 0].plot(range(1, epochs + 1), val_accs, label='验证准确率', color='red', linewidth=2)
axes[1, 0].set_xlabel('训练轮数(Epoch)')
axes[1, 0].set_ylabel('准确率')
axes[1, 0].set_title('准确率变化曲线')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# （4）测试集类别分布
class_counts = pd.Series(y_true).value_counts().sort_index()
axes[1, 1].bar(label_encoder.classes_, class_counts.values, color='purple', alpha=0.8)
axes[1, 1].set_xlabel('销量等级')
axes[1, 1].set_ylabel('样本数量')
axes[1, 1].set_title('测试集类别分布')

plt.tight_layout()
plt.show()

# ------------------------------------------------------------------------------
# 8. 分类报告
# ------------------------------------------------------------------------------
print("\n=== 详细分类报告 ===")
print(classification_report(y_true, y_pred, target_names=label_encoder.classes_))
