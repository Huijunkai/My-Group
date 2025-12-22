import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
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

# 1. 数据集构造
np.random.seed(42)
n_samples = 2000

# 特征构造
monthly_fee = np.random.uniform(low=50, high=500, size=n_samples)  # 月消费(50-500元)
contract_term = np.random.choice(a=[0, 6, 12, 24, 36], size=n_samples,
                                 p=[0.3, 0.2, 0.2, 0.15, 0.15])  # 合约期限
service_calls = np.random.randint(low=0, high=11, size=n_samples)  # 客服通话次数(0-10次)
value_added = np.random.randint(low=0, high=2, size=n_samples)  # 是否开通增值服务(0=否,1=是)

# 构造流失标签（基于业务逻辑）
churn_prob = (monthly_fee / 500) * 0.4 + (1 - contract_term / 36) * 0.3 + \
             (service_calls / 10) * 0.2 + (1 - value_added) * 0.1
churn = np.where(np.random.random(n_samples) < churn_prob, 1, 0)

# 构建DataFrame
data = pd.DataFrame({
    '月消费(元)': monthly_fee,
    '合约期限(月)': contract_term,
    '客服通话次数': service_calls,
    '是否开通增值服务': value_added,
    '是否流失': churn
})

# 2. 数据预处理
# 特征/标签分离
X = data[['月消费(元)', '合约期限(月)', '客服通话次数', '是否开通增值服务']].values
y = data['是否流失'].values.reshape(-1, 1)

# 划分训练集、验证集、测试集（6:2:2）
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# 特征标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)


# 自定义Dataset类
class ChurnDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# 创建DataLoader
batch_size = 32
train_dataset = ChurnDataset(X_train_scaled, y_train)
val_dataset = ChurnDataset(X_val_scaled, y_val)
test_dataset = ChurnDataset(X_test_scaled, y_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 3. 模型定义（仅全连接层）
class ChurnNet(nn.Module):
    def __init__(self, input_dim=4):
        super(ChurnNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()  # 二分类输出
        )

    def forward(self, x):
        return self.layers(x)

# 模型初始化
model = ChurnNet().to(device)

# 4. 训练配置--
criterion = nn.BCELoss()  # 二分类交叉熵损失
optimizer = optim.Adam(model.parameters(), lr=1e-3)
epochs = 50

# 训练记录
train_losses = []
val_losses = []
train_accs = []
val_accs = []

# 5. 训练与验证流程
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
        predicted = (outputs >= 0.5).float()
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
            predicted = (outputs >= 0.5).float()
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

# 6. 测试集评估---
model.eval()
y_pred = []
y_true = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        predicted = (outputs >= 0.5).float()
        y_pred.extend(predicted.cpu().numpy())
        y_true.extend(labels.numpy())

y_pred = np.array(y_pred)
y_true = np.array(y_true)

# 计算分类指标
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
test_acc = accuracy_score(y_true, y_pred)

print("\n=== 客户流失预测测试集评估结果 ===")
print(f"准确率: {test_acc:.4f}")
print(f"精确率: {precision:.4f}")
print(f"召回率: {recall:.4f}")
print(f"F1分数: {f1:.4f}")

# 7. 可视化
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('客户流失预测分析结果', fontsize=16)

# （1）训练/验证损失曲线
axes[0, 0].plot(range(1, epochs + 1), train_losses, label='训练Loss', color='blue', linewidth=2)
axes[0, 0].plot(range(1, epochs + 1), val_losses, label='验证Loss', color='red', linewidth=2)
axes[0, 0].set_xlabel('训练轮数(Epoch)')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].set_title('Loss变化曲线')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# （2）训练/验证准确率曲线
axes[0, 1].plot(range(1, epochs + 1), train_accs, label='训练准确率', color='blue', linewidth=2)
axes[0, 1].plot(range(1, epochs + 1), val_accs, label='验证准确率', color='red', linewidth=2)
axes[0, 1].set_xlabel('训练轮数(Epoch)')
axes[0, 1].set_ylabel('准确率')
axes[0, 1].set_title('准确率变化曲线')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# （3）混淆矩阵
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 2],
            xticklabels=['不流失', '流失'], yticklabels=['不流失', '流失'])
axes[0, 2].set_xlabel('预测标签')
axes[0, 2].set_ylabel('真标签')
axes[0, 2].set_title('混淆矩阵')

# （4）月消费与流失率关系
data['月消费区间'] = pd.cut(data['月消费(元)'], bins=[50, 150, 250, 350, 500])
monthly_churn = data.groupby('月消费区间')['是否流失'].mean()
axes[1, 0].bar(range(len(monthly_churn)), monthly_churn.values, color='skyblue', alpha=0.8)
axes[1, 0].set_xticks(range(len(monthly_churn)))
axes[1, 0].set_xticklabels(monthly_churn.index, rotation=45)
axes[1, 0].set_xlabel('月消费区间(元)')
axes[1, 0].set_ylabel('流失率')
axes[1, 0].set_title('月消费与流失率关系')
axes[1, 0].set_ylim(0, 0.7)

# （5）合约期限与流失率关系
contract_churn = data.groupby('合约期限(月)')['是否流失'].mean().sort_index()
axes[1, 1].bar(contract_churn.index, contract_churn.values, color='lightgreen', alpha=0.8)
axes[1, 1].set_xlabel('合约期限(月)')
axes[1, 1].set_ylabel('流失率')
axes[1, 1].set_title('合约期限与流失率关系')
axes[1, 1].set_ylim(0, 0.7)

# （6）特征重要性对比（基于相关系数）
feature_corr = data[['月消费(元)', '合约期限(月)', '客服通话次数', '是否开通增值服务', '是否流失']].corr()[
    '是否流失'].drop('是否流失')
axes[1, 2].barh(feature_corr.index, abs(feature_corr.values), color='orange', alpha=0.8)
axes[1, 2].set_xlabel('与流失标签的相关系数绝对值')
axes[1, 2].set_title('特征重要性排序')

plt.tight_layout()
plt.show()

# 8. 特征重要性分析
print("\n=== 特征重要性分析（相关系数绝对值）===")
for feat, corr in sorted(zip(feature_corr.index, abs(feature_corr.values)), key=lambda x: x[1], reverse=True):
    print(f"{feat}: {corr:.4f}")

# print("\n=== 关键结论 ===")
# top_feat = sorted(zip(feature_corr.index, abs(feature_corr.values)), key=lambda x: x[1], reverse=True)[0][0]
# print(f"对客户流失影响最大的特征是：{top_feat}")
# print(
#     "原因：该特征与流失标签的相关性最高，直接反映了客户的留存意愿（如高月消费客户可能因成本因素更易流失，短期合约客户留存稳定性差）")
