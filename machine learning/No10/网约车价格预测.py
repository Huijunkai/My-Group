import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import warnings

warnings.filterwarnings('ignore')

# 设置中文显示（解决乱码问题）
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 设备配置（GPU优先，自动切换）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'使用设备: {device}')

# ------------------------------------------------------------------------------
# 1. 数据集构造（严格遵循文档逻辑）
# ------------------------------------------------------------------------------
np.random.seed(42)
n_samples = 2000

# 特征构造（贴合网约车定价逻辑）
distance = np.random.uniform(low=1, high=20, size=n_samples)  # 行驶里程(1-20km)
duration = np.random.uniform(low=5, high=120, size=n_samples)  # 行驶时长(5-120分钟)
rush_hour = np.random.randint(low=0, high=2, size=n_samples)  # 是否高峰期(0=否,1=是)
rainy = np.random.randint(low=0, high=3, size=n_samples)  # 是否雨天(0=无雨,1=小雨,2=大雨)

# 价格计算（基础价+里程费+时长费+高峰/雨天溢价）
base_price = 8  # 基础价8元
mileage_fee = distance * 2.5  # 里程费2.5元/km
time_fee = duration * 0.3  # 时长费0.3元/分钟
rush_premium = np.where(rush_hour == 1, 1.2, 1.0)  # 高峰期溢价1.2倍
rain_premium = np.where(rainy == 0, 1.0, np.where(rainy == 1, 1.1, 1.3))  # 雨天溢价（无雨1.0/小雨1.1/大雨1.3）
total_price = (base_price + mileage_fee + time_fee) * rush_premium * rain_premium + np.random.normal(loc=0, scale=2,
                                                                                                     size=n_samples)
total_price = np.clip(total_price, a_min=8, a_max=200)  # 价格限制在8-200元

# 构建DataFrame
data = pd.DataFrame({
    '行驶里程(km)': distance,
    '行驶时长(min)': duration,
    '是否高峰期': rush_hour,
    '是否雨天': rainy,
    '网约车价格(元)': total_price
})

# 数据格式转换（方便后续分组统计）
data['是否高峰期_文本'] = data['是否高峰期'].map({0: '非高峰', 1: '高峰'})
data['是否雨天_文本'] = data['是否雨天'].map({0: '无雨', 1: '小雨', 2: '大雨'})

# ------------------------------------------------------------------------------
# 2. 数据预处理（标签标准化+数据集划分）
# ------------------------------------------------------------------------------
# 特征/标签分离
X = data[['行驶里程(km)', '行驶时长(min)', '是否高峰期', '是否雨天']].values
y = total_price.reshape(-1, 1)  # 回归标签为二维数组

# 划分训练集、验证集、测试集（6:2:2）
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# 特征标准化 + 标签标准化（提升回归效果，文档要求）
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_val_scaled = scaler_X.transform(X_val)
X_test_scaled = scaler_X.transform(X_test)

y_train_scaled = scaler_y.fit_transform(y_train)
y_val_scaled = scaler_y.transform(y_val)
y_test_scaled = scaler_y.transform(y_test)


# 自定义Dataset类（适配PyTorch DataLoader）
class RideDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# 创建DataLoader
batch_size = 32
train_dataset = RideDataset(X_train_scaled, y_train_scaled)
val_dataset = RideDataset(X_val_scaled, y_val_scaled)
test_dataset = RideDataset(X_test_scaled, y_test_scaled)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# ------------------------------------------------------------------------------
# 3. 模型定义（仅全连接层，文档要求）
# ------------------------------------------------------------------------------
class RideNet(nn.Module):
    def __init__(self, input_dim=4):
        super(RideNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)  # 回归任务输出单个连续值
        )

    def forward(self, x):
        return self.layers(x)


# 模型初始化
model = RideNet().to(device)

# ------------------------------------------------------------------------------
# 4. 训练配置（损失函数+优化器）
# ------------------------------------------------------------------------------
criterion = nn.MSELoss()  # 回归任务用均方误差损失
optimizer = optim.Adam(model.parameters(), lr=1e-3)
epochs = 80  # 足够迭代次数确保收敛

# 训练记录（保存训练/验证Loss）
train_losses = []
val_losses = []

# ------------------------------------------------------------------------------
# 5. 训练与验证流程（完整流程：训练→验证→日志输出）
# ------------------------------------------------------------------------------
for epoch in range(epochs):
    # 训练阶段
    model.train()
    train_loss = 0.0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # 反向传播与优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 统计训练损失
        train_loss += loss.item() * inputs.size(0)

    # 计算训练集平均损失
    avg_train_loss = train_loss / len(train_loader.dataset)
    train_losses.append(avg_train_loss)

    # 验证阶段（无梯度计算）
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)

    # 计算验证集平均损失
    avg_val_loss = val_loss / len(val_loader.dataset)
    val_losses.append(avg_val_loss)

    # 每10轮打印日志
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], 训练Loss: {avg_train_loss:.4f}, 验证Loss: {avg_val_loss:.4f}')

# ------------------------------------------------------------------------------
# 6. 测试集评估（回归指标+预测结果反标准化）
# ------------------------------------------------------------------------------
model.eval()
y_pred_scaled = []
y_true_scaled = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        y_pred_scaled.extend(outputs.cpu().numpy())
        y_true_scaled.extend(labels.numpy())

# 反标准化（恢复原始价格尺度，用于可视化和评估）
y_pred = scaler_y.inverse_transform(np.array(y_pred_scaled))
y_true = scaler_y.inverse_transform(np.array(y_true_scaled))

# 计算回归核心指标
r2 = r2_score(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mae = mean_absolute_error(y_true, y_pred)

print("\n=== 网约车价格预测测试集评估结果 ===")
print(f'R²分数: {r2:.4f}')  # 拟合优度（越接近1越好）
print(f'RMSE（均方根误差）: {rmse:.2f} 元')  # 预测误差
print(f'MAE（平均绝对误差）: {mae:.2f} 元')  # 平均误差

# ------------------------------------------------------------------------------
# 7. 可视化（严格匹配文档要求的4类图，共2x2子图布局）
# ------------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('网约车价格预测分析结果', fontsize=16, fontweight='bold')

# （1）训练/验证Loss变化曲线（文档要求第1张图）
axes[0, 0].plot(range(1, epochs + 1), train_losses, label='训练Loss', color='blue', linewidth=2.5)
axes[0, 0].plot(range(1, epochs + 1), val_losses, label='验证Loss', color='red', linewidth=2.5)
axes[0, 0].set_xlabel('训练轮数(Epoch)', fontsize=12)
axes[0, 0].set_ylabel('Loss（MSE）', fontsize=12)
axes[0, 0].set_title('Loss变化曲线', fontsize=14, fontweight='bold')
axes[0, 0].legend(fontsize=11)
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_ylim(0, max(max(train_losses), max(val_losses)) * 1.1)  # 适配Loss范围

# （2）真实价格vs预测价格散点图（文档要求第2张图）
axes[0, 1].scatter(y_true, y_pred, alpha=0.6, color='darkblue', s=30)
# 添加完美预测线（y=x）
min_price = min(y_true.min(), y_pred.min())
max_price = max(y_true.max(), y_pred.max())
axes[0, 1].plot([min_price, max_price], [min_price, max_price], 'r--', linewidth=2.5, label='完美预测线(y=x)')
# 标注示例点（参考文档格式）
axes[0, 1].annotate(f'(x,y)=({y_true.mean():.1f}, {y_pred.mean():.1f})',
                    xy=(y_true.mean(), y_pred.mean()), xytext=(y_true.mean() + 10, y_pred.mean() - 10),
                    fontsize=11, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
axes[0, 1].set_xlabel('真实价格(元)', fontsize=12)
axes[0, 1].set_ylabel('预测价格(元)', fontsize=12)
axes[0, 1].set_title('真实价格vs预测价格', fontsize=14, fontweight='bold')
axes[0, 1].legend(fontsize=11)
axes[0, 1].grid(True, alpha=0.3)

# （3）行驶里程/时长与价格相关性（文档要求第3张图：左上+右上）
# 子图3-1：行驶里程与价格相关性
axes[1, 0].scatter(data['行驶里程(km)'], data['网约车价格(元)'], alpha=0.5, color='green', s=20)
axes[1, 0].set_xlabel('行驶里程(km)', fontsize=12)
axes[1, 0].set_ylabel('网约车价格(元)', fontsize=12)
axes[1, 0].set_title('行驶里程与价格相关性', fontsize=14, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)

# 子图3-2：行驶时长与价格相关性
axes[1, 1].scatter(data['行驶时长(min)'], data['网约车价格(元)'], alpha=0.5, color='orange', s=20)
axes[1, 1].set_xlabel('行驶时长(min)', fontsize=12)
axes[1, 1].set_ylabel('网约车价格(元)', fontsize=12)
axes[1, 1].set_title('行驶时长与价格相关性', fontsize=14, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)

# 调整子图间距，避免重叠
plt.tight_layout()
plt.subplots_adjust(top=0.92)
plt.show()

# 额外补充：高峰期/不同天气价格对比图（文档要求第4张图）
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('网约车价格影响因素对比', fontsize=16, fontweight='bold')

# （4-1）高峰期vs非高峰期价格对比
rush_price = data.groupby('是否高峰期_文本')['网约车价格(元)'].mean()
axes[0].bar(rush_price.index, rush_price.values, color=['lightblue', 'red'], alpha=0.8, width=0.5)
# 标注平均价格
for i, v in enumerate(rush_price.values):
    axes[0].text(i, v + 1, f'{v:.1f}元', ha='center', va='bottom', fontsize=11, fontweight='bold')
axes[0].set_xlabel('时段', fontsize=12)
axes[0].set_ylabel('平均价格(元)', fontsize=12)
axes[0].set_title('高峰期/非高峰期价格对比', fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3, axis='y')
axes[0].set_ylim(0, max(rush_price.values) * 1.15)

# （4-2）不同天气价格对比
rain_price = data.groupby('是否雨天_文本')['网约车价格(元)'].mean()
axes[1].bar(rain_price.index, rain_price.values, color=['lightgreen', 'skyblue', 'darkblue'], alpha=0.8, width=0.5)
# 标注平均价格
for i, v in enumerate(rain_price.values):
    axes[1].text(i, v + 1, f'{v:.1f}元', ha='center', va='bottom', fontsize=11, fontweight='bold')
axes[1].set_xlabel('天气状况', fontsize=12)
axes[1].set_ylabel('平均价格(元)', fontsize=12)
axes[1].set_title('不同天气价格对比', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3, axis='y')
axes[1].set_ylim(0, max(rain_price.values) * 1.15)

plt.tight_layout()
plt.subplots_adjust(top=0.85)
plt.show()

# ------------------------------------------------------------------------------
# 8. 关键结论分析（文档要求）
# ------------------------------------------------------------------------------
print("\n=== 关键结论分析 ===")
print(f"1. 模型拟合效果：R²分数为{r2:.4f}，说明模型能解释{r2 * 100:.1f}%的价格变异，拟合效果良好；")
print(f"2. 预测精度：RMSE为{rmse:.2f}元，平均预测误差控制在{mae:.2f}元内，满足实际应用需求；")
print(f"3. 核心影响因素：")
print(f"   - 行驶里程：与价格呈强正相关（里程费是主要收入来源）；")
print(f"   - 行驶时长：与价格呈正相关（时长费补充收入）；")
print(f"   - 高峰期：高峰期平均价格比非高峰期高{(rush_price['高峰'] - rush_price['非高峰']):.1f}元（溢价1.2倍）；")
print(f"   - 天气：大雨天价格最高（溢价1.3倍），小雨天次之（溢价1.1倍），无雨天价格最低。")
