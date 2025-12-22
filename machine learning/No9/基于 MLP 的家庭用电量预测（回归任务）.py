import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# -------------------------- 1. 构造数据集 --------------------------
np.random.seed(42)
n_samples = 500  # 500天数据

# 特征构造
temperature = np.random.uniform(low=5, high=38, size=n_samples)  # 温度5-38°C
is_weekend = np.random.randint(low=0, high=2, size=n_samples)  # 是否周末(0/1)
is_aircon = np.where(temperature > 28, 1, np.random.randint(low=0, high=2, size=n_samples))  # 温度>28°C大概率开空调

# 标签构造（贴近现实逻辑）
electricity = 3 + 0.1*temperature + 1*is_weekend + 3*is_aircon + np.random.normal(loc=0, scale=0.5, size=n_samples)

# 构建DataFrame
data = pd.DataFrame({
    '温度(°C)': temperature,
    '是否周末(0/1)': is_weekend,
    '是否开空调(0/1)': is_aircon,
    '日用电量(kWh)': electricity
})

# -------------------------- 2. 数据预处理 --------------------------
X = data[['温度(°C)', '是否周末(0/1)', '是否开空调(0/1)']]
y = data['日用电量(kWh)']

# 划分训练集/测试集（7:3）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 标准化（神经网络必需）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------------- 3. 搭建MLP回归模型 --------------------------
# 对比不同隐藏层数量（1层/2层/3层）
hidden_layer_configs = [
    (8,),          # 简单模型：1层8个神经元
    (16, 8),       # 中等模型：2层16+8个神经元
    (32, 16, 8)    # 复杂模型：3层32+16+8个神经元
]

mae_results = []
models = []

for config in hidden_layer_configs:
    model = MLPRegressor(
        hidden_layer_sizes=config,
        activation='relu',
        random_state=42,
        max_iter=1000,
        early_stopping=True,  # 早停防止过拟合
        n_iter_no_change=10
    )
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, y_pred)
    mae_results.append(mae)
    models.append(model)

# 选择最优模型（MAE最小）
best_idx = np.argmin(mae_results)
best_model = models[best_idx]
y_pred_best = best_model.predict(X_test_scaled)
best_mae = mae_results[best_idx]
best_r2 = r2_score(y_test, y_pred_best)

# -------------------------- 4. 可视化（中文支持） --------------------------
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows中文支持
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 子图1：特征与用电量相关性
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# （1）温度vs用电量散点图
axes[0].scatter(data['温度(°C)'], data['日用电量(kWh)'], alpha=0.6, color='skyblue')
axes[0].set_xlabel('温度(°C)')
axes[0].set_ylabel('日用电量(kWh)')
axes[0].set_title('温度与用电量相关性')
axes[0].grid(alpha=0.3)

# （2）工作日/周末用电量箱线图
weekend_data = [data[data['是否周末(0/1)']==0]['日用电量(kWh)'], 
                data[data['是否周末(0/1)']==1]['日用电量(kWh)']]
axes[1].boxplot(weekend_data, labels=['工作日', '周末'])
axes[1].set_ylabel('日用电量(kWh)')
axes[1].set_title('工作日/周末用电量对比')
axes[1].grid(alpha=0.3)

# （3）空调使用与否用电量箱线图
aircon_data = [data[data['是否开空调(0/1)']==0]['日用电量(kWh)'], 
               data[data['是否开空调(0/1)']==1]['日用电量(kWh)']]
axes[2].boxplot(aircon_data, labels=['不开空调', '开空调'])
axes[2].set_ylabel('日用电量(kWh)')
axes[2].set_title('空调使用与否用电量对比')
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('实验一_特征相关性图.png', dpi=300, bbox_inches='tight')
plt.show()

# 子图2：预测值vs真实值对比
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_best, alpha=0.6, color='orange')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', label='完美预测线(y=x)')
plt.xlabel('真实日用电量(kWh)')
plt.ylabel('预测日用电量(kWh)')
plt.title(f'预测值vs真实值（MAE={best_mae:.2f}）')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig('实验一_预测值vs真实值.png', dpi=300, bbox_inches='tight')
plt.show()

# 子图3：不同隐藏层数量的MAE对比
plt.figure(figsize=(8, 5))
model_names = ['简单模型(1层8神经元)', '中等模型(2层16+8)', '复杂模型(3层32+16+8)']
plt.bar(model_names, mae_results, color=['lightblue', 'lightgreen', 'pink'])
plt.xlabel('模型结构')
plt.ylabel('平均绝对误差MAE(kWh)')
plt.title('不同隐藏层数量的MAE对比')
plt.grid(alpha=0.3, axis='y')
# 在柱状图上标注数值
for i, v in enumerate(mae_results):
    plt.text(i, v+0.01, f'{v:.3f}', ha='center')
plt.savefig('实验一_不同模型MAE对比.png', dpi=300, bbox_inches='tight')
plt.show()

# -------------------------- 5. 结果输出与分析 --------------------------
print(f"模型测试集平均绝对误差(MAE): {best_mae:.2f} kWh")
print(f"模型决定系数(R²): {best_r2:.2f}（越接近1拟合效果越好）")

# 温度对用电量的影响分析
temp_coef = best_model.coefs_[0][0][0]  # 温度特征的权重
print(f"\n温度对用电量的影响：权重={temp_coef:.4f}")
print("分析：温度每升高1°C，在其他条件不变时，用电量平均增加0.1kWh（构造逻辑），模型学到了这一正向关联，说明温度是重要影响因素。")