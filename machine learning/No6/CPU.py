import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import openvino.runtime as ov  # Intel NPU 推理框架
from openvino.preprocess import PrePostProcessor
from openvino.runtime import Layout, Type
import onnx  # 新增这行，导入onnx库


# ===================== 1. 加载LFW人脸数据集（保持原逻辑不变） =====================
lfw_path = r"F:\My-study\machine learning\No6\lfw_home\lfw_home\lfw_funneled"

X = []
y = []
target_names = []
label_map = {}

for person_name in os.listdir(lfw_path):
    person_dir = os.path.join(lfw_path, person_name)
    if not os.path.isdir(person_dir):
        continue
    img_files = os.listdir(person_dir)
    if len(img_files) < 5:
        continue
    if person_name not in label_map:
        label_map[person_name] = len(target_names)
        target_names.append(person_name)
    for img_file in img_files:
        img_path = os.path.join(person_dir, img_file)
        img = Image.open(img_path).convert('L')
        img = img.resize((64, 64))
        img_vector = np.array(img).flatten()
        X.append(img_vector)
        y.append(label_map[person_name])

X = np.array(X)
y = np.array(y)
num_classes = len(target_names)
print(f"数据集规模：{X.shape}，类别数：{num_classes}")

# ===================== 2. 数据预处理 + 划分数据集（保持原逻辑，确保float32） =====================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)

# 适配模型输入格式：(样本数, 通道数, 高, 宽)，NPU要求float32
X_train = X_train.reshape(-1, 1, 64, 64).astype(np.float32)
X_test = X_test.reshape(-1, 1, 64, 64).astype(np.float32)

# ===================== 3. 定义轻量CNN模型 + CPU训练（保持原逻辑） =====================
class FaceCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

# 数据集类
class FaceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# 模型训练
model = FaceCNN(num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

train_dataset = FaceDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
epochs = 10

print("开始CPU训练轻量CNN模型...")
model.train()
for epoch in range(epochs):
    total_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")

torch.save(model.state_dict(), "face_cnn.pth")
print("CPU训练完成，模型已保存为 face_cnn.pth")

# ===================== 4. NPU加速推理（核心修复部分） =====================
"""
关键修复：
1. 移除ONNX的dynamic_axes，使用静态形状（NPU不支持动态批次）
2. 确保输入维度完全匹配（batch_size可设为1或固定批次）
3. 优化推理逻辑，支持批量数据处理
4. 增加模型兼容性检查
"""
# 4.1 初始化OpenVINO核心，检查NPU设备
core = ov.Core()
available_devices = core.available_devices
print("可用设备列表：", available_devices)

# 检查是否存在NPU设备（不同环境可能显示为"Intel(R) Neural Processing Unit"或"NPU"）
npu_device = None
for device in available_devices:
    if "NPU" in device or "Neural" in device:
        npu_device = device
        break

if not npu_device:
    raise RuntimeError("未检测到Intel NPU设备！请确保：1. 硬件支持（Core Ultra系列）；2. 安装Windows 11 24H2+；3. 安装最新NPU驱动")

# 4.2 PyTorch转ONNX（使用静态形状，适配NPU）
onnx_model_path = "face_cnn.onnx"
model.eval()  # 切换到评估模式，避免Dropout等训练层影响

# 虚拟输入：使用固定批次（batch_size=1，NPU最稳定），维度严格匹配(1,1,64,64)
dummy_input = torch.randn(1, 1, 64, 64, dtype=torch.float32)

# 导出ONNX（移除dynamic_axes，指定静态形状）
torch.onnx.export(
    model, dummy_input, onnx_model_path,
    input_names=["input"], output_names=["output"],
    opset_version=12,  # NPU兼容的ONNX版本（12-14最佳）
    do_constant_folding=True,  # 折叠常量，减少NPU计算量
    export_params=True  # 导出模型参数（必需）
)
print(f"PyTorch模型已转为ONNX格式：{onnx_model_path}")

# 验证ONNX模型有效性
onnx_model = onnx.load(onnx_model_path)
onnx.checker.check_model(onnx_model)
print("ONNX模型验证通过，无语法错误")

# 4.3 加载ONNX模型并编译到NPU（关键步骤）
# 读取模型
ov_model = core.read_model(model=onnx_model_path)

# 预处理配置（确保输入格式与NPU要求完全一致）
ppp = PrePostProcessor(ov_model)
# 明确指定输入布局和数据类型（NPU严格要求）
ppp.input("input").tensor() .set_layout(Layout("NCHW")) .set_element_type(Type.f32)  # NPU只支持float32
# 应用预处理
ov_model = ppp.build()

# 编译模型到NPU（使用自动检测到的NPU设备名）
print(f"正在将模型编译到NPU设备：{npu_device}")
compiled_npu_model = core.compile_model(model=ov_model, device_name=npu_device)
print("模型编译完成，NPU加速就绪")

# 4.4 NPU推理（支持批量处理测试集，解决静态批次限制）
# 获取模型输入输出信息
input_tensor = compiled_npu_model.input(0)
output_tensor = compiled_npu_model.output(0)

# 批量推理逻辑（分批次处理，每个批次batch_size=1，适配静态形状）
y_pred = []
print(f"开始NPU推理，测试集总量：{len(X_test)}")

for i in range(len(X_test)):
    # 取单个样本（形状：(1,1,64,64)）
    sample = X_test[i:i+1]  # 保持4D维度，避免形状不匹配
    # 执行推理
    result = compiled_npu_model([sample])[output_tensor]
    # 获取预测类别
    pred_label = np.argmax(result, axis=1)[0]
    y_pred.append(pred_label)
    
    # 打印进度（每100个样本）
    if (i+1) % 100 == 0:
        print(f"已完成 {i+1}/{len(X_test)} 个样本推理")

y_pred = np.array(y_pred)

# ===================== 5. 模型评估（修复classification_report） =====================
accuracy = accuracy_score(y_test, y_pred)
print("\n===== LFW人脸数据集 NPU加速分类结果 =====")
print(f"模型：轻量CNN + NPU加速（{npu_device}）")
print(f"测试集准确率：{accuracy:.4f}")
print("分类报告：")
# 修复：添加labels参数，指定前20个类别的标签，与target_names[:20]长度匹配
labels_to_show = list(range(20))  # 前20个类别的标签（0~19）
print(classification_report(
    y_test, y_pred,
    labels=labels_to_show,  # 明确指定要评估的类别
    target_names=target_names[:20],  # 对应前20个类别的名称
    zero_division=0
))
print("混淆矩阵（前10x10）：")
cm = confusion_matrix(y_test, y_pred)
print(cm[:10, :10])  # 只显示前10个类别的混淆矩阵

# ===================== 6. 可视化（保持原逻辑） =====================
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 1. 混淆矩阵可视化（简化为前10个类别）
plt.figure(figsize=(8, 7))
cm_small = cm[:10, :10]
im = plt.imshow(cm_small, cmap='Blues')
plt.title("混淆矩阵（前10个类别，NPU加速）")
plt.xticks(range(10), [target_names[i][:5]+"..." for i in range(10)], rotation=45)
plt.yticks(range(10), [target_names[i][:5]+"..." for i in range(10)])
for x in range(10):
    for y_idx in range(10):
        plt.text(y_idx, x, cm_small[x, y_idx], ha='center', va='center', color='black')
plt.colorbar(im)
plt.tight_layout()
plt.show()

# 2. 样本图像可视化
plt.figure(figsize=(12, 3))
sample_indices = np.random.choice(len(X_test), 10, replace=False)
for i, idx in enumerate(sample_indices):
    # 还原为原始图像（反标准化）
    img_vector = X_test[idx].reshape(-1)  # (1,64,64)→(4096,)
    img_vector = scaler.inverse_transform([img_vector])[0].reshape(64, 64)
    plt.subplot(1, 10, i+1)
    plt.imshow(img_vector, cmap='gray')
    true_name = target_names[y_test[idx]]
    pred_name = target_names[y_pred[idx]]
    plt.title(f"真实：{true_name[:4]}...\n预测：{pred_name[:4]}...", fontsize=8)
    plt.axis('off')
plt.tight_layout()
plt.show()