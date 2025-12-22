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
import intel_extension_for_pytorch as ipex  # Intel GPU扩展
import openvino.runtime as ov
from openvino.preprocess import PrePostProcessor
from openvino.runtime import Layout, Type
import onnx


# ===================== 1. 加载LFW人脸数据集 =====================
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


# ===================== 2. 数据预处理 + 划分数据集 =====================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)

# 适配模型输入格式：(样本数, 通道数, 高, 宽)，Intel GPU要求float32
X_train = X_train.reshape(-1, 1, 64, 64).astype(np.float32)
X_test = X_test.reshape(-1, 1, 64, 64).astype(np.float32)


# ===================== 3. 定义轻量CNN模型 + Intel Arc GPU训练 =====================
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


# 数据集类（适配Intel Arc GPU）
class FaceDataset(Dataset):
    def __init__(self, X, y, device):
        self.X = torch.tensor(X, dtype=torch.float32).to(device)
        self.y = torch.tensor(y, dtype=torch.long).to(device)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# 检测Intel Arc GPU设备
def get_intel_device():
    if torch.xpu.is_available():
        return torch.device("xpu"), torch.xpu.get_device_name(0)
    else:
        raise RuntimeError("未检测到Intel Arc GPU！请确保已安装Intel oneAPI和IPEX")

device, device_name = get_intel_device()
print(f"使用Intel GPU设备：{device_name}")


# 初始化模型、损失函数、优化器（迁移到Intel GPU）
model = FaceCNN(num_classes).to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# 加载数据集（迁移到Intel GPU）
train_dataset = FaceDataset(X_train, y_train, device=device)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
epochs = 20


# Intel GPU加速训练（IPEX优化）
model, optimizer = ipex.optimize(model, optimizer=optimizer, dtype=torch.float32)
print("开始Intel Arc GPU训练轻量CNN模型...")
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


# 保存Intel GPU训练后的模型（迁移到CPU）
model = model.to("cpu")
torch.save(model.state_dict(), "face_cnn_intel_gpu.pth")
print("Intel Arc GPU训练完成，模型已保存为 face_cnn_intel_gpu.pth")


# ===================== 4. NPU加速推理 =====================
# 初始化OpenVINO核心，检查NPU设备
core = ov.Core()
available_devices = core.available_devices
print("可用设备列表：", available_devices)

npu_device = None
for dev in available_devices:
    if "NPU" in dev or "Neural" in dev:
        npu_device = dev
        break

if not npu_device:
    raise RuntimeError("未检测到Intel NPU设备！请确保硬件/驱动支持")


# 重新加载模型（CPU环境），转ONNX（静态形状）
model = FaceCNN(num_classes)
model.load_state_dict(torch.load("face_cnn_intel_gpu.pth", map_location="cpu"))
model.eval()

# 虚拟输入（静态形状，适配NPU）
dummy_input = torch.randn(1, 1, 64, 64, dtype=torch.float32)
onnx_model_path = "face_cnn_intel_gpu.onnx"

# 导出ONNX
torch.onnx.export(
    model, dummy_input, onnx_model_path,
    input_names=["input"], output_names=["output"],
    opset_version=12,
    do_constant_folding=True,
    export_params=True
)
print(f"PyTorch模型已转为ONNX格式：{onnx_model_path}")


# 验证ONNX模型
onnx_model = onnx.load(onnx_model_path)
onnx.checker.check_model(onnx_model)
print("ONNX模型验证通过，无语法错误")


# 加载ONNX模型并编译到NPU
ov_model = core.read_model(model=onnx_model_path)

# 预处理配置
ppp = PrePostProcessor(ov_model)
ppp.input("input").tensor().set_layout(Layout("NCHW")).set_element_type(Type.f32)
ov_model = ppp.build()

# 编译模型到NPU
print(f"正在将模型编译到NPU设备：{npu_device}")
compiled_npu_model = core.compile_model(model=ov_model, device_name=npu_device)
print("模型编译完成，NPU加速就绪")


# NPU推理（批量处理测试集）
input_tensor = compiled_npu_model.input(0)
output_tensor = compiled_npu_model.output(0)
y_pred = []
print(f"开始NPU推理，测试集总量：{len(X_test)}")

for i in range(len(X_test)):
    sample = X_test[i:i+1]  # 保持4D维度
    result = compiled_npu_model([sample])[output_tensor]
    pred_label = np.argmax(result, axis=1)[0]
    y_pred.append(pred_label)
    
    if (i+1) % 100 == 0:
        print(f"已完成 {i+1}/{len(X_test)} 个样本推理")

y_pred = np.array(y_pred)


# ===================== 5. 模型评估 =====================
accuracy = accuracy_score(y_test, y_pred)
print("\n===== LFW人脸数据集 NPU加速分类结果 =====")
print(f"模型：轻量CNN（Intel Arc GPU训练） + NPU加速（{npu_device}）")
print(f"测试集准确率：{accuracy:.4f}")
print("分类报告：")
labels_to_show = list(range(20))
print(classification_report(
    y_test, y_pred,
    labels=labels_to_show,
    target_names=target_names[:20],
    zero_division=0
))
print("混淆矩阵（前10x10）：")
cm = confusion_matrix(y_test, y_pred)
print(cm[:10, :10])


# ===================== 6. 可视化 =====================
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 1. 混淆矩阵可视化（前10个类别）
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
    img_vector = X_test[idx].reshape(-1)
    img_vector = scaler.inverse_transform([img_vector])[0].reshape(64, 64)
    plt.subplot(1, 10, i+1)
    plt.imshow(img_vector, cmap='gray')
    true_name = target_names[y_test[idx]]
    pred_name = target_names[y_pred[idx]]
    plt.title(f"真实：{true_name[:4]}...\n预测：{pred_name[:4]}...", fontsize=8)
    plt.axis('off')
plt.tight_layout()
plt.show()



# pip install torch==2.0.0+cpu torchvision==0.15.1+cpu intel-extension-for-pytorch==2.0.100 -f https://developer.intel.com/ipex-whl-stable