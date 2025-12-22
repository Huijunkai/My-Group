import torch
# 检查是否支持 ROCm
print(torch.cuda.is_available())  # 输出 True 则表示成功
print(torch.version.hip)  # 输出 ROCm 版本号