import scipy.stats as stats
import pandas as pd

# 1. 用read_excel读取Excel文件（需要openpyxl库支持）
data = pd.read_excel(r'F:\My-study\Data collection and preprocessing\NO3-work\dataclean\data.xlsx')

# 2. 数据预处理：去除缺失值（避免计算均值/标准差出错）
data = data.dropna(subset=['value'])  # 只保留value列非空的行
data['value'] = pd.to_numeric(data['value'], errors='coerce')  # 确保value列为数值类型
data = data.dropna(subset=['value'])  # 去除转换失败的行

# 3. 计算均值和标准差
u = data['value'].mean()
std = data['value'].std()

# 4. KS检验（正态性检验）
statistic, p_value = stats.kstest(data['value'], 'norm', (u, std))
print(f"KS检验统计量: {statistic}, p值: {p_value}")
