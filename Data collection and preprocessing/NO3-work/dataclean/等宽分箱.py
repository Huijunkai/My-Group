import numpy as np                  #导入numpy
import pandas as pd                 #导入pandas
import math                         #导入math

#初始化数据
salary = np.array([2200,2300,2400,2500,2500,2800,3000,3200,
                   3500,3800,4000,4500,4700,4800,4900,5000])

# 等宽分箱（right=False表示左闭右开区间）
x_cuts = pd.cut(salary, bins=4, right=False)
# 获取每个区间的数量，并按区间顺序排序（关键修复）
number = pd.value_counts(x_cuts).sort_index()  # 按区间顺序排序
# 转换为numpy数组，方便按索引访问
number_arr = number.values
rows = number_arr.max()                         # 取所有区间中最大的数据量
widthList = np.full([4, rows], 0)               # 定义初始化等宽箱
i = 0                                           # 定义数据索引指针

# 遍历每个箱，赋值数据（修复：使用number_arr数组访问）
for j in range(4):
    # 遍历当前箱的每一列
    for a in range(number_arr[j]):
        widthList[j][a] = salary[i]
        i += 1

print("等宽分箱")                          # 打印分隔符
print(widthList)                           # 输出等宽箱

# 均值平滑（修复：使用number_arr访问每个箱的数量）
mean_width = np.full([4, rows], 0)
for i in range(4):
    # 计算当前箱的均值（只计算有效数据）
    box_mean = int(widthList[i, :number_arr[i]].sum() / number_arr[i])
    for j in range(number_arr[i]):
        mean_width[i][j] = box_mean

print("\n等宽分箱——均值平滑")                # 打印分隔符
print(mean_width)                          # 输出均值等宽箱

# 中值平滑（修复：使用number_arr访问每个箱的数量）
median_width = np.full([4, rows], 0)
for i in range(4):
    # 计算当前箱的中值（只计算有效数据）
    box_median = np.median(widthList[i, :number_arr[i]]).astype(int)
    for j in range(number_arr[i]):
        median_width[i][j] = box_median

print("\n等宽分箱——中值平滑")                # 打印分隔符
print(median_width)                        # 输出中值等宽分箱

# 边界值平滑（修复：使用number_arr访问每个箱的数量）
edge_width = np.full([4, rows], 0)           # 边界值平滑，初始化

for i in range(4):
    # 获取当前箱的有效数据
    box_data = widthList[i, :number_arr[i]]
    if len(box_data) == 0:
        continue  # 防止空箱报错
    edgeLeft = box_data[0]                  # 左边界（第一个元素）
    edgeRight = box_data[-1]                # 右边界（最后一个元素）
    
    # 遍历当前箱的每一列
    for j in range(number_arr[i]):
        if j == 0:
            edge_width[i][j] = edgeLeft     # 第一列用左边界
        elif j == number_arr[i] - 1:
            edge_width[i][j] = edgeRight    # 最后一列用右边界
        else:
            # 判断距离哪个边界更近（用平方差比较，避免开根号）
            dist_left = (edgeLeft - box_data[j]) ** 2
            dist_right = (edgeRight - box_data[j]) ** 2
            edge_width[i][j] = edgeRight if dist_left > dist_right else edgeLeft

print("\n等宽分箱——边界值平滑")              # 打印分隔符
print(edge_width)                          # 输出等宽分箱