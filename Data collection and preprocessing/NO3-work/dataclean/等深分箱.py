import numpy as np                  #导入numpy
import math                         #导入math
salary=np.array([2200,2300,2400,2500,2500,2800,3000,3200,3500,3800,4000,4500,4700,4800,4900,5000])  #定义数据

depth=salary.reshape(int(salary.size/4),4)  #等深分箱
print("等深分箱")                          #打印等深分箱
print(depth)                               #输出等深分箱后的数据

mean_depth=np.full((depth.shape[0],depth.shape[1]),0)  #初始化mean_depth
for i in range(0,depth.shape[0]):                      #使用均值进行平滑
    for j in range(0,depth.shape[1]):
        mean_depth[i][j]=depth[i].mean()
print("等深分箱——均值平滑技术")        #打印分割符
print(mean_depth)                     #输出均值平滑结果

median_depth=np.full((depth.shape[0],depth.shape[1]),0) #初始化median_depth
for i in range(0,depth.shape[0]):                       #使用中值进行填充
    for j in range(0,depth.shape[1]):
        median_depth[i][j]=np.median(depth[i])
print("等深分箱——中值平滑技术")        #打印分隔符
print(median_depth)                   #输出中值平滑结果

edgeLeft = np.arange(depth.shape[0])  #定义左边界
edgeRight=np.arange(depth.shape[0])   #定义右边界
edge_depth=np.full((depth.shape[0],depth.shape[1]),0) #初始化edge_depth
for i in range(0,depth.shape[0]):     #遍历等深箱行
    edgeLeft[i]=depth[i][0]           #第i行左边界
    edgeRight[i]=depth[i][-1]         #第i行右边界
    for j in range(0,depth.shape[1]): #遍历等箱列
        if(j==0):                     #第一列即左边界
            edge_depth[i][j]=depth[i][0]  #赋值
        if(j==3):                     #最后一列即右边界
            edge_depth[i][j]=depth[i][3]  #赋值
        else:
            # 判断距离左边界近还是距离右边界近
            if(math.pow((edgeLeft[i]-depth[i][j]),2)>math.pow((edgeRight[i]-depth[i][j]),2)):
                edge_depth[i][j]=edgeRight[i]  #赋予右边界值
            else:
                edge_depth[i][j]=edgeLeft[i]   #赋予左边界值
print("等深分箱法——边界值平滑")        #打印分隔符
print(edge_depth)                     #输出边界值平滑结果