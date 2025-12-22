# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 13:11:13 2024

@author: Administrator
"""

#导入科学计算库与绘图库
import numpy as np 
import matplotlib.pyplot as plt 
#导入 PyTorch 框架库
import torch 
import torch.nn as nn 
from torch.autograd import Variable 
#根据指定斜率与截距的直线生成真实数据点
k,b = 2,5 
#真实数据点
x0 = np.arange(0,10,0.2) 
y0 = k*x0+5 
#生成噪声服从正态分布的观测数据点
yn = y0+np.random.normal(0,2,50) 
#构建用于 PyTorch 框架的数据
X_Train=torch.from_numpy(x0) 
Y_Train=torch.from_numpy(yn) 
X_Train=X_Train.unsqueeze(1) 
Y_Train=Y_Train.unsqueeze(1) 
X_Train=X_Train.type(torch.FloatTensor) 
Y_Train=Y_Train.type(torch.FloatTensor) 
#定义神经网络类并实例化神经网络对象
class LR(nn.Module): 
     def __init__(self, In, H, Out): 
         super(LR,self).__init__()
         self.linear1=nn.Linear(In, H)
         self.linear2=nn.Linear(H, Out) 
     def forward(self, x): 
        x1=self.linear1(x) 
        y=self.linear2(x1) 
        return y 
net=LR(1,2,1) #输入与输出维度为 1，隐层维度为 2 
Loss=nn.MSELoss() #采用标准差损失
Opt=torch.optim.SGD(net.parameters(),lr=1e-2) #设置优化器
#神经网络训练
T=1000 #设定训练次数
for epoch in range(T): 
    Y_=net(X_Train) #前向传播
    L=Loss(Y_Train,Y_) #计算误差
    Opt.zero_grad() #梯度清零
    L.backward() #误差反传
    Opt.step() #更新参数
    if (epoch==0) | ((epoch+1) % 10 == 0):
        print('Epoch[{}/{}], Loss:{:.4f}'.format(epoch+1, T, L.item())) 
#神经网络测试
net.eval() 
X = Variable(X_Train) 
Y_Pred = net(X) #预测出的 Y 值
Y_Pred = Y_Pred.data.numpy() #转换为 NumPy 数组格式
#绘图真实直线、观测数据点与预测的直线
plt.figure() 
plt.plot(x0,y0,'r',label='Real Line') #真实直线
plt.plot(x0,yn,'b.',label='Noisy Points') #观测数据点
plt.plot(x0,Y_Pred,'g',label='Predicted Line') #预测的直线
plt.legend(loc='upper left')