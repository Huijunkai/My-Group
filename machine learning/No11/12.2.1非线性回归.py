# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 13:16:05 2024

@author: Administrator
"""

import numpy as np 
#导入 PyTorch 框架库
import torch 
from torch.autograd import Variable 
import matplotlib.pyplot as plt 
from torch import nn 
import torch.nn.functional as F 
#构造数据
a,b,c=2,5,6 #设置参数
x = torch.linspace(-8,8,100) 
x_train = x.unsqueeze(1) 
y = a* x.pow(2) - b *x + c 
y += 20*(torch.rand(x.size())-0.5) 
y_train = y.unsqueeze(1) 
#定义神经网络类并实例化神经网络对象
class NLR(nn.Module): 
    def __init__(self,n_feature,n_hidden,n_output): 
        super(NLR, self).__init__() 
        self.hidden = nn.Linear(n_feature,n_hidden) #隐层
        self.predict = nn.Linear(n_hidden,n_output) #输出层
    def forward(self, x): 
        x = self.hidden(x) #由输入层到隐层
        x_act = F.relu(x) #ReLU 激活
        y = self.predict(x_act) #由隐层到输出层
        return y 
net = NLR(1,10,1) #输入层与输出层维度为 1，隐层维度为 10 
#设置损失函数
Loss = nn.MSELoss() #采用标准差损失
Opt = torch.optim.SGD(net.parameters(), lr=1e-3) #设置优化器
#神经网络训练
T = 5000 #设定迭代次数
X = Variable(x_train) 
Y = Variable(y_train) 
#设置绘图输出模式
plt.ion() 
plt.show() 
for epoch in range(T): 
    Y_pred = net(X) # 前向传播
    L = Loss(Y_pred, Y) #计算误差
    Opt.zero_grad() #梯度清零
    L.backward() #误差反传
    Opt.step() #更新参数
 #输出中间结果
    if (epoch==0) | ((epoch+1) % 10 == 0): 
        print('Epoch[{}/{}], Loss:{:.4f}'.format(epoch+1, T, L.item())) 
 #展示中间结果
        plt.cla() 
        plt.plot(X.numpy(), Y.numpy(), 'ro') 
        plt.plot(X.numpy(), Y_pred.detach().numpy(), c='b', lw=4) 
        plt.text(-2,100,'Loss:%.4f'%L.item(), fontdict={'size':25, 'color': 'green'}) 
        plt.pause(0.1) 
#神经网络测试
net.eval() 
Y_pred = net(Variable(x_train)) 
plt.plot(x_train.numpy(), y_train.numpy(), 'ro', label='Noisy Points') 
plt.plot(x_train.numpy(), Y_pred.detach().numpy(), c='b', lw=2, label='Fitted Line[n_hidden: 10]') 
plt.legend(loc='upper left') 
plt.show() 
