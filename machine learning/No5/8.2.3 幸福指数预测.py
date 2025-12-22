# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 11:52:23 2024

@author: Administrator
"""

import pandas as pd #导入数据结构与分析库
#导入绘图库
from matplotlib import pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split #导入数据划分模型
from sklearn.svm import SVR #导入SVM回归模块
from sklearn.metrics import r2_score #导入R2分数模块
from sklearn.preprocessing import StandardScaler #导入数据标准化模块
#加载数据
# data_path = 'D://529857988//wangwei@zknu.cn//data_ml//part//'
data = pd.read_csv('machine learning/No5/happyscore_income.csv',encoding=u'gbk')
print('样本数与特征数',data.shape)
# 特征相关性分析
corr = data[data.columns[0:]].corr()
plt.figure(figsize=(10,8))
ax = sns.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=False, annot=True,fmt='.1f')
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=30,
    horizontalalignment='right'
)
#plt.title("特征相关性", fontsize=20)
plt.show()
# 幸福指数与平均收入之间相关性
plt.figure(2)
sns.regplot(x=data['avg_income'],y=data['happyScore'])
# 幸福指数与中等收入之间相关性
plt.figure(3)
sns.regplot(x=data['median_income'],y=data['happyScore'])
# 幸福指数与收入不均衡指标之间相关性
plt.figure(4)
sns.regplot(x=data['income_inequality'],y=data['happyScore'])
# 构建幸福指数预测模型
x = data[['avg_income','median_income']]
y = data['happyScore']
# 数据标准化处理
scaler = StandardScaler()
x_ = scaler.fit_transform(x)
# 将数据划分为训练数据与测试数据
x_train, x_test, y_train, y_test = train_test_split(x_,y,test_size=0.5)
# 支持向量回归模型构建[C=1]
SV = SVR(kernel='rbf',C=2, gamma=1)
# 支持向量回归模型训练 
SV = SV.fit(x_train,y_train)
# 支持向量回归模型评估
print('训练样本拟合优度(C=2):',r2_score(y_train,SV.predict(x_train)))
print('测试样本拟合优度(C=2):',r2_score(y_test,SV.predict(x_test)))
# 支持向量回归模型构建[参数设置2]
SV = SVR(kernel='rbf',C=0.1, gamma=1)
# 支持向量回归模型训练 
SV = SV.fit(x_train,y_train)
# 支持向量回归模型评估
print('训练样本拟合优度(C=0.1):',r2_score(y_train,SV.predict(x_train)))
print('测试样本拟合优度(C=0.1):',r2_score(y_test,SV.predict(x_test)))
