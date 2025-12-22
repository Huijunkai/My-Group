# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 11:44:12 2024

@author: Administrator
"""

import numpy as np #导入科学计算库
#导入绘图库
import matplotlib.pyplot as plt
from sklearn.svm import SVC #导入SVC模块
from sklearn.datasets import make_blobs #导入make_blobs数据库
#构造两类线性可分样本
x, y = make_blobs(n_samples=50, centers=[[1,1],[2,2]],cluster_std=[0.3,0.2],random_state=10)
#构建SVM分类器模型
model= SVC(C=10, kernel='rbf',gamma=15)
#训练SVM分类器模型
model.fit(x,y)
#显示分类结果与分界线
plt.figure()
plt.scatter(x[y==0,0], x[y==0,1], s=50, c='r',linewidths=1,edgecolors='k', label='Class one')
plt.scatter(x[y==1,0], x[y==1,1], s=50, c='g',linewidths=1,edgecolors='k', label='Class two')
x_ = np.linspace(np.min(x[:,0]), np.max(x[:,0]), 50)
y_ = np.linspace(np.min(x[:,1]), np.max(x[:,1]), 50)
yy, xx = np.meshgrid(y_, x_)
xy = np.vstack([xx.ravel(), yy.ravel()]).T
DF = model.decision_function(xy).reshape(xx.shape)
plt.contour(xx, yy, DF, colors='k',levels=[-1, 0, 1], alpha=0.4,linestyles=['--', '-', '--'])
# 显示支持向量
plt.scatter(model.support_vectors_[:, 0],model.support_vectors_[:, 1], s=100, linewidth=1, facecolors='b');
plt.legend(loc='best')
plt.grid(True)
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()



