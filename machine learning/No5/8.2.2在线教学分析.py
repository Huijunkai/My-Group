# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 11:49:11 2024

@author: Administrator
"""

import pandas as pd #导入数据结构与分析库
import numpy as np #导入科学计算库
from sklearn.svm import SVC #导入SVM分类模块
from sklearn.model_selection import train_test_split #导入数据划分模型
from sklearn.model_selection import GridSearchCV #导入网格式参数调优模块
from sklearn.preprocessing import StandardScaler #导入特征标准化库
from sklearn.decomposition import PCA #导入主成分分析模块
#加载数据
data = pd.read_csv('machine learning/No5/students_adaptability_level_online_education.csv',encoding=u'gbk')
#数据编码
edu_encoding = {
    'Gender' : { 
        'Boy': 1 ,
        'Girl':  0
    },
    'Education Level' : {
        'University' : 2,
        'College'    : 1,
        'School'     : 0
    },
    'IT Student' : {
        'No'  : 0,
        'Yes' : 1
    },
    'Financial Condition' : {
        'Poor' : 0,
        'Mid'  : 1,
        'Rich' : 2
    },
    'Network Type' : {
        '4G' : 2,
        '3G' : 1,
        '2G' : 0
    },
    'Internet Type'   :{
        'Wifi':        1,
        'Mobile Data': 0
    },
    'Adaptivity Level' : {
        'Low'      : 0,
        'Moderate' : 1,
        'High'     : 2
    },
    'Device'  : {
        'Tab': 1, 
        'Mobile' : 0, 
        'Computer': 2
    }
}
for column in data:
    if column in edu_encoding.keys():
        try:
            data[column] = data[column].apply( lambda x : edu_encoding[column][x] )
        except:
            print(f"Skipped {column}")
#显示数据基本信息（样本数与特征数）
print('数据基本信息:',data.shape) 
x = data.drop('Adaptivity Level' , axis= 1)
y = data['Adaptivity Level']
#数据标准化
scaler = StandardScaler()
x_ = scaler.fit_transform(x)
#将数据划分为训练数据与测试数据
x_train, x_test, y_train, y_test = train_test_split(x_,y,test_size=0.3)
# 最优超参数组合列表
param_grid = [
        {'kernel': ['linear'], 'C': [1, 5, 10, 20, 30, 50, 100]},
        {'kernel': ['poly'], 'C': [1], 'degree': [2, 3, 4]},
        {'kernel': ['rbf'], 'C': [1, 5, 10, 20, 30, 50, 100], 'gamma':[1, 0.1, 0.01, 0.001]}
        ]
#构建SVM分类器
model = SVC()
#通过交叉验证确定最优参数
grid_search = GridSearchCV(model,param_grid,cv=3)
grid_search.fit(x_train, y_train)
#显示参数优化结果
print('最优模型:',grid_search.best_estimator_) 
print('最优参数:',grid_search.best_params_) 
print('最高分值:',grid_search.best_score_)
#最优SVM分类器
opt_model = model.set_params(**grid_search.best_params_)
#或者opt_model = grid_search.best_estimator_
#训练SVM分类器
opt_model.fit(x_train,y_train)
#输出预测精度
print('预测精度:',opt_model.score(x_test,y_test))  
# 主成分分析
pca = PCA(n_components=0.95, whiten=True).fit(x_train)
x_train_pca = pca.transform(x_train)
x_test_pca = pca.transform(x_test)
#构建SVM分类器
model = SVC()
#通过交叉验证确定最优C值
grid_search = GridSearchCV(model,param_grid,cv=5)
grid_search.fit(x_train_pca, y_train)
#显示参数优化结果
print('最优模型(PCA):',grid_search.best_estimator_) 
print('最优参数(PCA):',grid_search.best_params_) 
print('最高分值(PCA):',grid_search.best_score_)
# 采用最优参数进行模型训练与测试
opt_model = grid_search.best_estimator_
opt_model.fit(x_train_pca,y_train)
# 输出测试精度
print("预测精度(PCA):", opt_model.score(x_test_pca,y_test)) 
