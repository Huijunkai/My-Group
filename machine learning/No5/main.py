import numpy as np

import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import make_blobs

x, y = make_blobs(n_samples=50, centers=[[1,1],[2,2]], cluster_std=[0.3,0.2], random_state=10)
model = SVC(C=10.0, kernel='linear')
model.fit(x,y)
plt.figure()
plt.scatter(x[y==0,0],x[y==0,1],s=50,c='r',linewidths=1,edgecolors='k',label='Class one')
plt.scatter(x[y==0,0],x[y==0,1],s=50,c='g',linewidths=1,edgecolors='k',label='Class two')
x_ = np.linspace(np.min(x[:,0]), np.max(x[:,0]), 50)
y_ = np.linspace(np.min(x[:,1]), np.max(x[:,1]), 50)
yy, xx = np.meshgrid(y_,x_)
xy = np.vstack([xx.ravel(), yy.ravel()]) .T
DF = model.decision_function(xy).reshape(xx.shape)
plt.contour(xx, yy, DF, colors='k',levels = [-1, 0, 1],alpha=0.4,linestyles=['--','-','--'])
plt.scatter(model.support_vectors_[:,0],model.support_vectors_[:,1], s =100,linewidths=1,facecolors='b');
plt.legend(loc='best')
plt.grid(True)
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()