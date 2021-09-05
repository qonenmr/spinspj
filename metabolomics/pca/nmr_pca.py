# -*- coding: utf-8 -*-
"""
Created on Sat Feb  6 23:25:01 2021

@author: ThinkPad
"""


import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from command import spinspj 

filePath = r".\system\data\pyexample\ST000101.txt"
origin_data = spinspj.loadmetabolicdata(filePath,1,3);
metaclasses = spinspj.loadmetaclass(filePath,1,2);
samplenames = spinspj.loadmetasample(filePath,1,0,1);

data = np.array(origin_data)
data = np.transpose(data)
rownum = data.shape[0]
colnum = data.shape[1]

for k in range(rownum):
    m=np.mean(data[k,:])
    stddata = np.std(data[k,:])
    data[k,:] = (data[k,:] - m)/stddata

data2 = np.transpose(data)    
pca1=PCA(n_components=8)
newData=pca1.fit(data2)
print(pca1.explained_variance_ratio_)
print(pca1.explained_variance_)
eig=pca1.explained_variance_
xcomp = [1, 2, 3, 4, 5, 6, 7, 8]

plt.figure(1)
plt.bar(xcomp, pca1.explained_variance_ratio_[0:8], label='', color='blueviolet')
plt.xlabel('principal component')
plt.ylabel('R2X')
plt.title('principal component analysis')
plt.show()


plt.figure(2)
fitted_data = pca1.fit_transform(data.T)        # numpy.ndarray
plt.scatter(fitted_data[:, 0], -fitted_data[:, 1],marker='x')

for i in range(1,6):
    plt.annotate("Mixture B" + str(i), xy=(fitted_data[:, 0][i-1]+2, -fitted_data[:, 1][i-1]), color='black')
for i in range(1,6):
    plt.annotate("Mixture A" + str(i), xy=(fitted_data[:, 0][i+4]+2, -fitted_data[:, 1][i+4]), color='red')


plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.title('score scatter plot')
plt.show()
print('over')
