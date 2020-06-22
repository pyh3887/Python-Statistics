# 비지도 학습 중 하나인 군집(cluster) 분석

# 게층적 군집분석 : 응집형, 분리형

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster._feature_agglomeration import AgglomerationTransform

plt.rc('font', family = 'malgun gothic')

np.random.seed(123)

var = ['x','y']
labels = ['점0','점1','점2','점3','점4']
x = np.random.random_sample([5,2]) * 10
df = pd.DataFrame(x,columns = var, index = labels)
print(df)

#plt.scatter(x[:,0], x[:,1],c='b',marker='o',s=50)
#plt.grid(True)
#plt.show()

from scipy.spatial.distance import pdist,squareform
distmatrix = pdist(df , metric = 'euclidean')
print('distmatrix :' ,distmatrix)

row_dist = pd.DataFrame(squareform(distmatrix), columns=labels, index = labels)
print(row_dist)

from scipy.cluster.hierarchy import linkage # 응집형 계층적 클러스터링 수행

row_cluster = linkage(distmatrix, method = 'ward')
df = pd.DataFrame(row_cluster , columns = ['클러스터1', '클러스터2', '거리','클러스터 멤버수'],index = ['클러스터1%d'%(i+1) for i in range(row_cluster.shape[0])])
print(df)

from scipy.cluster.hierarchy import dendrogram
row_dend = dendrogram(row_cluster, labels = labels)
#plt.show()

print()
#병합 군집 알고리즘 모델 
from sklearn.cluster import AgglomerativeClustering
ac = AgglomerativeClustering(n_clusters = 3 , affinity='euclidean', linkage='ward')
labels = ac.fit_predict(x)
print('군집 분류 결과 :' ,labels)

a= labels.reshape(-1, 1)
print(a)
x1 = np.hstack([x,a])
print(x1)

x_0 = x1[x1[:,2]==0, :]
x_1 = x1[x1[:,2]==1, :]
x_2 = x1[x1[:,2]==2, :]

print(x_0)

plt.scatter(x_0[:,0],x_0[:,1])
plt.scatter(x_1[:,0],x_1[:,1])
plt.scatter(x_2[:,0],x_2[:,1])
plt.legend(['cluster0','cluster1','cluster2'])
plt.show()




