# 비 지도학습의 비계층적 군집분석 : k-means k의 갯수를 정해주는 것이 가장 큰 관건

import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs # clustering 연습용 dataset
from networkx.utils.decorators import random_state

print(make_blobs)

x, y = make_blobs(n_samples = 150, n_features=2, centers=3 ,cluster_std= 0.5, shuffle=True, random_state=0)

print(x.shape)
#print(x)
#print(y)

plt.scatter(x[:,0],x[:,1])
plt.grid(True)
        
plt.show()

from sklearn.cluster import KMeans
kmodel = KMeans(n_clusters = 3, init = 'random', random_state=0)
#kmodel = KMeans(n_clusters = 3, init = 'k-means++', random_state=0)

pred = kmodel.fit_predict(x)
print('pred :' , pred)

plt.scatter(x[pred == 0, 0], x[pred == 0, 1], c='red',marker='o',s=50,label='cluster1')
plt.scatter(x[pred == 1, 0], x[pred == 1, 1], c='yellow',marker='s',s=50,label='cluster2')
plt.scatter(x[pred == 2, 0], x[pred == 2, 1], c='blue',marker='v',s=50,label='cluster3')
plt.legend()
plt.grid(True)
plt.show()

#그래프를 보면 클러스터 1 ~3 에 속하는 데이터들의 실루엣 계수가 0으로 된 값이 아무것도 없으며, 실루엣 계수의 평균이 0.7보다 크므로 잘 분류된걸 확인할수 있다.
import numpy as np

from sklearn.metrics import silhouette_samples

from matplotlib import cm

X,y = make_blobs(n_samples=150, n_features=2, centers=3, cluster_std=0.5,shuffle=True, random_state=0)
km = KMeans(n_cluster=3 , random_state=0)
y_km = km.fit_predict(X)

def plotSilhouette(x, pred):

    cluster_labels = np.unique(pred)

    n_clusters = cluster_labels.shape[0]   # 클러스터 개수를 n_clusters에 저장

    sil_val = silhouette_samples(x, pred, metric='euclidean')  # 실루엣 계수를 계산

    y_ax_lower, y_ax_upper = 0, 0

    yticks = []

    for i, c in enumerate(cluster_labels):

        # 각 클러스터에 속하는 데이터들에 대한 실루엣 값을 수평 막대 그래프로 그려주기

        c_sil_value = sil_val[pred == c]

        c_sil_value.sort()

        y_ax_upper += len(c_sil_value)

       

        plt.barh(range(y_ax_lower, y_ax_upper), c_sil_value, height=1.0, edgecolor='none')

        yticks.append((y_ax_lower + y_ax_upper) / 2)

        y_ax_lower += len(c_sil_value)

   

    sil_avg = np.mean(sil_val)         # 평균 저장

    plt.axvline(sil_avg, color='red', linestyle='--')  # 계산된 실루엣 계수의 평균값을 빨간 점선으로 표시

    plt.yticks(yticks, cluster_labels + 1)

    plt.ylabel('클러스터')

    plt.xlabel('실루엣 개수')

    plt.show() 

'''

그래프를 보면 클러스터 1~3 에 속하는 데이터들의 실루엣 계수가 0으로 된 값이 아무것도 없으며, 실루엣 계수의 평균이 0.7 보다 크므로 잘 분류된 결과라 볼 수 있다.

'''

X, y = make_blobs(n_samples=150, n_features=2, centers=3, cluster_std=0.5, shuffle=True, random_state=0)

km = KMeans(n_clusters=3, random_state=0) 

y_km = km.fit_predict(X)


plotSilhouette(X, y_km)
        












