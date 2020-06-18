# PCA(주성분) 분석 - 차원 축소의 일종(이미지 크기 축소 , 데이터 압축 - 국어 + 영어 > 어문, 노이즈 제거 )

# 비지도 학습 중 하나 

# IRIS DATA의 차원 축소(독립변수 갯수 축소 : 데이터 압축)
from sklearn.decomposition import PCA
import pandas as pd 
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import seaborn as sns

plt.rc('font', family ='malgun gothic')

iris = load_iris()
n = 10 
x = iris.data[:n, :2] # sepal.length ,sepal.width
print('차원 축소 전 x : ', x)

plt.plot(x.T , 'o:')
plt.xticks(range(4),['꽃받침길이','꽃받침너비'])
plt.xlim(-0.5,2)
plt.ylim(2.5,6)
plt.title('아이리스 크기 특성')
plt.legend(['표본{}'.format(i+1) for i in range(n)])
plt.show() # 두 개의 데이터는 크기 변동이 비슷하게 움직임

ax = sns.scatterplot(0,1, data=pd.DataFrame(x), s= 100, color='.2', marker='s')
for i in range(n):
    ax.text(x[i,0],x[i,1] + 0.03, '표본{}'.format(i+1))

plt.xlabel('꽃받침길이')
plt.ylabel('꽃받침너비')
plt.title('아이리스 크기  특성(2차원)')
plt.axis('equal')
plt.grid()
plt.show()

#PCA 수행 (차원축소 : 근사)
pca1 = PCA(n_components =1 )
x_low = pca1.fit_transform(x) # 특징 행렬을 낮은 차원의 근사행렬로 변환
print('x_low :' , x_low) #

x2 = pca1.inverse_transform(x_low)
print('차원 축소 후 x2' , x2 )
print(x_low[7])
print(x2[7,:])

sns.scatterplot(0, 1 ,data = pd.Dataframe(x) , s = 100 ,color='.2', marker ='s')


for i in range(n):
    d = 0.03 if x[i,1] > x2[i,1] else -0.04
    ax.text(x[1,0] - 0.065, x[i,1] + d , '표본{}'.format(i+1))
    plt.plot([x[i,0],x2[i,0]],[x[i,1],x2[i,1]], 'k--')


plt.plot(x2[:0],x2[:0])
    
plt.show()
    








