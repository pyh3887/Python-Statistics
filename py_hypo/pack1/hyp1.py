# 분산 / 표준편차의 중요성 - 데이터의 치우침을 표현하는 대표적인 값 중 하나 

import scipy.stats as stats
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(1)
#기대값 : 어떤 확률을 가진 사건을 무한 반복할때 얻을 수 있는 값의 평균으로써 기대할수 있는 값 . 간단하게 평균이라 생각하자. 

print(stats.norm(loc=1,scale=2).rvs(10))

print()
centers = [1,1.5,2]
col = 'rgb'

std = 0.1 # 표준편차
data_1 = []
for i in range(3):
    data_1.append(stats.norm(loc=centers[i],scale=std).rvs(100))
    #print(data_1)
    plt.plot(np.arange(len(data_1[i])) + i * len(data_1[0]) ,data_1[i],'*')
    

plt.show()

std = 2 #표준편차

data_1 = []
for i in range(3):
    data_1.append(stats.norm(loc=centers[i],scale=std).rvs(100))
    #print(data_1)
    plt.plot(np.arange(len(data_1[i])) + i * len(data_1[0]) ,data_1[i],'*')
    
plt.show()

#표준편차는 클수록,분산은 작을수록좋다. 패턴을 알기 쉽기 때문에 평균값과의 거리를 나타냄




