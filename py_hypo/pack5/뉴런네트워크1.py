# Perceptron : 인공 신경망 

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from pack5.nn1_clas import MyPerceptron

df = pd.read_csv('../testdata/iris.data')

print(df.head(3))
print(df.corr())
x = df.iloc[0:100,[0,2]].values #setosa ,versicolor sepal.length, petal.length를씀
print(x[:2])
y = df.iloc[0:100,4].values
print(y[:2],np.unique(y))
y = np.where(y=='Iris-setosa',-1,1)
print(y)

# 시각화 
# plt.scatter(x[:50,0], x[:50,1], c ='red' , marker='o',label='setosa')
# plt.scatter(x[50:100,0], x[50:100,1], c='blue',marker='x',label='versicolor')
# plt.xlabel('sepa.lenth')
# plt.ylabel('petal.length')
# plt.legend(loc='upper left')
# plt.show()

pmodel = MyPerceptron(eta=0.01,n_iter =10)
pmodel.fit(x, y)
print(pmodel.predict(x))

print()
new_x = [[5.1,1.4],[2.1,7.4]]
print(pmodel.predict(new_x))

# Perceptron 의 반복에 따른 대비 오차 시각화
plt.plot(range(1,len(pmodel.errors_)+1),pmodel.errors_,marker='o')
plt.show() # 여섯번째 이후에 에러가 최소화되면서 수렴이 됨, 샘플을 제대로 분류를 시작하는 모델 완성




