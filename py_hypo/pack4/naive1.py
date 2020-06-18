# Navie Bayes Classification : 베이즈 정리를 적용한 확률 분류기 

# 텍스트 분류에 효과적 - 스팸메일 , 게시판 카테고리 등의 분류에 많이 사용됨 

# ML에서는 fueature가 주어졌을 때 label 의 확률을 구하는데 사용

#P(L|feature) = P(fea

from sklearn.naive_bayes import GaussianNB
import numpy as np 
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder

x = np.array([1,2,3,4,5])
x = x[:, np.newaxis]
y = np.array([1,3,5,7,9])
model = GaussianNB().fit(x,y)
print(model)
pred = model.predict(x)
print(pred)

#새값 
new_x = np.array([[0.5], [7.1], [12.0]])
new_pred = model.predict(new_x)
print(new_pred)

print('--OneHotEncoding 희소 행렬의 일종 -----------------')

x= np.array([1,2,3,4,5])
x = np.eye(x.shape[0])
print(x)
y = np.array([1,3,5,7,9])

model = GaussianNB().fit(x,y)
print(model)
pred = model.predict(x)
print(pred)