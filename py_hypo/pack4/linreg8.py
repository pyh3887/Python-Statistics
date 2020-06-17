'''
다향회귀
선형 가정이 어긋날 경우(정규성 만족못할경우) 다항식 항을 추가한 후 다항회귀 모델을 작성후 사용
'''
import numpy as np
import matplotlib.pyplot as plt

x = np.array([1,2,3,4,5])
y = np.array([4,2,1,4,7])

#plt.scatter(x, y)
#plt.show()
    
#  선형 회귀모델
from sklearn.linear_model import LinearRegression
x=x[:,np.newaxis]  # 차원확대
#print(x)

model = LinearRegression().fit(x, y)
yfit = model.predict(x)
print(yfit)  #  단순선형회귀 모델에 의한 예측값 [2.  2.8 3.6 4.4 5.2]


from sklearn.metrics import mean_squared_error
import numpy as np

lin_mse = mean_squared_error(y, yfit)
lin_rmse = np.sqrt(lin_mse)

print("평균 제곱 오차 : ", lin_mse)
print("평균 제곱근 편차 : ", lin_rmse)

'''
plt.scatter(x, y)
plt.plot(y,yfit,c='red')
plt.show()
'''

#  모델에 유연성을 주기위해 다항식 특징을 추가
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=3, include_bias=False)  #  degree = 열의 개수 , include_bias = 편향이 한쪽으로 치우침
xx = poly.fit_transform(x)
print(xx)

'''
[[  1.   1.   1.]
 [  2.   4.   8.] ....
x 를 나타내는 열과 x2(제곱)을 나타내는 두번째열과 x3(세제곱)을 나타내는 세번째열로 구성이됨, 이를 통해 선형회귀를 수행 
'''

model2 = LinearRegression().fit(xx, y)
yfit2 = model2.predict(xx)
print(yfit2)

plt.scatter(x, y)
plt.plot(x,yfit2,c='red')
plt.show()

lin_mse = mean_squared_error(y, yfit2)
lin_rmse = np.sqrt(lin_mse)

print("평균 제곱 오차 : ", lin_mse)
print("평균 제곱근 편차 : ", lin_rmse)











