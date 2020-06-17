# 선형 회귀 : 독립 변수(연속), 종속 변수(연속)
# 회귀 분석은 각 데이터에 대한 잔차 제곱합이 최소가 되는 선형 회귀식을 도출하는 방법
# 맛보기

import statsmodels.api as sm
from sklearn.datasets import make_regression
import numpy as np

np.random.seed(12)

print('방법 1 -------------------------------------')
# 방법1 : make_regression을 이용 - 모델이 만들어 지진 않는다.
x, y, coef = make_regression(n_samples=50, n_features=1, bias=100, coef=True)
print(x[:5])    # sample 독립 변수 자료
print(y[:5])    # sample 종속 변수 자료
print(coef)     # 기울기
# y = wx + b    89.47430739278907(coef) * x + 100(bias)
yhat = 89.47430739278907 * -1.70073563 + 100
print('yhat :', yhat)
yhat = 89.47430739278907 * -0.67794537 + 100
print('yhat :', yhat)

new_x = 0.5
pred_yhat = 89.47430739278907 * new_x + 100
print('pred_yhat :', pred_yhat)

xx = x
yy = y

print('\n방법 2(가장 많이 쓰는 방법) -------------------------------------')
# 방법1 : LinearRegression을 이용 - 모델이 만들어  진다.
from sklearn.linear_model import LinearRegression

model = LinearRegression()
print(model)
fit_model = model.fit(xx, yy)   # fit()으로 데이터를 학습하여 최적의 모형을 추정한다.
print(fit_model.coef_)      # 기울기    89.47430739
print(fit_model.intercept_)      # 절편   100.0

# 예측 값 구하기1 : 수식을 직접 적용
new_x = 0.5
pred_yhat2 = fit_model.coef_ * new_x + fit_model.intercept_
print('pred_yhat2 :', pred_yhat2)

# 예측 값 구하기2 : predict()
pred_yhat3 = fit_model.predict([[new_x]])
print('pred_yhat3 :', pred_yhat3)

# 예측 값 구하기3 : predict()
x_new, _, _ = make_regression(n_samples=5, n_features=1, bias=100, coef=True)
print(x_new)
pred_yhat4 = fit_model.predict(x_new)
print('pred_yhat4 :', pred_yhat4)

print('\n방법 3(가장 많이 쓰는 방법) -------------------------------------')
# 방법3 : LinearRegression 을 이용 - 모델이 만들어  진다.
import statsmodels.formula.api as smf
import pandas as pd
print(xx.shape)
x1 = xx.flatten()   # 차원 축소
print(x1[:5], x1.shape)
y1 = yy

data = np.array([x1, y1])
df = pd.DataFrame(data.T)       # .T 행렬 변환
df.columns = ['x1', 'y1']
print(df.head(3))

model2 = smf.ols(formula='y1 ~ x1', data=df).fit()
print(model2.summary())     # OLS Regression Results
# 절편 100.0000, 기울기 89.4743

print(df[:3])
print(model2.predict()[:3])     # 기존 데이터에 대한 예측 값

# 새로운 값에 대한 예측 결과
print('x1 :', x1[:2])   # x1 : [-1.70073563 -0.67794537]
newx = pd.DataFrame({'x1':[-1.9 -0.8]})
predy = model2.predict(newx)
print('예측 결과 : \n', predy)
