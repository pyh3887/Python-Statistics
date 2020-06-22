#*** Regressor 롤 모델비교

import pandas as pd 
import numpy as np 
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score

adver = pd.read_csv('../testdata/Advertising.csv')
print(adver.head(3))

x = np.array(adver.loc[:,'tv':'newspaper'])
y = np.array(adver.sales)

print(x[:2])
print(y[:2])

print('KNeighborsRegressor ---------------')

kmodel = KNeighborsRegressor(n_neighbors = 3).fit(x,y) # knn으로 modeling
print(kmodel)
kpred = kmodel.predict(x) 
print('kpred:', kpred[:3]) #예측값
print('kmodel r2:', r2_score(y,kpred)) # 설명력   0.968012077694316 상관계수 제곱

print('\n LinearRegression--------------')
lmodel = LinearRegression().fit(x, y)

lpred = lmodel.predict(x)
print('lpred:', lpred[:3])
print('lmodel r2:', r2_score(y,lpred)) # 설명력  0.89721063


print('\n RandomForestRegressor--------------') #트리를 이어줌

rmodel = RandomForestRegressor(n_estimators=100, criterion='mse').fit(x,y) # mse 평균 제곱 
print(rmodel)
rpred = rmodel.predict(x)
print('rpred :', rpred[:3])
print('rmodel r2 :', r2_score(y,rpred)) # 설명력 0.997389


print('\n XGBRegressor') # 에러난 부분에서 수정해가면서 처리
xmodel = XGBRegressor(n_estimators=100).fit(x,y)
print(xmodel)
xpred = xmodel.predict(x)
print('rpred : ', xpred[:3])
print('xmodel r2 :', r2_score(y,xpred)) # 설명력 0.99999966












