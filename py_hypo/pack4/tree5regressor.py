# DecisionTreeRegressor , RandomForestRegressor ... 등의 모델로 연속형 자료 예측

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import  RandomForestRegressor
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_boston
from sklearn.metrics import r2_score

boston = load_boston()
#print(boston.DESCR)
dfx = pd.DataFrame(boston.data, columns= boston.feature_names)
dfy = pd.DataFrame(boston.target, columns = ['MEDV'])
df = pd.concat([dfx,dfy],axis=1)

cols = ['MEDV','RM','PTRATIO','LSTAT'] # 'MEDV'와 상관관계가 강화 열 일부 선택
# sns.pairplot(df[cols])
# plt.show()

x = df[['LSTAT']].values
y = df['MEDV'].values
print(x[:3])
print(y[:3])

#실습  1: DecisionTreeRegressor

model = DecisionTreeRegressor(max_depth=3).fit(x,y)
print('예측값 :', model.predict(x)[:3]) # [30.47142857 25.84701493 37.315625  ]
print('실제값 :', y[:3].ravel()) #[24.  21.6 34.7]

r2 = r2_score(y, model.predict(x))
print('설명력(결정계수) :' ,r2)  # 설명력(결정계수) : 0.6993833085636556

# 실습 2 : RandomForest
model2 = RandomForestRegressor(n_estimators= 1000 , criterion= 'mse' ).fit(x,y) # 예측에서는 mse
print('예측값:' , model2.predict(x)[:3])
print('설명력(결정계수) :' , y[:3].ravel())


r2r = r2_score(y, model.predict(x))
print('설명력(결정계수) : ', r2r)

#시각화 

plt.scatter(x,y,c='lightgray',label ='trian data')
plt.scatter(x,model2.predict(x), c= 'r',label='predict data')
plt.xlabel('LSTAT')
plt.ylabel('MEDV')
plt.legend()
plt.show() 

# 새값으로 예측 
import numpy as np

print(x[:3]) # [[4.98][9.14][4.03]
x_new = np.array([[10],[15],[1]])
print('예상 집 값 : ',model2.predict(x_new)) #[20.39295286 18.0398931  48.2515 ]























